#!/bin/bash
# 在双机 GRPO 的推理机上做中途评测（推理机空出来的算力），低优先级排队，不抢训练采样：
#   bash scripts/remote/q9b_mideval.sh <名字> <policy LoRA 目录 | policy_v0>
# 在三套验收原文上生成（同 run_r3full.sh 的评测口径），再用 9B 金标评测器（rm1__rm1_gold_9b）逐句打分，
# 写到 GRPO 的评测目录 $O/eval/<名字>_{a,b,c}.jsonl / _9b.json，和训完后的评测同名同格式。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; PAIR=${PAIR:-q9b}
NAME=${1:?name}; SRC=${2:?adapter dir or policy_v0}
O=${O:-outputs/rt/core/loop2_q9b_p9r9}; E=$O/eval; mkdir -p $E
URL=http://127.0.0.1:6006/v1
export VLLM_API_KEY=$(cat $FS/runs/pair/$PAIR/key) EVAL_PRIORITY=${EVAL_PRIORITY:-10} PYTHONHASHSEED=0
log() { echo "$(date '+%F %T') mideval $NAME $*"; }
load() {   # 名字 适配器目录：去掉视觉塔参数后热加载（已加载就跳过）
  curl -s -H "Authorization: Bearer $VLLM_API_KEY" $URL/models | grep -q "\"$1\"" && return 0
  python scripts/strip_visual_lora.py "$2" $A/lora_eval/$1 > /dev/null || return 1
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$1\", \"lora_path\": \"$A/lora_eval/$1\"}" $URL/load_lora_adapter)
  [ "$code" = 200 ]
}
mkdir -p $A/lora_eval
C9=rm1__rm1_gold_9b
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$C9/eval/report.json'))['temperature'])")
load evalgold $FS/adapters/$C9-9b/adapter || { log "load evaluator FAILED"; exit 1; }
if [ "$SRC" = policy_v0 ]; then M=policy_v0
else
  [ -f $SRC/policy/adapter_model.safetensors ] && SRC=$SRC/policy
  M=eval_$NAME; load $M $SRC || { log "load policy FAILED"; exit 1; }
fi
log "generate with $M (priority $EVAL_PRIORITY)"
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url $URL --model $M --out $E/${NAME}_a.jsonl --concurrency 16 &
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url $URL --model $M --out $E/${NAME}_b.jsonl --concurrency 16 &
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300c/answers.jsonl --temperature 0.7 --top_p 0.95 --url $URL --model $M --out $E/${NAME}_c.jsonl --concurrency 16 &
wait
for s in a b c; do [ -s $E/${NAME}_$s.jsonl ] || { log "gen $s FAILED"; exit 1; }; done
J="["; for s in a b c; do J+="{\"model\":\"evalgold\",\"temp\":$T,\"answers\":\"$E/${NAME}_$s.jsonl\",\"out\":\"$E/${NAME}_${s}_9b.json\"},"; done
echo "${J%,}]" > $E/jobs_mid_$NAME.json
python -u -m ragtruth.critic_score_multi --url $URL --jobs $E/jobs_mid_$NAME.json --concurrency 64 || { log "score FAILED"; exit 1; }
# 评测用的策略 LoRA 卸掉（评测器留着，后面的检查点还要用）
[ "$M" != policy_v0 ] && curl -s -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' -d "{\"lora_name\": \"$M\"}" $URL/unload_lora_adapter
log "MIDEVAL_DONE"
