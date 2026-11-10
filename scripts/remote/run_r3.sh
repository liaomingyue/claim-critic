#!/bin/bash
# 按「合并句子导致 r2 漏判」的方向改，再重跑。
#   ① merge_aug：种子集 Claude 核实过的错句 / 改对句拼成长句、列表句（1000 对 → 2000 行），加到 r2 训练集 → rm1__core_r3；run_mcrm（校准 + RAGTruth++）
#   ② r3 打分：三套验收（critic_cap3，本地算）+ cont2 起点 / 第 100 步回答（按句长的盲区召回，和 r2 比）
#   ③ 用 r3 当奖励接着训：同 cont2 的起点（adapters/mc-grpo-r3_core_cont/step100）100 步 → ARM=cont3，9B 评测
#   ④ r3 给 cont3 回答打分（盲区召回）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/r3.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
waitjobs() { for p in $(jobs -p); do [ "$p" = "$KA" ] || wait $p; done; }
if [ ! -f outputs/rt/rm/rm1__core_r3/eval/report.json ]; then
  python -u -m ragtruth.merge_aug --base rm1__core_r2 --name rm1__core_r3 --max_pairs 1000 >> $LOG 2>&1 || { log "merge FAILED"; exit 1; }
  grep -a MERGE_AUG $LOG | tail -1 | cut -c1-600
  RM=rm1__core_r3 GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/rm1__core_r3/eval/report.json ] || { log "train FAILED"; exit 1; }
fi
log "R3_TRAINED $(python -c "import json;r=json.load(open('outputs/rt/rm/rm1__core_r3/eval/report.json'));print({k:r.get(k) for k in ('temperature','test','calib')})" | cut -c1-500)"
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__core_r3/eval/report.json'))['temperature'])")
score_r3() {   # $1 = 目录，$2.. = 前缀
  local K=$1; shift
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r3s.log c3=$FS/adapters/rm1__core_r3-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; return 1; }
  for m in "$@"; do for s in a b c; do [ -f $K/${m}_${s}_r3.json ] || python -u -m ragtruth.critic_score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model c3 --temp $T --out $K/${m}_${s}_r3.json >> $LOG 2>&1 & done; done
  [ "$K" = outputs/rt/core/loop2_cont2/eval ] && for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
    [ -f $d/critic_rm1__core_r3.json ] || python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c3 --temp $T --out $d/critic_rm1__core_r3.json >> $LOG 2>&1 &
  done
  waitjobs; free_gpu
}
score_r3 outputs/rt/core/loop2_cont2/eval v0 s100
python outputs/blind_recall.py outputs/rt/core/loop2_cont2/eval r3 v0 s100 | tee -a $LOG
log "R3_SCORED"
# ③ 接着训
[ -f $FS/adapters/mc-grpo-r3_core_cont3/step100/adapter_model.safetensors ] || [ -f $FS/adapters/mc-grpo-r3_core_cont3/step100/policy/adapter_model.safetensors ] || \
  START=$FS/adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r3 ARM=cont3 CODE=$PWD bash $S/run_loop2cont.sh >> $LOG 2>&1
grep -q R3FULL_DONE $A/r3full_core_cont3.log || { log "cont3 FAILED"; exit 1; }
free_gpu
score_r3 outputs/rt/core/loop2_cont3/eval v0 s50 s100
python outputs/blind_recall.py outputs/rt/core/loop2_cont3/eval r3 v0 s50 s100 | tee -a $LOG
K=outputs/rt/core/loop2_cont3/eval; K2=outputs/rt/core/loop2_cont2/eval
python -m ragtruth.pol9b_cmp --arm start=$K:v0 --arm c50=$K:s50 --arm c100=$K:s100 --arm r2c100=$K2:s100 --arm r2start=$K2:v0 --sets a b c --base start --out $K/pol9b_cmp.json > /dev/null 2>&1 || log "cmp FAILED"
log "R3_DONE"
