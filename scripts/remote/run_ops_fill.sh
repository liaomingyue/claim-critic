#!/bin/bash
# 闭环后半段：按操作合成的错误补 critic 盲区（ragtruth/ops_fill.py）。
#   ① 错误生成器（GEN，默认 st8 第 100 步）在训练集原文的回答上每个回答 × 每种操作采 1 条 → 把关模型 + 当前 critic 打分
#   ② 挑盲区（三项达标且 critic < 0.5）；每种操作 30 条交 Claude（本地），等 labels.jsonl
#   ③ 等的时候先训 all 组（盲区全量、不核实）；标签到了再训 cl（Claude 留下的）和 nc（同样条数、不看 Claude）
#   ④ 三个新 critic 在三套验收回答上打分（本地用 scripts/analysis/critic_round_cmp.py 对比）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis; LOG=$A/ops_fill.log
RD=${RD:-of1}; E=${E:-st8}; TAG=tr_$E; CR=${CR:-rm1__core_r2_bfcl}; NAME=${NAME:-rm1__core_$RD}; B=outputs/rt/ops_fill/$RD
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
GEN=${GEN:-$FS/adapters/gen-ops-grpo-$E/step100/policy}
for f in $R/$CR/train.jsonl $R/$CR/calib.jsonl $R/$CR/meta.json $R/$CR/eval/report.json $C/genpool_train.jsonl $GEN/adapter_model.safetensors; do [ -s $f ] || { log "OPSFILL FAILED: missing $f"; exit 1; }; done
for d in grpo300 grpo300b grpo300c; do [ -s outputs/rt/minicpm/$d/answers.jsonl ] || { log "OPSFILL FAILED: missing answers $d"; exit 1; }; done

if [ ! -s $B/items.jsonl ]; then
  free_gpu
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-of-sample.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/$CR-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$GEN >> $LOG 2>&1 || { log "OPSFILL FAILED: serve"; exit 1; }
  python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 1 --pool genpool_train.jsonl --out samples_ops_$TAG.jsonl >> $LOG 2>&1 || { log "OPSFILL FAILED: sample"; exit 1; }
  python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t $CR), \"disc\": $(t rm2__d1)}" $TAG >> $LOG 2>&1 || { log "OPSFILL FAILED: score"; exit 1; }
  free_gpu
  python -u -m ragtruth.ops_fill select --tag $TAG --round $RD >> $LOG 2>&1 || { log "OPSFILL FAILED: select"; exit 1; }
  gzip -kf $C/samples_ops_$TAG.jsonl $C/samples_ops_${TAG}_rm4.jsonl
fi
log "$(grep -a '^OPS_FILL_SELECT' $LOG | tail -1 | cut -c1-1500)"
log "OPSFILL_READY_FOR_CLAUDE"

train() {   # 建好数据的组训成 critic（已有报告的跳过）
  [ -f $R/$1/eval/report.json ] && return 0
  free_gpu; RM=$1 GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f $R/$1/eval/report.json ] || { log "OPSFILL FAILED: train $1"; return 1; }
  log "TRAINED $1 T=$(t $1)"
}
python -u -m ragtruth.ops_fill build --round $RD --base $CR --name $NAME --arms all >> $LOG 2>&1 || { log "OPSFILL FAILED: build all"; exit 1; }
train ${NAME}all || exit 1
for i in $(seq 1 480); do [ -s $B/labels.jsonl ] && break; sleep 60; done
ARMS="all"
if [ -s $B/labels.jsonl ]; then
  python -u -m ragtruth.ops_fill build --round $RD --base $CR --name $NAME --arms cl nc >> $LOG 2>&1 || { log "OPSFILL FAILED: build cl nc"; exit 1; }
  log "BUILT $(grep -a '^OPS_FILL_BUILD' $LOG | tail -1 | cut -c1-1500)"
  train ${NAME}cl && train ${NAME}nc && ARMS="all cl nc"
else
  log "no claude labels after 8h, scoring the all arm only"
fi
free_gpu
SERVE=""; for a in $ARMS; do SERVE="$SERVE $a=$FS/adapters/$NAME$a-mc/adapter"; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-of-score.log $SERVE >> $LOG 2>&1 || { log "OPSFILL FAILED: serve critics"; exit 1; }
for a in $ARMS; do
  PIDS=""
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
    [ -f $d/critic_$NAME$a.json ] && continue
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model $a --temp $(t $NAME$a) --out $d/critic_$NAME$a.json >> $LOG 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait $p; done
done
free_gpu
mkdir -p $FS/results/ops_fill_$RD && cp $LOG $B/*.jsonl $FS/results/ops_fill_$RD/ 2>/dev/null
for a in $ARMS; do mkdir -p $FS/results/ops_fill_$RD/$NAME$a; cp $R/$NAME$a/meta.json $R/$NAME$a/eval/report.json $FS/results/ops_fill_$RD/$NAME$a/; for d in grpo300 grpo300b grpo300c; do cp outputs/rt/minicpm/$d/critic_$NAME$a.json $FS/results/ops_fill_$RD/$NAME$a/critic_$d.json; done; done
log "OPSFILL_DONE $ARMS"
