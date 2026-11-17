#!/bin/bash
# 第二轮补 critic（of2）：在 RAG 剩余显然错最集中的类别上多采样、补到一定数量，重训 critic，再接着训 RAG。全程不核实。
#   两组 critic，都是 基线 + 第一轮盲区全量（1,420 条）+ 针对性补充（本地挑好后传到 outputs/rt/ops_fill/of2/target_<组>.jsonl）：
#     zone  = 和盲区同一条规则（三项达标且 critic < 0.5），每种操作最多 300 条
#     clear = 把关模型判为显然错且 critic 没把握（< 0.8）
#   ① 建数据、训两个 critic，三套验收回答上打分  ② 各自当奖励，从 RAG-S3-bf 接着训 100 步（cont10 = zone，cont11 = clear），9B 评测
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; R=outputs/rt/rm; LOG=$A/of2.log
RD=${RD:-of2}; CR=${CR:-rm1__core_r2_bfcl}; NAME=${NAME:-rm1__core_$RD}; B=outputs/rt/ops_fill/$RD; ARMS="zone clear"
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
log "waiting for target files in $B"
for i in $(seq 1 120); do [ -s $B/target_zone.jsonl ] && [ -s $B/target_clear.jsonl ] && [ -f $B/READY ] && break; sleep 30; done
[ -f $B/READY ] || { log "OF2 FAILED: no target files"; exit 1; }
for a in $ARMS; do [ -s $R/$NAME$a/train.jsonl ] && [ -s $R/$NAME$a/meta.json ] || { python -u -m ragtruth.ops_fill build_target --round $RD --base $CR --name $NAME --arms $a >> $LOG 2>&1 || { log "OF2 FAILED: build $a"; exit 1; }; }; done
log "$(grep -a '^OPS_FILL_BUILD_TARGET' $LOG | tail -2 | cut -c1-400 | tr '\n' ' ')"
for a in $ARMS; do
  [ -f $R/$NAME$a/eval/report.json ] && continue
  free_gpu; RM=$NAME$a GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f $R/$NAME$a/eval/report.json ] || { log "OF2 FAILED: train $a"; exit 1; }
  log "TRAINED $NAME$a T=$(t $NAME$a)"
done
free_gpu
SERVE=""; for a in $ARMS; do SERVE="$SERVE $a=$FS/adapters/$NAME$a-mc/adapter"; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-of2-score.log $SERVE >> $LOG 2>&1 || { log "OF2 FAILED: serve critics"; exit 1; }
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
mkdir -p $FS/results/ops_fill_$RD
for a in $ARMS; do mkdir -p $FS/results/ops_fill_$RD/$NAME$a; cp $R/$NAME$a/meta.json $R/$NAME$a/eval/report.json $FS/results/ops_fill_$RD/$NAME$a/; for d in grpo300 grpo300b grpo300c; do cp outputs/rt/minicpm/$d/critic_$NAME$a.json $FS/results/ops_fill_$RD/$NAME$a/critic_$d.json; done; done
log "OF2_CRITICS_DONE"
ARMS="cont10=${NAME}zone cont11=${NAME}clear" CODE=$PWD bash $S/run_rag_s5.sh >> $LOG 2>&1 || { log "OF2 FAILED: rag"; exit 1; }
log "OF2_DONE"
