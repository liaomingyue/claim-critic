#!/bin/bash
# 生成器中间检查点的留出评测（单卡）：每个检查点在留出回答 × 20 种操作上各采 4 条，用第二轮的把关模型 + critic 打分，和段终点同一套流程（run_genops_grpo9.sh 的 ev），方便挑最好的检查点。
#   CK="<标签>=<adapter 目录> ..."；检查点还没训出来就等（最多 4 小时）。输出 samples_ops_ev_<标签>.jsonl / _rm9v5.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis; LOG=$A/ckev9.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/models9b.sh; ensure_9b || { log "CKEV FAILED: 9b model"; exit 1; }; B9=$A/models/Qwen3.5-9B
GSEV=${G_SEV:-sev__v5}; GNAT=${G_NAT:-rm2__v5}; GOPM=${G_OPM:-opm__v2}; GCR=${G_CR:-rm1__core_of9all}; SUF=${GSUF:-rm9v5}
ad() { echo $FS/adapters/${1}_9b-9b/adapter; }
t() { python -c "import json;print(json.load(open('$R/${1}_9b/eval/report.json'))['temperature'])"; }
TJ="{\"sev4\": $(t $GSEV), \"nat4\": $(t $GNAT), \"opm\": $(t $GOPM), \"rm1\": $(t $GCR), \"disc\": $(t rm2__d2)}"
LORAS="sev4=$(ad $GSEV) nat4=$(ad $GNAT) opm=$(ad $GOPM) rm1=$(ad $GCR) disc=$(ad rm2__d2)"
log "temperatures $TJ"
for kv in ${CK:?}; do
  L=${kv%%=*}; G=${kv#*=}
  for i in $(seq 1 240); do [ -f $G/adapter_model.safetensors ] && break; [ $((i % 10)) = 1 ] && log "waiting for $G"; sleep 60; done
  [ -f $G/adapter_model.safetensors ] || { log "CKEV FAILED: missing $G"; continue; }; sleep 90   # 训练机可能还在写检查点
  [ -s $C/samples_ops_ev_${L}_$SUF.jsonl ] && { log "skip $L (done)"; continue; }
  free_gpu
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-ev-$L.log $LORAS g=$G >> $LOG 2>&1 || { log "ev serve $L FAILED"; continue; }
  [ -s $C/samples_ops_ev_$L.jsonl ] || python -u -m ragtruth.gen_ops sample --model g --tokenizer $B9 --card --style free --n 4 --out samples_ops_ev_$L.jsonl >> $LOG 2>&1 || { log "ev sample $L FAILED"; continue; }
  GENOPS_SUF=$SUF python -u $T/genops_score.py "$TJ" ev_$L >> $LOG 2>&1 || { log "ev score $L FAILED"; continue; }
  log "EV $L done"
done
free_gpu
log "CKEV_DONE"
