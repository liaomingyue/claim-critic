#!/bin/bash
# 多造样本扩充 sev 训练数据（2B、9B 都采）。生成器 = 定版 gen-v2（先句子后 edit），训练池 657 条回答 × 3 个按权重分配的机制 × 2 条。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sevaug.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
smp() {   # smp <适配器> <基座> <输出>
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $2 --log $A/vllm-aug.log gen=$FS/adapters/$1/adapter >> $LOG 2>&1 || { log "serve $1 FAILED"; return 1; }
  python -u -m ragtruth.gen_v2 sample --model gen --tokenizer $2 --pool genpool_train.jsonl --plan sevaug_plan.jsonl --n 2 --out $3 >> $LOG 2>&1 || log "sample $1 FAILED"
  log "SAMPLED $1 $(grep -a GENV2_SAMPLE $LOG | tail -1 | cut -c1-300)"
}
source $S/model_env.sh minicpm5-2b && smp gen-v2-mc $MODEL_LOCAL samples_aug_mc.jsonl
source $S/model_env.sh qwen3.5-9b && smp gen-v2-9b $MODEL_LOCAL samples_aug_9b.jsonl
free_gpu
log "SEVAUG_SAMPLE_DONE"
