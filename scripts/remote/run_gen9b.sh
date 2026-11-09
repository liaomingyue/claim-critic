#!/bin/bash
# 2026-11-08 用户：是不是 2B 太差？用同一份种子集 D3 训 9B 错误改写模型，看造错和解释效果。
#   ① SFT：Qwen3.5-9B + LoRA，数据 outputs/rt/core/seed_sft_train.jsonl（同 gen-core-mc），配置 configs/rt/sft_gen_core_9b.yaml → adapters/gen-core-9b
#   ② 采样：在 2B sevpool 抽检过的同一批改写提示上（genpool_audit.jsonl，683 个，随代码上传）每个采 4 条 → outputs/rt/core/samples_9b_audit.jsonl
#      之后本地配对抽样、交 Claude 审（不在这里调 Claude）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen9b.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
CK=$A/ckpt/gen-core-9b
if [ ! -f $FS/adapters/gen-core-9b/adapter/adapter_model.safetensors ]; then
  free_gpu
  mkdir -p configs/rt/exp
  sed -e "s|base_model: .*|base_model: $B9|" -e "s|output_dir: .*|output_dir: $CK|" configs/rt/sft_gen_core_9b.yaml > configs/rt/exp/gen-core-9b.yaml
  log "train gen-core-9b ($(wc -l < $C/seed_sft_train.jsonl) rows)"
  python -u -m train.sft --config configs/rt/exp/gen-core-9b.yaml --mode coldstart > $A/gen-core-9b.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "gen-core-9b FAILED: $(tail -n 5 $A/gen-core-9b.log | cut -c1-300)"; exit 1; }
  mkdir -p $FS/adapters/gen-core-9b && rsync -a $CK/adapter $FS/adapters/gen-core-9b/ && cp $CK/sft_meta.json $FS/adapters/gen-core-9b/ 2>/dev/null
  cp $A/gen-core-9b.log $FS/adapters/gen-core-9b/train.log
fi
log "GEN9B_TRAINED $(grep -ao "'train_loss': '[0-9.]*'" $A/gen-core-9b.log $FS/adapters/gen-core-9b/train.log 2>/dev/null | tail -n 1)"
if [ ! -s $C/samples_9b_audit.jsonl ]; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-gen9b.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
  python -u -m ragtruth.core sample --model gen --tokenizer $B9 --pool genpool_audit.jsonl --out samples_9b_audit.jsonl --n 4 >> $LOG 2>&1 || { log "sample FAILED"; exit 1; }
  free_gpu
fi
grep -a CORE_SAMPLE $LOG | tail -1 | cut -c1-400
log "GEN9B_DONE"
