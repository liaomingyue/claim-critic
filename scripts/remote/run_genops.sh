#!/bin/bash
# 按错误操作清单下指令的错误合成模型（ragtruth/gen_ops.py：输入原文 + 完整回答 + 20 种操作之一，输出第几句 + 改写句 + 改动描述）。
#   ① 训 gen-ops-mc（MiniCPM5-2B），数据 outputs/rt/core/seed_ops_train.jsonl（种子集真实错误里的单一操作样本，本地建好随代码上传）
#   ② 在留出原文的 70 条回答（genpool_eval.jsonl）× 每种操作各采 1 条 → samples_ops_mc.jsonl（之后本地交 Claude 核实真错、实际操作、自然度）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/genops.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
name=gen-ops-mc; CK=$A/ckpt/$name
if [ ! -f $FS/adapters/$name/adapter/adapter_model.safetensors ]; then
  free_gpu; mkdir -p configs/rt/exp
  sed -e "s|base_model: .*|base_model: $MC|" -e "s|output_dir: .*|output_dir: $CK|" configs/rt/sft_gen_ops_mc.yaml > configs/rt/exp/$name.yaml
  log "train $name ($(wc -l < $C/seed_ops_train.jsonl) samples)"
  python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "$name FAILED: $(tail -n 3 $A/$name.log | cut -c1-300)"; exit 1; }
  mkdir -p $FS/adapters/$name && rsync -a $CK/adapter $FS/adapters/$name/ && cp $A/$name.log $FS/adapters/$name/train.log
  log "TRAINED $name $(grep -ao "'train_loss': '[0-9.]*'" $A/$name.log | tail -n 1)"
fi
free_gpu
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-$name.log gen=$FS/adapters/$name/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.gen_ops sample --model gen --tokenizer $MC --out samples_ops_mc.jsonl >> $LOG 2>&1 || { log "sample FAILED"; free_gpu; exit 1; }
free_gpu
log "SAMPLED $(grep -a GENOPS_SAMPLE $LOG | tail -1 | cut -c1-600)"
log "GENOPS_DONE"
