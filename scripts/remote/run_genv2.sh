#!/bin/bash
# 错误改写模型第二版格式（ragtruth/gen_v2.py：输入原文 + 完整回答 + 9 个细类机制之一，输出第几句 + 逐词改动 + 改写句）。
#   ① 训 gen-v2-mc（MiniCPM5-2B）、gen-v2-9b（Qwen3.5-9B），数据 outputs/rt/core/seed_v2_train.jsonl（1411 条，种子集真实错误，本地建好随代码上传）
#   ② 两个模型在留出原文的 70 条回答（genpool_eval.jsonl）× 9 个机制 × 每个 2 条上采样 → samples_v2_{mc,9b}.jsonl（之后本地交 Claude 审可控性）
# 等同机的 critic 验证（run_critic_sev.sh）做完再开始。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/genv2.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q CRITIC_SEV_DONE $A/critic_sev.log 2>/dev/null; do sleep 30; done
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
train() {   # train <名字> <配置> <基座>
  local name=$1 cfg=$2 base=$3 CK=$A/ckpt/$1
  [ -f $FS/adapters/$name/adapter/adapter_model.safetensors ] && return 0
  free_gpu; mkdir -p configs/rt/exp
  sed -e "s|base_model: .*|base_model: $base|" -e "s|output_dir: .*|output_dir: $CK|" $cfg > configs/rt/exp/$name.yaml
  log "train $name"
  python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "$name FAILED: $(tail -n 3 $A/$name.log | cut -c1-300)"; return 1; }
  mkdir -p $FS/adapters/$name && rsync -a $CK/adapter $FS/adapters/$name/ && cp $A/$name.log $FS/adapters/$name/train.log
  log "TRAINED $name $(grep -ao "'train_loss': '[0-9.]*'" $A/$name.log | tail -n 1)"
}
sample() {  # sample <名字> <基座> <输出>
  [ -s $C/$3 ] && return 0
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $2 --log $A/vllm-$1.log gen=$FS/adapters/$1/adapter >> $LOG 2>&1 || { log "serve $1 FAILED"; return 1; }
  python -u -m ragtruth.gen_v2 sample --model gen --tokenizer $2 --pool genpool_eval.jsonl --n 2 --out $3 >> $LOG 2>&1 || log "sample $1 FAILED"
  free_gpu
  log "SAMPLED $1 $(grep -a GENV2_SAMPLE $LOG | tail -1 | cut -c1-400)"
}
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
train gen-v2-mc configs/rt/sft_gen_v2_mc.yaml $MC && sample gen-v2-mc $MC samples_v2_mc.jsonl
source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
train gen-v2-9b configs/rt/sft_gen_v2_9b.yaml $B9 && sample gen-v2-9b $B9 samples_v2_9b.jsonl
log "GENV2_DONE"
