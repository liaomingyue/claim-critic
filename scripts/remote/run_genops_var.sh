#!/bin/bash
# 错误合成模型（按操作清单下指令）的提示词 / 训练设置对照：每个版本训一个 2B 生成器，在同样 70 条留出回答上每种操作各采 1 条。
# 训练数据本地建好随代码上传（outputs/rt/core/seed_ops<tag>_train.jsonl）。版本：名字|数据后缀|采样时的提示词开关|训练轮数
#   ex    指令里附例子
#   ext   附例子 + 指令放在原文和回答之后
#   ext4  同 ext，训 4 轮
#   ext6s 同 ext，只用改对句和错句重合度 >= 0.6 的样本，训 4 轮（样本少一半）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/genops_var.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
for v in "ex|_ex|--examples|1" "ext|_ext|--examples --tail|1" "ext4|_ext|--examples --tail|4" "ext6s|_ext6|--examples --tail|4"; do
  IFS='|' read -r tag data flags ep <<< "$v"
  name=gen-ops-$tag-mc; CK=$A/ckpt/$name
  [ -s $C/samples_ops_$tag.jsonl ] && continue
  if [ ! -f $FS/adapters/$name/adapter/adapter_model.safetensors ]; then
    free_gpu; mkdir -p configs/rt/exp
    sed -e "s|base_model: .*|base_model: $MC|" -e "s|output_dir: .*|output_dir: $CK|" -e "s|seed_ops_train.jsonl|seed_ops${data}_train.jsonl|" -e "s|num_train_epochs: .*|num_train_epochs: $ep|" configs/rt/sft_gen_ops_mc.yaml > configs/rt/exp/$name.yaml
    log "train $name ($(wc -l < $C/seed_ops${data}_train.jsonl) samples, $ep epochs)"
    python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1
    [ -f $CK/adapter/adapter_model.safetensors ] || { log "$name FAILED: $(tail -n 3 $A/$name.log | cut -c1-300)"; continue; }
    mkdir -p $FS/adapters/$name && rsync -a $CK/adapter $FS/adapters/$name/ && cp $A/$name.log $FS/adapters/$name/train.log
    log "TRAINED $name $(grep -ao "'train_loss': '[0-9.]*'" $A/$name.log | tail -n 1)"
  fi
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-$name.log gen=$FS/adapters/$name/adapter >> $LOG 2>&1 || { log "serve $name FAILED"; continue; }
  python -u -m ragtruth.gen_ops sample --model gen --tokenizer $MC $flags --out samples_ops_$tag.jsonl >> $LOG 2>&1 || log "sample $name FAILED"
  free_gpu
  log "SAMPLED $tag $(grep -a GENOPS_SAMPLE $LOG | tail -1 | cut -c1-300)"
done
log "GENOPS_VAR_DONE"
