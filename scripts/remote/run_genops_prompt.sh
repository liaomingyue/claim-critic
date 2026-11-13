#!/bin/bash
# 只靠提示词的错误合成：不微调，基座模型（2B、9B）直接按操作卡片改写，三种提示词写法（卡片 / 加步骤 / 加步骤再先写打算）。
# 同样 70 条留出回答 × 每种操作各采 1 条 → samples_ops_p<写法>_<模型>.jsonl。服务脚本要求至少挂一个 LoRA，挂一个现成的但不用它（请求的模型名是 base）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/genops_prompt.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
for v in "mc|minicpm5-2b|gen-ops-mc" "9b|qwen3.5-9b|rm1__core_9b-9b"; do
  IFS='|' read -r tag model dummy <<< "$v"
  source $S/model_env.sh $model || { log "model $model FAILED"; continue; }
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MODEL_LOCAL --log $A/vllm-p$tag.log dummy=$FS/adapters/$dummy/adapter >> $LOG 2>&1 || { log "serve $tag FAILED"; continue; }
  for st in card strict plan; do
    [ -s $C/samples_ops_p${st}_$tag.jsonl ] && continue
    python -u -m ragtruth.gen_ops sample --model base --tokenizer $MODEL_LOCAL --card $([ $st = card ] || echo --style $st) --out samples_ops_p${st}_$tag.jsonl >> $LOG 2>&1 || log "sample $st $tag FAILED"
    log "SAMPLED p${st}_$tag $(grep -a GENOPS_SAMPLE $LOG | tail -1 | cut -c1-300)"
  done
  free_gpu
done
log "GENOPS_PROMPT_DONE"
