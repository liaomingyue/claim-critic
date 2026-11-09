#!/bin/bash
# 2026-11-08 用户：生成器训练数据里的改动描述用 9B 写意思层面的自然描述（大段改写结合错误类型概括），再让 Claude 抽查。
# Qwen3.5-9B 基座（服务名 base；不是评测用的 9B critic）给种子集错误对写描述 → outputs/rt/core/seed_v2_desc9b.jsonl。只写描述、不筛选数据。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/desc9b.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
true
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
pkill -f '[v]llm serve'; sleep 5
source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-desc9b.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.gen_v2 describe --model base --out ${OUT:-seed_v2_desc9b.jsonl} >> $LOG 2>&1 || log "describe FAILED"
pkill -f '[v]llm serve'
grep -a DESC9B $LOG | tail -1
log "DESC9B_DONE ${OUT:-seed_v2_desc9b.jsonl}"
