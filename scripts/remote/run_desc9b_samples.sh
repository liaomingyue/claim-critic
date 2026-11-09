#!/bin/bash
# 2026-11-08 用户：9B 给生成器 v2 的采样写意思层面描述（新版 sev 输入的一部分，另配完整逐词改动）；不调 Claude、不筛选。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/desc9b.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
pkill -f '[v]llm serve'; sleep 5
source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-desc9b.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.gen_v2 describe --model base --samples samples_v2_mc.jsonl samples_v2_9b.jsonl --out samples_v2_desc9b.jsonl >> $LOG 2>&1 || log "describe FAILED"
pkill -f '[v]llm serve'
grep -a DESC9B $LOG | tail -1
log "DESC9B_SAMPLES_DONE"
