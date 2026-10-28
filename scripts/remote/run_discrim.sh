#!/bin/bash
# 真实性判别器（9B）：建数据 → 训练 → 在留出集上评测准确率。日志 /root/autodl-tmp/discrim.log，结束写 DISCRIM_DONE。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/discrim.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/root/autodl-tmp/models/Qwen3.5-9B; CK=/root/autodl-tmp/ckpt/discrim-9b
MM='{"image":0,"video":0}'
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }

python -m ragtruth.discrim build --out_dir outputs/rt/discrim >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "train discrim"
python -u -m train.sft --config configs/rt/discrim_9b.yaml --mode coldstart > /root/autodl-tmp/discrim-9b.log 2>&1 \
  || { log "train FAILED"; exit 1; }
cp $B/tokenizer_config.json $B/tokenizer.json $B/vocab.json $B/merges.txt $B/chat_template.jinja \
   $B/preprocessor_config.json $B/video_preprocessor_config.json $CK/ 2>/dev/null
stop
nohup vllm serve $CK --served-model-name discrim --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 \
  --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-discrim.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q discrim && break; sleep 10; done
python -m ragtruth.discrim score --rows outputs/rt/discrim/heldout.jsonl --out outputs/rt/discrim/heldout_scores.jsonl --eval \
  >> $LOG 2>&1 || { log "eval FAILED"; exit 1; }
stop
log "DISCRIM_DONE"
