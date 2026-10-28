#!/bin/bash
# 补跑 e2 评测：克隆后 torch 残留文件导致 vLLM 起不来，清理后在 e3 训练旁边用 8001 端口单独起一个小 vLLM。
cd /root/autodl-tmp/critic_model
export OPENAI_API_KEY=x
LOG=/root/autodl-tmp/exp.log
nohup vllm serve /root/autodl-tmp/ckpt/sft-0.6b-e2 --served-model-name sft-0.6b-e2 --max-model-len 8192 \
  --gpu-memory-utilization 0.3 --port 8001 > /root/autodl-tmp/vllm-sft-0.6b-e2.log 2>&1 &
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q sft-0.6b-e2 && break; sleep 10; done
python -m ragtruth.eval --model sft-0.6b-e2 --base_url http://127.0.0.1:8001/v1 --concurrency 128 --no_think \
  --style verdict --max_tokens 64 --out outputs/rt/eval/sft-0.6b-e2 > /root/autodl-tmp/eval-sft-0.6b-e2.log 2>&1
pkill -f '[p]ort 8001'
echo "$(date '+%F %T') side-evaluated e2" >> $LOG.side
