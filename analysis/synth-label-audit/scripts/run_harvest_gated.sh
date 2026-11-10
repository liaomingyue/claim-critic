#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-fs/code/critic_model || exit 1
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
A=/root/autodl-tmp; MC=$A/models/MiniCPM5-2B
GEN=outputs/rt/gentarget/gcore_gated/ckpt/step50/policy
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 4
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$m" -lt 1000 ] && break; sleep 3; done
# 子集: 2000 prompt
head -2000 outputs/rt/gentarget/prompts.jsonl > outputs/rt/gentarget/prompts_sub.jsonl
echo "$(date '+%F %T') serve gated-gen + rm1__core + rm2__coreerr + sev__core"
bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $MC --rank 32 --log $A/vllm-hvg.log \
  gen=$GEN rm1=/root/autodl-fs/adapters/rm1__core-mc/adapter rm2=/root/autodl-fs/adapters/rm2__coreerr-mc/adapter sev=/root/autodl-fs/adapters/sev__core-mc/adapter || { echo HV_SERVE_FAIL; exit 1; }
echo "$(date '+%F %T') harvest"
python -u -m ragtruth.gen_target_data harvest --url http://127.0.0.1:8000/v1 --model gen --inp prompts_sub.jsonl --samples 8 \
  --sev sev --sev_temp 1.75 --rm1 rm1 --rm1_temp 1.2 --rm2 rm2 --rm2_temp 1.6 --gate 0.5 --out harvest_gated.jsonl 2>&1 | tail -3
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
echo "$(date '+%F %T') HARVEST_GATED_DONE"
