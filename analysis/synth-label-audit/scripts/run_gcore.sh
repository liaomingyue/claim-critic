#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-fs/code/critic_model || exit 1
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_SYNC_MODE=fs; unset LORA_SYNC_FS_DIR
A=/root/autodl-tmp; MC=$A/models/MiniCPM5-2B
INIT=/root/autodl-fs/adapters/gen-core-mc/adapter
SHAPE=${SHAPE:-gated}; STEPS=${STEPS:-5}; TAG=${TAG:-smoke}
O=outputs/rt/gentarget/gcore_$TAG; mkdir -p $O
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 4
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$m" -lt 1000 ] && break; sleep 3; done
echo "$(date '+%F %T') serve policy+RMs (shape=$SHAPE steps=$STEPS)"
bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $MC --mem 0.4 --rank 32 --log $A/vllm-gcore.log \
  policy_v0=$INIT rm1=/root/autodl-fs/adapters/rm1__core-mc/adapter rm2=/root/autodl-fs/adapters/rm2__coreerr-mc/adapter sev=/root/autodl-fs/adapters/sev__core-mc/adapter || { echo GCORE_SERVE_FAIL; exit 1; }
echo "$(date '+%F %T') start gen_grpo"
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT \
  --prompts outputs/rt/gentarget/prompts.jsonl --eval_prompts outputs/rt/gentarget/eval_prompts.jsonl \
  --reward target --div_mode place --err_mode ab --w_div 0.1 --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --raw_ratio 4 \
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 \
  --steps $STEPS --lr 5e-6 --n 16 --prompts_per_step 16 --eval_k 4 --save_every 50 \
  --vllm_url http://127.0.0.1:8000/v1 --temps rm1=rm1__core,rm2=rm2__coreerr,sev=sev__core \
  --shape $SHAPE --gate 0.5 --base_fail 0.1 --blind_lo 0 --blind_thr 0.5 --proc_producer 1 --pipeline 1 \
  --version_prefix core_ --out $O > $A/gcore_$TAG.log 2>&1 && echo GCORE_DONE || echo GCORE_FAILED
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
echo "$(date '+%F %T') GCORE_$TAG END"
