#!/bin/bash
# machine2 跑 add 变体 GRPO(MC 在共享盘,输出到共享盘 gcore_add)
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-fs/code/critic_model || exit 1
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_SYNC_MODE=fs LORA_SYNC_FS_DIR=/root/autodl-tmp/lora_sync_add
A=/root/autodl-tmp; MC=/root/autodl-fs/models/MiniCPM5-2B
INIT=/root/autodl-fs/adapters/gen-core-mc/adapter
SHAPE=add; STEPS=50; TAG=add; WH2=0.6; WR2=0.3
O=outputs/rt/gentarget/gcore_$TAG; mkdir -p $O
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 4
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$m" -lt 1000 ] && break; sleep 3; done
echo "$(date '+%F %T') [m2-add] serve policy+RMs (shape=$SHAPE)"
bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $MC --mem 0.4 --rank 32 --log $A/vllm-gadd.log \
  policy_v0=$INIT rm1=/root/autodl-fs/adapters/rm1__core-mc/adapter rm2=/root/autodl-fs/adapters/rm2__coreerr-mc/adapter sev=/root/autodl-fs/adapters/sev__core-mc/adapter || { echo GADD_SERVE_FAIL; exit 1; }
echo "$(date '+%F %T') [m2-add] start gen_grpo (w_hard2=$WH2 w_real2=$WR2)"
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT \
  --prompts outputs/rt/gentarget/prompts.jsonl --eval_prompts outputs/rt/gentarget/eval_prompts.jsonl \
  --reward target --div_mode place --err_mode ab --w_div 0.1 --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --raw_ratio 4 \
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 \
  --steps $STEPS --lr 5e-6 --n 16 --prompts_per_step 16 --eval_k 4 --save_every 50 \
  --vllm_url http://127.0.0.1:8000/v1 --temps rm1=rm1__core,rm2=rm2__coreerr,sev=sev__core \
  --shape $SHAPE --w_hard2 $WH2 --w_real2 $WR2 --gate 0.5 --base_fail 0.1 --blind_lo 0 --blind_thr 0.5 --proc_producer 1 --pipeline 1 \
  --version_prefix cadd_ --out $O > $A/gcore_$TAG.log 2>&1 && echo GADD_DONE || echo GADD_FAILED
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
echo "$(date '+%F %T') GADD END"
