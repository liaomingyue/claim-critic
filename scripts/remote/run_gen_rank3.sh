#!/bin/bash
# 只做「原生 + 组内排行」两组（r1：SFT 起点；r2：v2h 起点），都只训 50 步。
#   r1 已在 run_gen_rank.sh 里按 100 步在跑 → 等第 50 步检查点存好就停掉，试采；r2 只训 50 步，试采。h4 续训 / r3 取消。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_rank3.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
V2H=$FS/adapters/gen-v2h-mc/adapter; SEV=sev__v3c0
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
probe() {
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-rkp3.log $RMS $1=$2 >> $LOG 2>&1 || { log "probe $1 vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_v2_grpo probe --model $1 --tokenizer $MC >> $LOG 2>&1
  log "PROBE $1 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2000)"
}
ck() { local c=$O/$1/ckpt/step50; [ -d $c/policy ] && c=$c/policy; echo $c; }
until [ -f $(ck r1)/adapter_model.safetensors ]; do sleep 30; done
sleep 60
pkill -f run_gen_rank.sh; pkill -f 'train.gen_grpo'; sleep 10; free_gpus
log "r1 stopped at step 50"
[ -s $O/probe_r1s50.jsonl ] || probe r1s50 $(ck r1)
BASE="--causal 1 --base $MC --prompts $O/prompts.jsonl --reward v2sev --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128
  --steps 50 --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50 --v2_shape hard --err_gate 0.9 --nat_gate 0.5 --hard_shape rank --hard_base 0"
if [ ! -d $O/r2/ckpt/step50 ]; then
  free_gpus
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-rk3.log --priority $RMS policy_v0=$V2H >> $LOG 2>&1 || { log "r2 vllm FAILED"; exit 1; }
  log "train r2 (init v2h, 50 steps)"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE --init $V2H --version_prefix r2_ --out $O/r2 >> $A/gengrpo_r2.log 2>&1 && log "r2 done" || log "r2 FAILED"
fi
[ -s $O/probe_r2s50.jsonl ] || probe r2s50 $(ck r2)
free_gpus
log "GEN_RANK3_DONE"
