#!/bin/bash
# 2026-11-09 用户同意方向①②：
#   ② 难样本热启动：gen-v2-mc 在训练提示（去掉 64 个试采提示）× 8 采样，训练同款奖励打分，挑过门槛且 critic 0.1–0.7 的（每提示 ≤ 2）
#      + 原 SFT 数据 → 重训 gen-v2h-mc → 试采
#   ① 组内相对难度奖励（--hard_shape rank：0.5 × 组内排名 + 0.5 × tri，critic < 0.1 / 没过门槛 0，无保底），各 100 步：
#      r1 从 gen-v2-mc 起；r2 从 gen-v2h-mc 起。第 50、100 步各试采一次（同一批 64 提示 × 32）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_rank.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
V2=$FS/adapters/gen-v2-mc/adapter; V2H=$FS/adapters/gen-v2h-mc/adapter; SEV=sev__v3c0
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
probe() {   # probe <名字> <适配器目录>
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-rkp.log $RMS $1=$2 >> $LOG 2>&1 || { log "probe $1 vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_v2_grpo probe --model $1 --tokenizer $MC >> $LOG 2>&1
  log "PROBE $1 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2000)"
}
# ② 采样 + 热启动 SFT
if [ ! -f $V2H/adapter_model.safetensors ]; then
  if [ ! -s $O/sft_hard.jsonl ]; then
    free_gpus
    A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-hv.log $RMS policy_v0=$V2 >> $LOG 2>&1 || { log "harvest vllm FAILED"; exit 1; }
    python -u -m ragtruth.gen_v2_grpo harvest --model policy_v0 --tokenizer $MC --n ${HN:-8} >> $LOG 2>&1 || { log "harvest FAILED"; exit 1; }
    log "HARVEST $(grep -a GEN_V2_GRPO_HARVEST $LOG | tail -1)"
  fi
  free_gpus
  CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_v2h_mc.yaml --mode coldstart > $A/gen-v2h-mc.log 2>&1 || { log "sft v2h FAILED"; exit 1; }
  mkdir -p $FS/adapters/gen-v2h-mc && rsync -a $A/ckpt/gen-v2h-mc/adapter $FS/adapters/gen-v2h-mc/ || { log "save v2h FAILED"; exit 1; }
  log "trained gen-v2h-mc $(grep -o "'train_loss': '[0-9.]*'" $A/gen-v2h-mc.log | tail -1)"
fi
[ -s $O/probe_v2h.jsonl ] || probe v2h $V2H
# ① 组内相对难度 GRPO
BASE="--causal 1 --base $MC --prompts $O/prompts.jsonl --reward v2sev --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128
  --steps 100 --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50 --v2_shape hard --err_gate 0.9 --nat_gate 0.5 --hard_shape rank --hard_base 0"
declare -A INITS=([r1]=$V2 [r2]=$V2H)
for e in ${EXPS:-r1 r2}; do
  R=$O/$e
  if [ ! -d $R/ckpt/step100 ]; then
    free_gpus
    A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-rk.log --priority $RMS policy_v0=${INITS[$e]} >> $LOG 2>&1 || { log "$e vllm FAILED"; continue; }
    log "train $e (init ${INITS[$e]})"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE --init ${INITS[$e]} --version_prefix ${e}_ --out $R >> $A/gengrpo_$e.log 2>&1 && log "$e done" || log "$e FAILED"
    cp $A/gengrpo_$e.log $R/ 2>/dev/null
  fi
  for st in 50 100; do
    CK=$R/ckpt/step$st; [ -d $CK/policy ] && CK=$CK/policy
    [ -d $CK ] && [ ! -s $O/probe_${e}s$st.jsonl ] && probe ${e}s$st $CK
  done
done
free_gpus
log "GEN_RANK_DONE"
