#!/bin/bash
# 方向 B——用 v2h 批量采样，筛「过门槛（sev 有错 ≥ 0.9、自然 ≥ 0.5）、critic 0.1–0.5、sev 判要求类型 ≥ 0.5」，
#   每提示 ≤ 2（+ 上一轮难样本里同样满足的），和原 SFT 数据重训 gen-v2h2-mc，再在同一批 64 提示 × 32 上试采。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_b.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
source $S/bg_sync.sh
V2H=$FS/adapters/gen-v2h-mc/adapter; V2H2=$A/ckpt/gen-v2h2-mc/adapter; SEV=sev__v3c0   # 训练 / 试采读本地盘，共享盘后台同步
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
if [ ! -s $O/sft_hard_v2.jsonl ]; then
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-b.log $RMS policy_v0=$V2H >> $LOG 2>&1 || { log "harvest vllm FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2_grpo harvest --model policy_v0 --tokenizer $MC --n ${HN:-12} --lo 0.1 --hi 0.5 --min_req 0.5 --tag _v2 --prev_hard harvest_hard.jsonl >> $LOG 2>&1 || { log "harvest FAILED"; exit 1; }
  log "HARVEST $(grep -aE 'GEN_V2_GRPO_HARVEST|PREV_HARD' $LOG | tail -2 | tr '\n' ' ')"
fi
if [ ! -f $A/ckpt/gen-v2h2-mc/SAVED ]; then
  if pgrep -f 'train.sft --config configs/rt/sft_gen_v2h2' > /dev/null; then   # 已经在训（上一次启动留下的）→ 等它训完
    while pgrep -f 'train.sft --config configs/rt/sft_gen_v2h2' > /dev/null; do sleep 30; done
  else
    free_gpus
    CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_v2h2_mc.yaml --mode coldstart > $A/gen-v2h2-mc.log 2>&1 || { log "sft v2h2 FAILED"; exit 1; }
  fi
  [ -f $A/ckpt/gen-v2h2-mc/SAVED ] || { log "sft v2h2 FAILED (no SAVED)"; exit 1; }
  log "trained gen-v2h2-mc $(grep -ao "'train_loss': '[0-9.]*'" $A/gen-v2h2-mc.log | tail -1)"
fi
[ -f $FS/adapters/gen-v2h2-mc/adapter/adapter_model.safetensors ] || bg_sync $V2H2 $FS/adapters/gen-v2h2-mc/
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bp.log $RMS v2h2=$V2H2 >> $LOG 2>&1 || { log "probe vllm FAILED"; exit 1; }
python -u -m ragtruth.gen_v2_grpo probe --model v2h2 --tokenizer $MC >> $LOG 2>&1
log "PROBE v2h2 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2500)"
free_gpus
log "GEN_B_DONE"
