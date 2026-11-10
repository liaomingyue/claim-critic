#!/bin/bash
# 0.7 以上的比例要压下去、往 0.5 以下推；奖励函数做几组对比，各训 50 步，再看每道门槛的通过率。
#   共同：起点 SFT（gen-v2-mc）；门槛 sev P(有错) ≥ 0.9 + RM2 自然 ≥ 0.5（没过 0 分）；每步 8 提示 × 32；生成和打分流水线并发。
#   h1：三角（0.3–0.5 为 1，0.5→0.9 降）+ 保底 0.1（≥ 0.9 的明显错拿 0.1，优于没过门槛）——现版
#   h4：三角、去掉保底（≥ 0.9 和没过门槛一样是 0；0.7–0.9 仍有少量分当台阶）
#   h2：只奖励 < 0.7（0.1–0.5 为 1、0.5→0.7 降），无保底——最直接，但组内有梯度的少（试采 29 / 64 组）
#   每组训完：第 50 步检查点在同一批 64 个试采提示（训练用不到）× 32 上试采，和起点（probe_ge）比每道门槛的通过率、critic 分布。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gengrpo_cmp.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
INIT=$FS/adapters/gen-v2-mc/adapter; SEV=sev__v3c0; STEPS=${STEPS:-50}
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
BASE="--causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --reward v2sev --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128
  --steps $STEPS --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50 --v2_shape hard --err_gate 0.9 --nat_gate 0.5"
declare -A ARGS=([h1]="--hard_shape tri --hard_base 0.1" [h4]="--hard_shape tri --hard_base 0" [h2]="--hard_shape low --hard_base 0")
for e in ${EXPS:-h1 h4 h2}; do
  R=$O/$e
  if [ ! -d $R/ckpt/step$STEPS ]; then
    free_gpus
    A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-cmp.log --priority $RMS policy_v0=$INIT >> $LOG 2>&1 || { log "$e vllm FAILED"; continue; }
    log "train $e: ${ARGS[$e]}"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE ${ARGS[$e]} --version_prefix ${e}_ --out $R >> $A/gengrpo_$e.log 2>&1 && log "$e done" || log "$e FAILED"
    grep -aE '^\{"step"' $A/gengrpo_$e.log | awk 'NR%10==0' | cut -c1-200 | sed "s/^/[$e] /" | tee -a $LOG
    cp $A/gengrpo_$e.log $R/ 2>/dev/null
  fi
  CK=$R/ckpt/step$STEPS; [ -d $CK/policy ] && CK=$CK/policy
  [ -d $CK ] || continue
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-cmpp.log $RMS $e=$CK >> $LOG 2>&1 || { log "$e probe vllm FAILED"; continue; }
  python -u -m ragtruth.gen_v2_grpo probe --model $e --tokenizer $MC >> $LOG 2>&1
  log "PROBE $e $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2000)"
done
free_gpus
log "GENGRPO_CMP_DONE"
