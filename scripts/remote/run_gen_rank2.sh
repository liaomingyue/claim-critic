#!/bin/bash
# 「原生奖励 + 组内排行」要和原生奖励对比；难样本 SFT 本身也是一个手段 → 补成 2×2（奖励 tri 无保底 / rank × 起点 SFT / 热启动），都到 100 步：
#   h4：原生（tri 无保底）、SFT 起点，从第 50 步状态接着训到 100 步；r3：原生、热启动 gen-v2h 起点，100 步。
#   （r1 = rank + SFT 起点、r2 = rank + 热启动，在 run_gen_rank.sh；gen-v2h 只 SFT 不 GRPO 的试采也在那边。）等 run_gen_rank.sh 跑完再开始。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_rank2.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
until grep -aq GEN_RANK_DONE $A/gen_rank.log 2>/dev/null; do sleep 60; done
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
V2=$FS/adapters/gen-v2-mc/adapter; V2H=$FS/adapters/gen-v2h-mc/adapter; SEV=sev__v3c0
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
probe() {
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-rkp2.log $RMS $1=$2 >> $LOG 2>&1 || { log "probe $1 vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_v2_grpo probe --model $1 --tokenizer $MC >> $LOG 2>&1
  log "PROBE $1 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2000)"
}
BASE="--causal 1 --base $MC --prompts $O/prompts.jsonl --reward v2sev --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128
  --steps 100 --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50 --v2_shape hard --err_gate 0.9 --nat_gate 0.5 --hard_shape tri --hard_base 0"
declare -A INITS=([h4]=$V2 [r3]=$V2H)
for e in h4 r3; do
  R=$O/$e
  if [ ! -d $R/ckpt/step100 ]; then
    free_gpus
    A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-rk2.log --priority $RMS policy_v0=${INITS[$e]} >> $LOG 2>&1 || { log "$e vllm FAILED"; continue; }
    RES=""; [ $e = h4 ] && RES="--resume $R/state/step50"
    log "train $e (init ${INITS[$e]}) $RES"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE --init ${INITS[$e]} --version_prefix ${e}b_ --out $R $RES >> $A/gengrpo_${e}_2.log 2>&1 && log "$e done" || log "$e FAILED"
  fi
  for st in 50 100; do
    [ $e = h4 ] && [ $st = 50 ] && continue   # h4 第 50 步已试采（probe_h4）
    CK=$R/ckpt/step$st; [ -d $CK/policy ] && CK=$CK/policy
    [ -d $CK ] && [ ! -s $O/probe_${e}s$st.jsonl ] && probe ${e}s$st $CK
  done
done
free_gpus
log "GEN_RANK2_DONE"
