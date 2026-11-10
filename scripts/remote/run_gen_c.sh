#!/bin/bash
# 先 GRPO 50 步把生成器往难的方向推（奖励不全 0），再从 GRPO 后的模型批量采样挑难样本做 SFT。等 run_gen_b.sh 跑完再开始。
#   g1：起点 v2h2，奖励 h4（三角 0.3–0.5 为 1、0.5→0.9 降、无保底；门槛 sev 有错 ≥ 0.9 + 自然 ≥ 0.5），提示按类型加权（prompts_w：
#       范围偏移 / 近义 / 拼接 / 张冠李戴各约 17%，推断 / 数字 / 附加语各约 9%，矛盾 / 添加事实各约 3%；不含 64 个试采提示），50 步 → 试采
#   再用 g1 在训练提示上 × 8 采样，同 B 的筛选（0.1–0.5、要求类型 ≥ 0.5，并入前两轮）→ gen-v2h3-mc → 试采
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_c.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
until grep -aqE 'GEN_B_DONE' $A/gen_b.log 2>/dev/null; do sleep 60; done
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
source $S/bg_sync.sh
# 训练 / 试采 / 采样只读写本地盘（$A），共享盘（$O、$FS/adapters）由后台同步
V2H2=$A/ckpt/gen-v2h2-mc/adapter; V2H3=$A/ckpt/gen-v2h3-mc/adapter; SEV=sev__v3c0; R=$A/runs
[ -f $V2H2/adapter_model.safetensors ] || { log "no v2h2, stop"; exit 1; }
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
probe() {
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-cp.log $RMS $1=$2 >> $LOG 2>&1 || { log "probe $1 vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_v2_grpo probe --model $1 --tokenizer $MC >> $LOG 2>&1
  log "PROBE $1 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2500)"
}
ck() { local c=$R/$1/ckpt/step50; [ -d $c/policy ] && c=$c/policy; echo $c; }
if [ ! -d $R/g1/ckpt/step50 ]; then
  free_gpus
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-c.log --priority $RMS policy_v0=$V2H2 >> $LOG 2>&1 || { log "g1 vllm FAILED"; exit 1; }
  log "train g1 (init v2h2, h4 reward, weighted prompts, 50 steps)"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $V2H2 --prompts $O/prompts_w.jsonl --reward v2sev \
    --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2 --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 \
    --max_tokens 512 --pipeline 1 --rm_concurrency 128 --steps 50 --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50 \
    --v2_shape hard --err_gate 0.9 --nat_gate 0.5 --hard_shape tri --hard_base 0 --version_prefix g1_ --out $R/g1 >> $A/gengrpo_g1.log 2>&1 &
  TP=$!; mirror_loop $TP $R/g1 $O/g1
  wait $TP && log "g1 done" || { log "g1 FAILED"; exit 1; }
fi
[ -s $O/probe_g1s50.jsonl ] || probe g1s50 $(ck g1)
if [ ! -s $O/sft_hard_v3.jsonl ]; then
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ch.log $RMS policy_v0=$(ck g1) >> $LOG 2>&1 || { log "harvest vllm FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2_grpo harvest --model policy_v0 --tokenizer $MC --n ${HN:-8} --lo 0.1 --hi 0.5 --min_req 0.5 --tag _v3 --prev_hard harvest_hard_v2.jsonl >> $LOG 2>&1 || { log "harvest FAILED"; exit 1; }
  log "HARVEST $(grep -aE 'GEN_V2_GRPO_HARVEST|PREV_HARD' $LOG | tail -2 | tr '\n' ' ')"
fi
if [ ! -f $V2H3/adapter_model.safetensors ]; then
  free_gpus
  CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_v2h3_mc.yaml --mode coldstart > $A/gen-v2h3-mc.log 2>&1 || { log "sft v2h3 FAILED"; exit 1; }
  bg_sync $V2H3 $FS/adapters/gen-v2h3-mc/
  log "trained gen-v2h3-mc $(grep -o "'train_loss': '[0-9.]*'" $A/gen-v2h3-mc.log | tail -1)"
fi
[ -s $O/probe_v2h3.jsonl ] || probe v2h3 $V2H3
free_gpus
log "GEN_C_DONE"
