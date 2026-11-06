#!/bin/bash
# 新生成器 GRPO（2026-11-05，用户：优化错误生成器；critic 用拒答修正版 ref；复用两台单卡）。双机模式，由 run_train_node.sh 调起
# （JOB=run_gensev.sh；推理机按 runs/pair/<配对>/loras 起 rm1 / rm2 / sev / policy_v0）。
# 奖励 shape=sev：严重程度 RM（sev__mc）P(显然) ≥ 0.5 才合格，原句有错用 sev 的 P(F)；合格 r = (1 − P(critic 判出)) × 自然度项；没过门槛 0.1 × P(显然) / 0.5。
# 进奖励：sev__mc（门槛）、rm1__mcgold_ref（critic，盲区 = 它判不出的）、rm2v2__mcgen3（自然度）、硬规则。只监控：无（9B 不进奖励、不筛样本）。
# 起点 gen-seed5-mc（同 a7–a12），同一提示池（FOCUS_TACTICS：张冠李戴、跨段拼接、数字、确定性 / 范围、顺序、关系反转、因果…），n 16 × 每步 16 个 prompt，STEPS 步。
# 训练进程独占 GIL（--proc_producer 1）、每个 prompt 生成完立刻打分（--pipeline 1）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/gensev.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 采样子进程别抢训练线程的 CPU（2026-11-05：子进程折合约 13 核，训练主线程单核跑满、发 kernel 慢）
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
O=outputs/rt/gentarget; SEED=${SEED:-seed5}; INIT=$FS/adapters/gen-$SEED-mc/adapter; R=$O/grpo_${SEED}_sev${TAG:-}
[ -n "${RM_URL:-}" ] || { log "needs RM_URL (run via run_train_node.sh)"; exit 1; }
[ -f $INIT/adapter_model.safetensors ] || { log "missing $INIT"; exit 1; }
# 基座先拷本地盘：直接从共享盘加载每个张量约 8 秒（2026-11-04 实测）
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
MC=$A/models/MiniCPM5-2B
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" $RM_URL/models)
curl -s -m 10 -H "Authorization: Bearer $VLLM_API_KEY" $RM_URL/models | grep -q '"sev"' || { log "infer node has no sev LoRA ($code)"; exit 1; }
unset LORA_SYNC_FS_DIR   # 生成器每步把 LoRA 存在共享盘上的运行目录里，推理机直接从那里拷到内存盘
export LORA_SYNC_MODE=fs
log "start gensev: out $R, steps ${STEPS:-50}"
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --eval_prompts $O/eval_prompts.jsonl \
  --reward target --div_mode place --err_mode ab --w_div 0.1 --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --raw_ratio 4 \
  --share 1 --share_impl train.prefix_share_llama --share_mb ${SHARE_MB:-8} --ref_on_vllm 1 \
  --steps ${STEPS:-50} --lr 5e-6 --n 16 --prompts_per_step 16 --eval_k 4 --save_every 50 \
  --vllm_url $RM_URL --temps rm1=rm1__mcgold_ref,rm2=rm2v2__mcgen3,sev=sev__mc \
  --shape sev --gate 0.5 --base_fail ${BASE_FAIL:-0.1} --blind_bonus ${BLIND_BONUS:-0} --blind_lo ${BLIND_LO:-0} --blind_thr ${BLIND_HI:-0.5} --proc_producer 1 --pipeline 1 --ref_batch ${REF_BATCH:-1} \
  --version_prefix ${SEED}sev_ --out $R ${RESUME:+--resume $RESUME} ${RESET_REWARD:+--reset_reward 1} >> $A/gensev_train.log 2>&1 && log "gensev train done" || { log "gensev train FAILED"; exit 1; }
grep -h QUICK_EVAL $A/gensev_train.log | cut -c1-600 | tee -a $LOG
mkdir -p $FS/adapters/gen-${SEED}-sev && rsync -a $R/ckpt/ $FS/adapters/gen-${SEED}-sev/ && log "ckpts saved to adapters/gen-${SEED}-sev"
log "GENSEV_DONE"
