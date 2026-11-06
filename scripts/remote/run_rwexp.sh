#!/bin/bash
# 奖励设计对照（2026-11-05，用户：分步奖励 + 去底噪计数；跑 50 步验证，不要跑完）。单卡一组：策略 vLLM（占 MC_MEM 显存）+ RM LoRA + HF 训练都在 GPU0。
#   ARM=R0  现在的公式：1 − 0.5 平均 P − 0.5 最大 P，满意度 0.5 ×（最多扣 0.25）
#   ARM=R1  整段层换成去底噪计数：e = max(0, p − 同组平均 P)，1 − 0.5 P(至少一句) − 0.5 min(1, Σe / K)，K = max(2, 0.5 × 参考回答句数)；满意度 0.8 ×（最多扣 0.4）
#   ARM=R2  R1 + 句子层：错句 token 额外加 0.6 × −max(0, p − b) / σ
#   ARM=R3  R2 + 后一句的负优势按 0.3 传给前一句
# 进奖励：CRITIC（数量曲线 / 底座对照选出的那个）、sat_pair__mc（满意度，只罚）、refuse（拒答）、硬规则。只监控：无（独立评测另跑，9B 不进奖励）。
# 同一初始策略（mc-init）、同一批 prompt 和顺序、lr 5e-6、n 16 × 每步 16 prompt、STEPS（默认 50）步。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; ARM=${ARM:?R0/R1/R2/R3}; LOG=$A/rwexp_$ARM.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
MC=$A/models/MiniCPM5-2B; SRC=${SRC:-outputs/rt/minicpm/grpo}; INIT=$FS/adapters/mc-init; CRITIC=${CRITIC:?critic name}
O=outputs/rt/minicpm/rwexp/$ARM; mkdir -p $O
case $ARM in
  R0) RW="--reward_shape legacy --sat_w 0.5" ;;
  R1) RW="--reward_shape count --count_ratio 0.5 --count_min 2 --sat_w 0.8" ;;
  R2) RW="--reward_shape count --count_ratio 0.5 --count_min 2 --sat_w 0.8 --sent_beta 0.6" ;;
  R3) RW="--reward_shape count --count_ratio 0.5 --count_min 2 --sat_w 0.8 --sent_beta 0.6 --sent_gamma 0.3" ;;
  *) log "unknown ARM $ARM"; exit 1 ;;
esac
for f in train_ref.jsonl dev_ref.jsonl; do [ -f $SRC/$f ] || { log "missing $SRC/$f"; exit 1; }; done
pkill -f '[v]llm serve'
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
mkdir -p $A/s0
A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --mem ${MC_MEM:-0.35} --base $MC --log $A/vllm-rwexp.log --priority \
  critic=$FS/adapters/$CRITIC-mc/adapter sat=$FS/adapters/sat_pair__mc-mc/adapter refuse=$FS/adapters/refuse-mc/adapter policy_v0=$INIT >> $LOG 2>&1 \
  || { log "serve FAILED"; exit 1; }
COMMON="--causal 1 --sys_qa 1 --base $MC --init $INIT --vllm_url http://127.0.0.1:8000/v1 --policy_url http://127.0.0.1:8000/v1 \
  --n 16 --share 1 --share_impl train.prefix_share_llama --share_ckpt 1 --share_mb ${SHARE_MB:-32} --prompts $SRC/train_ref.jsonl --dev $SRC/dev_ref.jsonl \
  --critic_report outputs/rt/rm/$CRITIC/eval/report.json --shadow= \
  --sat_model sat --sat_kind pair --sat_report outputs/rt/rm/sat_pair__mc/eval/report.json --refuse_model refuse --refuse_report outputs/rt/rm/refuse/eval/report.json --refuse_w 0.4 \
  $RW --proc_producer 1 --pipeline 1 --lr 5e-6 --state_every 1000 --trip 0"
# 单卡试跑 3 步（用户：单卡先试，不行再双机）：OOM 或每步 > 60 秒 → 写 need_pair 标记退出，由本地改成两台一组的双机模式
P=$O/smoke; rm -rf $P
if ! CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo $COMMON --steps 3 --eval_k 0 --save_every 1000 --version_prefix ${ARM}s_ --push_dir /dev/shm/rws_$ARM --out $P >> $A/rwexp_${ARM}_smoke.log 2>&1; then
  why=$(grep -aq "OutOfMemory\|CUDA out of memory" $A/rwexp_${ARM}_smoke.log && echo oom || echo error)
  log "smoke FAILED ($why)"; mkdir -p $FS/runs/rwexp && echo "$why $(hostname)" > $FS/runs/rwexp/need_pair_$ARM; pkill -f '[v]llm serve'; exit 2
fi
ts=$(python -c "import json;r=[json.loads(l) for l in open('$P/log.jsonl')];r=[x for x in r if 't_step' in x][1:] or r;print(round(sum(x['t_step'] for x in r)/len(r),1))")
log "smoke ok: t_step $ts s"
if python -c "import sys;sys.exit(0 if $ts > ${MAX_STEP_S:-60} else 1)"; then
  log "too slow on one card"; echo "slow $ts $(hostname)" > $FS/runs/rwexp/need_pair_$ARM; pkill -f '[v]llm serve'; exit 2
fi
log "start $ARM: critic $CRITIC, $RW, steps ${STEPS:-50}"
CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo $COMMON --steps ${STEPS:-50} --save_every 50 --eval_k 2 \
  --version_prefix ${ARM}_ --push_dir /dev/shm/rw_$ARM --out $O/run >> $A/rwexp_${ARM}_train.log 2>&1 && log "train done" || { log "train FAILED"; exit 1; }
grep -h QUICK_EVAL $A/rwexp_${ARM}_train.log | cut -c1-600 | tee -a $LOG
mkdir -p $FS/adapters/rwexp-$ARM && rsync -a $O/run/ckpt/ $FS/adapters/rwexp-$ARM/ && log "ckpt saved to adapters/rwexp-$ARM"
pkill -f '[v]llm serve'
log "RWEXP_DONE $ARM"
