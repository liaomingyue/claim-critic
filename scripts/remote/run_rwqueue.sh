#!/bin/bash
# 奖励对照排队（两张卡跑完当前任务后接着跑 reward 实验；50 步验证）。
# 每台机器等自己的 run_scale.sh 跑完（scale.log 里 SCALE_DONE），再从共享盘上领还没人跑的组（mkdir 原子占位），R0 → R3，领完为止。
# 四组同一个 critic（CRITIC，默认 rm1__mcgold_h3ctl：当时 Claude 200 条上最好的）。全部跑完 GPU 空闲，看门狗关机。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; Q=$FS/runs/rwexp; mkdir -p $Q
log() { echo "$(date '+%F %T') $*" | tee -a $A/rwqueue.log; }
export CRITIC=${CRITIC:-rm1__mcgold_h3ctl}
log "waiting for SCALE_DONE"
until grep -q SCALE_DONE $A/scale.log 2>/dev/null; do sleep 30; done
for ARM in ${ARMS:-R0 R1 R2 R3}; do
  mkdir $Q/claim_$ARM 2>/dev/null || { log "$ARM taken"; continue; }
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_$ARM/by
  log "run $ARM (critic $CRITIC)"
  ARM=$ARM bash scripts/remote/run_rwexp.sh && touch $Q/done_$ARM || log "$ARM FAILED"
done
log "RWQUEUE_DONE"
