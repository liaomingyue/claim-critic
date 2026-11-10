#!/bin/bash
# 通用领任务（机器可能因 GPU 空闲被看门狗关机，之后重排）。按 $FS/runs/q2/tasks 的顺序 mkdir 占位，谁先空谁领；全程保活。
#   evalh3ctl  h3ctl 组第 50 步的评测（补完；已有的跳过）
#   ref / z1500  R3 对照组（50 步 + 评测）
#   s2         拒绝采样 SFT（采样 + 选最优 + SFT + 评测）
#   s1         Claude 修正 SFT（拼回 + SFT + 评测）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; Q=$FS/runs/q2; mkdir -p $Q; LOG=/root/autodl-tmp/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!   # 保活进程 PID：下面的 wait 不能等它（踩过：worker1 生成完卡死在 wait）
[ -f $Q/tasks ] || echo 'evalh3ctl ref z1500 s2 s1' > $Q/tasks
for t in $(cat $Q/tasks); do
  mkdir $Q/claim_$t 2>/dev/null || { log "$t taken"; continue; }
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_$t/by
  log "run $t"
  case $t in
    evalh3ctl) NOWAIT=1 EVAL_STEPS=50 TAG= bash scripts/remote/run_r3eval.sh ;;
    ref|z1500) rmdir $FS/runs/r3arms/claim_$t 2>/dev/null; rm -rf $FS/runs/r3arms/claim_$t; ARM=$t bash scripts/remote/run_r3arm.sh ;;
    s2) bash scripts/remote/run_s2.sh ;;
    s1) bash scripts/remote/run_s1.sh ;;
    *) log "unknown $t" ;;
  esac && touch $Q/done_$t && log "done $t" || log "$t FAILED"
done
log "WORKER_DONE"
