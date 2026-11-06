#!/bin/bash
# 第三个领任务队列 q4（2026-11-05，用户：SFT 上限 + 被钻空子 + nat1500 档 200 步，开始干）。每领完一个重读任务列表。
#   nat200  nat1500 档（rm1__mcgold_ref_n1500：1500 对、RM2 ≥ 0.3）R3 训 200 步，评第 50 / 100 / 200 步（KL 上限 0.2，同三档放宽后）
#   sftcap  run_sftcap.sh
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; Q=$FS/runs/q4; mkdir -p $Q; LOG=/root/autodl-tmp/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
[ -f $Q/tasks ] || echo 'nat200 sftcap' > $Q/tasks
while true; do
  t=""; for x in $(cat $Q/tasks); do mkdir $Q/claim_$x 2>/dev/null && { t=$x; break; }; done
  [ -n "$t" ] || break
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_$t/by
  log "run $t"
  case $t in
    nat200) TRIP_KL=0.2 CRITIC=rm1__mcgold_ref_n1500 TAG=_n1500 STEPS=200 EVAL_STEPS="50 100 200" EVAL_V0=0 bash scripts/remote/run_r3full.sh ;;
    sftcap) bash scripts/remote/run_sftcap.sh ;;
    *) log "unknown $t"; false ;;
  esac && touch $Q/done_$t && log "done $t" || log "$t FAILED"
done
log "WORKER_DONE"
