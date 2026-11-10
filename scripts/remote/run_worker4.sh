#!/bin/bash
# 队列 q5（纯合成 vs 人工）：hum / syn 两组，谁先空谁领，每领完重读任务列表。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; Q=$FS/runs/q5; mkdir -p $Q; LOG=/root/autodl-tmp/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
[ -f $Q/tasks ] || echo 'hum syn' > $Q/tasks
while true; do
  t=""; for x in $(cat $Q/tasks); do mkdir $Q/claim_$x 2>/dev/null && { t=$x; break; }; done
  [ -n "$t" ] || break
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_$t/by
  log "run $t"
  case $t in
    hum|syn) ARM=$t bash scripts/remote/run_humsyn.sh ;;
    *) log "unknown $t"; false ;;
  esac && touch $Q/done_$t && log "done $t" || log "$t FAILED"
done
log "WORKER_DONE"
