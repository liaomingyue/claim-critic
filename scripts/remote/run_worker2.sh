#!/bin/bash
# 第二个领任务队列（2026-11-05，用户：上中下三档 critic 各训到 200 步，看能否拉开）。
# 和 run_worker.sh 不同：每领完一个就重读 $FS/runs/q3/tasks，新加的任务也能领到；没有可领的才退出。
#   r200<arm>  arm ∈ h3ctl / z1500 / ref：从第 50 步续训到 200 步（LR 常数，续训等价于一口气训），评第 100 / 200 步
#              不重评 v0（打分器会跳过已有打分文件，重新生成的 v0 会对上旧分数），统一用 h3ctl 组那次的 v0 当基准
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; Q=$FS/runs/q3; mkdir -p $Q; LOG=/root/autodl-tmp/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
[ -f $Q/tasks ] || echo 'r200ref r200z1500 r200h3ctl' > $Q/tasks
while true; do
  t=""; for x in $(cat $Q/tasks); do mkdir $Q/claim_$x 2>/dev/null && { t=$x; break; }; done
  [ -n "$t" ] || break
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_$t/by
  log "run $t"
  case $t in
    r200h3ctl) CRITIC=rm1__mcgold_h3ctl TAG= STEPS=200 EVAL_STEPS="100 200" EVAL_V0=0 bash scripts/remote/run_r3full.sh ;;
    r200z1500) CRITIC=rm1__mcgold_ref_z1500 TAG=_z1500 STEPS=200 EVAL_STEPS="100 200" EVAL_V0=0 bash scripts/remote/run_r3full.sh ;;
    r200ref) CRITIC=rm1__mcgold_ref TAG=_ref STEPS=200 EVAL_STEPS="100 200" EVAL_V0=0 bash scripts/remote/run_r3full.sh ;;
    *) log "unknown $t"; false ;;
  esac && touch $Q/done_$t && log "done $t" || log "$t FAILED"
done
log "WORKER_DONE"
