#!/bin/bash
# 队列式调度，不做固定分工。共享盘上的任务队列：$Q/tasks 每行「名字|依赖|命令」，依赖用逗号分隔：
#   另一个任务名（要它在 $Q/done 里）或 file:<路径>（相对代码目录或绝对路径，要存在）。哪台机器空了就领下一个依赖已满足的任务（mkdir 原子占位）。
#   queue_worker.sh <队列目录> [<开工前要等的日志文件> <要等的字符串>]
cd ${CODE:-/root/autodl-fs/code/critic_model}
Q=$1; A=/root/autodl-tmp; ME=$(hostname); LOG=$A/queue_$(basename $Q).log
log() { echo "$(date '+%F %T') [$ME] $*" | tee -a $LOG $Q/history.log > /dev/null; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
[ -n "$2" ] && { log "waiting for '$3' in $2 before joining"; until grep -aq "$3" $2 2>/dev/null; do sleep 30; done; }
mkdir -p $Q/claim $Q/done $Q/failed
log "joined"
while true; do
  left=0; ran=0
  while IFS='|' read -r name deps cmd; do
    [ -z "$name" ] && continue; [[ $name == \#* ]] && continue
    [ -e $Q/done/$name ] || [ -e $Q/failed/$name ] && continue
    left=1
    [ -d $Q/claim/$name ] && continue
    ok=1
    for d in ${deps//,/ }; do
      if [[ $d == file:* ]]; then [ -e "${d#file:}" ] || ok=0
      else [ -e $Q/done/$d ] || ok=0; fi
    done
    [ $ok = 1 ] || continue
    mkdir $Q/claim/$name 2>/dev/null || continue
    log "start $name"
    bash -c "$cmd" < /dev/null >> $A/task_$name.log 2>&1 && { touch $Q/done/$name; log "done $name"; } || { touch $Q/failed/$name; log "FAILED $name (see $A/task_$name.log on $ME)"; }
    ran=1; break
  done < $Q/tasks
  [ $left = 0 ] && { log "queue empty, exit"; break; }
  [ $ran = 0 ] && sleep 30
done
