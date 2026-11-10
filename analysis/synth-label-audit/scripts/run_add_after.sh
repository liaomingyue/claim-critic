#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
A=/root/autodl-tmp
echo "$(date '+%F %T') [add-waiter] waiting for PHASE34_DONE..."
until grep -q PHASE34_DONE $A/phase34.out 2>/dev/null; do
  # 若抢到第二台并在那边跑了 add,会留 /root/autodl-fs/runs/claims/add_running,则退出不重复
  [ -f /root/autodl-fs/runs/claims/add_running ] && { echo "add already running elsewhere, waiter exits"; exit 0; }
  sleep 60
done
[ -f /root/autodl-fs/runs/claims/add_running ] && { echo "add already running elsewhere, waiter exits"; exit 0; }
mkdir -p /root/autodl-fs/runs/claims && touch /root/autodl-fs/runs/claims/add_running
echo "$(date '+%F %T') [add-waiter] phase34 done → launch add GRPO on machine1"
pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 5
SHAPE=add TAG=add STEPS=50 WH2=0.6 WR2=0.3 bash /root/run_gcore.sh
echo "$(date '+%F %T') [add-waiter] add GRPO finished"
