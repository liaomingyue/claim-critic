#!/bin/bash
# 接手 run_scale.sh（2026-11-05）：等它把 WAIT_ARM 训完打完分（SCALE_ONE_DONE），趁它在等底座文件时停掉，改跑 run_scale2.sh TASKS。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp
until grep -qE "SCALE_ONE_DONE ${WAIT_ARM:?}( |$)" $A/scale.log 2>/dev/null; do sleep 10; done
sleep 5
pkill -f "[r]un_scale.sh"; pkill -f "python - outputs/rt/gentarget/harvest_h3.jsonl"
echo "$(date '+%F %T') swap: stopped run_scale.sh after $WAIT_ARM, starting run_scale2 TASKS=$TASKS" >> $A/scale.log
TASKS="$TASKS" bash scripts/remote/run_scale2.sh
