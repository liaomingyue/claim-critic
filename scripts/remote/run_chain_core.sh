#!/bin/bash
# 两台并行。本机：core1 跑完 → 循环一 → 循环二主组（rm1__core_r1）；「只核心集」组在第二台（run_loop2co.sh）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; S=$A/scripts; LOG=$A/chain.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) &
until grep -q "CORE1_DONE" $A/core1.log 2>/dev/null; do pgrep -f run_core1.sh > /dev/null || { log "core1 not running and not done"; exit 1; }; sleep 60; done
log "core1 done → loop1"; CODE=$PWD bash $S/run_loop1.sh; grep -q LOOP1_DONE $A/loop1.log || { log "loop1 FAILED"; exit 1; }
log "loop1 done → loop2 main"; ARM=main CRITIC=rm1__core_r1 CODE=$PWD bash $S/run_loop2.sh
log "CHAIN_DONE"
