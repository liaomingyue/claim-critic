#!/bin/bash
# 只跑生成器 st5（run_genops_st5.sh）然后关机。接在多维度分类那条链后面单独重跑时用。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; S=scripts/remote; LOG=$A/opd_chain2.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
pkill -f '[v]llm serve'; sleep 10
log "st5 restarted"
CODE=$PWD bash $S/run_genops_st5.sh
log "OPD_CHAIN2_DONE"
sleep ${GRACE:-1500}; shutdown
