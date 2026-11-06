#!/bin/bash
# 按长度分桶的验证（2026-11-05，用户：先做 A）：等这台的奖励对照领完（RWQUEUE_DONE），抢占位，用 ref 同一份数据、只加 group_by_length 重训，
# 比训练耗时和 Claude 200 条 / RAGTruth++ 指标（应与 ref 持平）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; R=rm1__mcgold_ref_gbl; LOG=$A/scale.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q RWQUEUE_DONE $A/rwqueue.log 2>/dev/null; do sleep 30; done
mkdir $FS/runs/scale_queue/claim_gbl 2>/dev/null || exit 0
echo "$(hostname) $(date '+%F %T')" > $FS/runs/scale_queue/claim_gbl/by
mkdir -p outputs/rt/rm/$R && cp outputs/rt/rm/rm1__mcgold_ref/{train,calib,test}.jsonl outputs/rt/rm/rm1__mcgold_ref/meta.json outputs/rt/rm/$R/
rm -rf $FS/runs/claims/mcrm_$R
t0=$(date +%s)
GBL=1 RM=$R GPU=0 PORT=8100 MC=$A/models/MiniCPM5-2B bash scripts/remote/run_mcrm.sh || { log "train $R FAILED"; exit 1; }
log "$R train+eval $(( ($(date +%s) - t0) / 60 )) min; train steps: $(grep -ao '[0-9]*/[0-9]* \[[0-9:]*<' $A/$R-mc.log | tail -n 1)"
R=$R bash scripts/remote/score_one.sh
