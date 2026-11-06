#!/bin/bash
# gbl 重跑（2026-11-05：第一次 group_by_length 在 transformers 5.x 下 TypeError，改 train_sampling_strategy 后重跑）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; R=rm1__mcgold_ref_gbl; LOG=$A/post.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
rm -rf $FS/runs/claims/mcrm_$R outputs/rt/rm/$R/FAILED
t0=$(date +%s)
GBL=1 RM=$R GPU=0 PORT=8100 MC=$A/models/MiniCPM5-2B bash scripts/remote/run_mcrm.sh || { log "train $R FAILED"; exit 1; }
log "$R train+eval $(( ($(date +%s) - t0) / 60 )) min; $(grep -ao "'train_runtime': '[0-9.]*'" $A/$R-mc.log | tail -n 1); steps $(grep -ao '[0-9]*/[0-9]* \[' $A/$R-mc.log | tail -n 1)"
R=$R bash scripts/remote/score_one.sh
log "GBL_DONE"
