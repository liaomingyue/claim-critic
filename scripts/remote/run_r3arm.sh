#!/bin/bash
# 领一组 R3 对照（2026-11-05，用户：A 跑完 gbl 后同时跑 ref 组）：共享盘 mkdir 占位，两台机器谁先空谁领，领不到就跳过。
# 用法：ARM=ref|z1500 bash run_r3arm.sh（50 步 + 评测 v0 / 第 50 步 × 三套验收原文）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; Q=$FS/runs/r3arms; mkdir -p $Q
case ${ARM:?} in ref) C=rm1__mcgold_ref ;; z1500) C=rm1__mcgold_ref_z1500 ;; nohs) C=rm1__mcgold_ref_nohs ;; *) echo "unknown $ARM"; exit 1 ;; esac
mkdir $Q/claim_$ARM 2>/dev/null || { echo "$ARM taken by $(cat $Q/claim_$ARM/by 2>/dev/null)"; exit 0; }
echo "$(hostname) $(date '+%F %T')" > $Q/claim_$ARM/by
CRITIC=$C TAG=_$ARM STEPS=50 EVAL_STEPS=50 bash scripts/remote/run_r3full.sh && touch $Q/done_$ARM
