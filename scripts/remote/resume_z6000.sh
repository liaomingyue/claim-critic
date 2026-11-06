#!/bin/bash
# z6000 训练已完成但进程在存完模型后退出（2026-11-05）：跳过训练补跑校准 / RAGTruth++ / 打分。和 nat1500 的训练同卡并行（显存够）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; R=rm1__mcgold_ref_z6000
rm -rf /root/autodl-fs/runs/claims/mcrm_$R outputs/rt/rm/$R/FAILED
SKIP_TRAIN=1 RM=$R GPU=0 PORT=8102 MC=$A/models/MiniCPM5-2B bash scripts/remote/run_mcrm.sh || exit 1
R=$R bash scripts/remote/score_one.sh
