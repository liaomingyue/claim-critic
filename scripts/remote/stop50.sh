#!/bin/bash
# 只跑 50 步（2026-11-05，用户：两个 RM 的 R3 对照先只跑 50 步）：等第 50 步检查点存好，停掉训练和等待中的补评，存 adapter，直接评第 50 步（含 v0、三套验收原文）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; TAG=${TAG:-}; O=outputs/rt/minicpm/r3full$TAG; LOG=$A/r3full$TAG.log
until [ -f $O/run/ckpt/step50/policy/adapter_model.safetensors ] || [ -f $O/run/ckpt/step50/adapter_model.safetensors ]; do sleep 20; done
sleep 60
echo "$(date '+%F %T') stop50: step50 saved, stopping training (user: 50 steps only)" >> $LOG
pkill -f "[r]un_r3eval.sh"; pkill -f "[r]un_r3full.sh"; pkill -f "[t]rain.ans_grpo"; sleep 10; pkill -f '[v]llm serve'
mkdir -p $FS/adapters/mc-grpo-r3$TAG && rsync -a $O/run/ckpt/ $FS/adapters/mc-grpo-r3$TAG/
NOWAIT=1 EVAL_STEPS=50 TAG=$TAG bash scripts/remote/run_r3eval.sh
