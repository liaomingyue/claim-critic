#!/bin/bash
# nat1500 档补跑（2026-11-05：worker6 网络断了约 20 分钟，看门狗按 GPU 空闲关机，评测可能没做完）。
# 训练没到 200 步就从最近状态续训；到了就直接进评测（resume 第 200 步时训练循环为空）。只评还没打分的检查点；
# 先删这些检查点残留的部分打分文件（生成会重做，打分器会跳过已有文件 → 防止新回答对上旧分数）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
E=outputs/rt/minicpm/r3full_n1500/eval; mkdir -p $E
ES=""; for s in 50 100 200; do [ -s $E/s${s}_c_9b.json ] && [ -s $E/s${s}_a_9b.json ] && [ -s $E/s${s}_b_9b.json ] || { ES+=" $s"; rm -f $E/s${s}_?_9b.json; }; done
echo "$(date '+%F %T') natfin: eval steps [$ES]" >> /root/autodl-tmp/worker.log
TRIP_KL=0.2 CRITIC=rm1__mcgold_ref_n1500 TAG=_n1500 STEPS=200 EVAL_STEPS="$ES" EVAL_V0=0 bash scripts/remote/run_r3full.sh \
  && echo "$(date '+%F %T') done natfin" >> /root/autodl-tmp/worker.log || echo "$(date '+%F %T') natfin FAILED" >> /root/autodl-tmp/worker.log
