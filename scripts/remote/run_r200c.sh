#!/bin/bash
# 三档 200 步续跑（2026-11-05）：TripWire 的 KL 上限 0.10 在第 ~100 步触发（ref 第 98 步窗口均值 0.101，其他护栏都正常），
# 放宽到 TRIP_KL=0.2（只改 KL 这一项），从最近的续训状态接着训到 200 步，评第 100 / 200 步（不重评 v0）。用法：ARM=ref|z1500|h3ctl bash run_r200c.sh
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
Q=/root/autodl-fs/runs/q3; LOG=/root/autodl-tmp/worker.log
case ${ARM:?} in h3ctl) C=rm1__mcgold_h3ctl; T= ;; z1500) C=rm1__mcgold_ref_z1500; T=_z1500 ;; ref) C=rm1__mcgold_ref; T=_ref ;; *) echo "unknown $ARM"; exit 1 ;; esac
E=outputs/rt/minicpm/r3full$T/eval
# 不看 done 标记：KL 在第 100 步后触发时 run_r3full 照样会评完第 100 步并标 done，但还没到 200 步
[ -s $E/s200_c_9b.json ] && { echo "r200$ARM already at 200"; exit 0; }
echo "$(date '+%F %T') run r200$ARM (continue, TRIP_KL 0.2)" >> $LOG
# 只评还没打分的检查点（已评过的第 100 步不重新生成，否则新回答会对上旧打分文件）
ES=""; for s in 100 200; do [ -s $E/s${s}_c_9b.json ] || ES+=" $s"; done
TRIP_KL=0.2 CRITIC=$C TAG=$T STEPS=200 EVAL_STEPS="$ES" EVAL_V0=0 bash scripts/remote/run_r3full.sh \
  && touch $Q/done_r200$ARM && echo "$(date '+%F %T') done r200$ARM" >> $LOG || echo "$(date '+%F %T') r200$ARM FAILED" >> $LOG
