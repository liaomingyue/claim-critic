#!/bin/bash
# R2：复现核心结论"提升来自新 critic"（技术分享表 8）。
#   从发布的 RAG-S2（mc-grpo-r3_core_cont/step100）出发，R3 奖励各训 100 步，只换 critic：
#     rp_bf：rm1__core_r2_bfcl（第一轮补盲后的新 critic）   ← 原实验 RAG-S3-bf
#     rp_r1：rm1__core_r1（原来的 critic）                  ← 原实验 RAG-S3-r1（对照）
#   然后起点 + 第 50 / 100 步在三套验收原文上生成，9B 金标 critic 打分，配对 bootstrap 比较。
#   bash scripts/repro/r2_rag.sh [rp_bf|rp_r1|cmp]    # 不带参数：两组依次跑完再比较；两台机器可以各跑一组（共享 FS_ROOT 和代码目录），最后跑 cmp
# 单卡 5090 一组约 6–8 小时（训练 + 生成 + 9B 打分）。结果不会逐位一致（vLLM 采样有非确定性）；判据是结论方向和显著性。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd "$ROOT"
START=$FS_ROOT/adapters/mc-grpo-r3_core_cont/step100
run_arm() {   # $1 组名 $2 critic
  [ -f outputs/rt/core/loop2_$1/eval/s100_c_9b.json ] && { rlog "R2 $1 已完成，跳过"; return 0; }
  rlog "R2 arm $1: critic $2, start $START"
  START=$START CRITIC=$2 ARM=$1 bash scripts/remote/run_loop2cont.sh > $TMP_ROOT/r2_$1.log 2>&1
  [ -f outputs/rt/core/loop2_$1/eval/s100_c_9b.json ] || { rlog "R2 $1 FAILED（见 $TMP_ROOT/r2_$1.log、$TMP_ROOT/r3full_core_$1*.log）"; return 1; }
  rlog "R2 arm $1 done"
}
cmp() {
  E=outputs/rt/core
  python -m ragtruth.pol9b_cmp --arm s2=$E/loop2_rp_bf/eval:v0 --arm rp_r1=$E/loop2_rp_r1/eval:s100 --arm rp_bf=$E/loop2_rp_bf/eval:s100 \
    --base s2 --n_boot 1000 --out $TMP_ROOT/r2_cmp.json > /dev/null || return 1
  python -m scripts.repro.compare_r2 --repro $TMP_ROOT/r2_cmp.json --orig results/loop2_cont5_ctl_cmp.json | tee $TMP_ROOT/r2_report.txt
  rlog "R2_DONE（报告：$TMP_ROOT/r2_report.txt）"
}
case ${1:-all} in
  rp_bf) run_arm rp_bf rm1__core_r2_bfcl ;;
  rp_r1) run_arm rp_r1 rm1__core_r1 ;;
  cmp) cmp ;;
  all) run_arm rp_bf rm1__core_r2_bfcl && run_arm rp_r1 rm1__core_r1 && cmp ;;
esac
