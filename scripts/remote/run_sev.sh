#!/bin/bash
# 严重程度 RM（Claude 标的显然 / 非显然 / 没有错误 + 解释对不对训 RM；可以先抢单卡）。单卡：
#   建 sev__mc（新标 787 条按原文 70/10/20 切 + h1 旧标 504 条只进训练 + rm3__mc 训练集多任务）和 sev__mc_half（新训练原文只用一半，学习曲线）
#   → 依次训练、校准、test 报告 → gates（P(显然) 各门槛精度，和旧 RM3 比）。机器跑完不关。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; LOG=$A/sev.log; S=$(dirname $(readlink -f $0))
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f outputs/rt/rm/rm3__mc/train.jsonl ] || { log "missing rm3__mc"; exit 1; }
python -u -m ragtruth.sev_rm build --name sev__mc >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
python -u -m ragtruth.sev_rm build --name sev__mc_half --frac 0.5 >> $LOG 2>&1 || { log "build half FAILED"; exit 1; }
log "built sev__mc / sev__mc_half"
for R in sev__mc sev__mc_half; do
  RM=$R GPU=0 PORT=8100 bash $S/run_mcrm.sh
  [ -f outputs/rt/rm/$R/eval/report.json ] || { log "$R FAILED"; continue; }
  python -u -m ragtruth.sev_rm gates --name $R 2>&1 | tee -a $LOG
done
log "SEV_DONE"
