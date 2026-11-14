#!/bin/bash
# 给错误合成把关的三个 2B 模型，用上全部 Claude 标注重训（数据 ragtruth/rm_ops.py 本地建好随代码上传）：
#   sev__v4 真错和严重度；rm2__v4 自然度；opm__v1 改动是不是指定的操作。两张卡并行：卡 0 训 sev 再训 opm，卡 1 训自然度。只有一张卡时顺序跑。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; S=scripts/remote; LOG=$A/rmops.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
one() { RM=$1 GPU=$2 PORT=$3 MC=$MC CODE=$PWD bash $S/run_mcrm.sh >> $LOG 2>&1
  [ -f outputs/rt/rm/$1/eval/report.json ] && log "TRAINED $1 $(python -c "import json;r=json.load(open('outputs/rt/rm/$1/eval/report.json'));print(r['temperature'], {k:r[k].get('auc') for k in ('calib','test') if isinstance(r.get(k),dict)})")" || log "$1 FAILED"; }
if [ "$N" -ge 2 ]; then
  ( one sev__v4 0 8100; one opm__v1 0 8100 ) &
  ( one rm2__v4 1 8101 ) &
  wait
else
  one sev__v4 0 8100; one rm2__v4 0 8100; one opm__v1 0 8100
fi
log "RMOPS_DONE"
