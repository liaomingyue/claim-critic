#!/bin/bash
# source scripts/remote/model_env.sh [模型名]：按 configs/models/<模型名>.yaml 导出 MODEL_* 变量，基座拷到本地盘（README「多模型支持」）。
# 默认 MODEL=minicpm5-2b。拷贝顺序读：直接从文件存储加载太慢（约 25 MB/s 随机读）。
eval "$(python -m common.models env ${1:-${MODEL:-minicpm5-2b}})" || return 1
# 9B 走 models9b.sh 的同一把锁（训练机开机就在后台拷，这里等它拷完，不另起一份 rsync 写正式目录）
if [ "$(basename $MODEL_LOCAL)" = Qwen3.5-9B ]; then source $(dirname ${BASH_SOURCE[0]})/models9b.sh && ensure_9b || return 1
else [ -f $MODEL_LOCAL/config.json ] || { mkdir -p $(dirname $MODEL_LOCAL) && rsync -a $MODEL_FS $(dirname $MODEL_LOCAL)/; }; fi
