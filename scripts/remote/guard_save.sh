#!/bin/bash
# 守护（2026-11-05）：R 的模型文件存好 180 秒后训练进程还没退出 → 结束它（旧代码收尾卡住会让 GPU 空闲、被看门狗关机）。之后用 SKIP_TRAIN 补评测。
set -u
CK=/root/autodl-tmp/ckpt/${R:?}-mc
until [ -f $CK/model.safetensors ] && [ -f $CK/tokenizer_config.json ]; do sleep 20; done
sleep 180
if pgrep -f "[c]onfigs/rt/exp/$R-mc.yaml" > /dev/null; then
  echo "$(date '+%F %T') guard: $R train.sft still alive 180s after save, killing" >> /root/autodl-tmp/scale.log
  pkill -9 -f "[c]onfigs/rt/exp/$R-mc.yaml"
fi
