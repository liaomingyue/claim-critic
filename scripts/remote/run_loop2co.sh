#!/bin/bash
# 两台机器并行。第二台：等满意度 / 拒答模型（机器 1 的 core1 在训）出报告，跑循环二「只核心集」组（critic rm1__core，不加循环一采集）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; LOG=$A/loop2co.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) &
# 新机器先把模型拷到本地盘（等机器 1 的报告时顺便做完）：2B 前台拷，9B 后台拷（训完评测要用）
source scripts/remote/model_env.sh minicpm5-2b && log "2B copied to $MODEL_LOCAL"
source scripts/remote/models9b.sh; start_9b_bg; log "9B copy started in background"
log "waiting for sat_pair__core / refuse__core reports"
until [ -f outputs/rt/rm/sat_pair__core/eval/report.json ] && [ -f outputs/rt/rm/refuse__core/eval/report.json ] && [ -f /root/autodl-fs/adapters/refuse__core-mc/adapter/adapter_model.safetensors ]; do sleep 60; done
ARM=coreonly CRITIC=rm1__core CODE=$PWD bash $(dirname $(readlink -f $0))/run_loop2.sh
log "LOOP2CO_DONE"
