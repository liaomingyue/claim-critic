#!/bin/bash
# 系统提示共享的显存 / 速度 / 一致性实测（cs_agent.share_probe），单卡 9B，结果写 outputs/rt/cs/share_probe.json，日志 $A/cs_probe.log。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/cs_probe.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $S/setup_tau2.sh || { log "CS_PROBE FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_PROBE FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
$PY -u -m cs_agent.share_probe --base $B9 --init $FS/adapters/cs-zero-9b --critic_step $FS/adapters/cs-critic-step-9b --critic_outcome $FS/adapters/cs-critic-outcome-9b \
  --eps outputs/rt/cs/sample_r0/base9_train.jsonl --out outputs/rt/cs/share_probe.json >> $LOG 2>&1 || log "CS_PROBE FAILED: probe"
cp $LOG outputs/rt/cs/ 2>/dev/null
log "CS_PROBE_DONE"
