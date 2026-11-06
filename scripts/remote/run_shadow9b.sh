#!/bin/bash
# MiniCPM GRPO 训完补打对照分（用户：训练外的调用用 9B）：两张卡各起一个 9B + 金标 critic 的服务，跑 ragtruth.shadow_rescore。机器不关。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/shadow9b.log; RUN=${RUN:-outputs/rt/minicpm/grpo/run}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
CR=rm1__rm1_gold_9b
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$CR/eval/report.json'))['temperature'])")
pkill -f '[v]llm serve'
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
if [ "${SINGLE_GPU:-0}" = 1 ]; then   # 单卡机（双机模式的训练机，2026-11-04）：只起一个服务
  A=$A bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-sh1.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "vllm FAILED"; exit 1; }
  SURLS=http://127.0.0.1:8000/v1
else
  mkdir -p $A/s0
  A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8101 --base $FS/models/Qwen3.5-9B --log $A/vllm-sh0.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "vllm0 FAILED"; exit 1; } &
  A=$A bash scripts/remote/serve_loras.sh --gpu 1 --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-sh1.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "vllm1 FAILED"; exit 1; }
  wait
  SURLS=http://127.0.0.1:8000/v1,http://127.0.0.1:8101/v1
fi
log "9b critic up, temp $T"
python -u -m ragtruth.shadow_rescore --run $RUN --dev ${DEV:-outputs/rt/minicpm/grpo/dev_ref.jsonl} --url $SURLS --temp $T >> $LOG 2>&1 || log "rescore FAILED"
pkill -f '[v]llm serve'
log "SHADOW9B_ALL_DONE"
