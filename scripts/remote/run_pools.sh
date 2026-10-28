#!/bin/bash
# 迭代实验的 3 轮合成池（主实例上跑，35B）：r1 = 已有合成结果 + 全文核对；r2、r3 = 新句子合成（不重复之前改过的句子）
# + 全文核对。每轮池写好后放一个 rK_ft.done，两条序列（run_iter.sh）据此开始该轮。日志 /root/autodl-tmp/pools.log。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/pools.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
S=outputs/rt/synth

pkill -f '[v]llm serve'; sleep 5; pkill -9 -f '[V]LLM::EngineCore'; sleep 3
nohup vllm serve /root/autodl-tmp/models/Qwen3.5-35B-A3B-GPTQ-Int4 --served-model-name gen --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --port 8000 --limit-mm-per-prompt '{"image":0,"video":0}' > /root/autodl-tmp/vllm-gen.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q gen && break; sleep 10; done

ft() {  # ft <轮次>：全文核对过滤
  [ -f $S/r$1_ft.done ] && return
  python -m ragtruth.fulltext --synth $S/r$1.jsonl --out $S/r$1_ft.jsonl >> /root/autodl-tmp/fulltext.log 2>&1 \
    || { log "fulltext r$1 FAILED"; exit 1; }
  touch $S/r$1_ft.done
  log "pool r$1 ready: $(tail -1 /root/autodl-tmp/fulltext.log)"
}

ft 1
EX="$S/r1.raw.jsonl"
for r in 2 3; do
  if [ ! -f $S/r$r.jsonl ]; then
    python -m ragtruth.synth --n 3000 --seed $r --exclude $EX --out $S/r$r.jsonl --concurrency 64 \
      > /root/autodl-tmp/synth_r$r.log 2>&1 || { log "synth r$r FAILED"; exit 1; }
    log "synth r$r: $(grep '^synthesized' /root/autodl-tmp/synth_r$r.log)"
  fi
  ft $r
  EX="$EX $S/r$r.raw.jsonl"
done
pkill -f '[v]llm serve'; sleep 10
log "POOLS_DONE"
