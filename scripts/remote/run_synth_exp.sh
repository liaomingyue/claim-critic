#!/bin/bash
# 5090 上的合成错误样本实验：等格式对比实验跑完 → 35B 合成并复核 → e3 critic 判识破率 → 三组训练集
# （direct / critic / random，都在 e3 训练集上成对加入）→ 依次训练并评测 0.6B。每一步写 /root/autodl-tmp/syn.log。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/syn.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models

serve() {  # serve <ckpt> <name> <extra args...>
  pkill -f '[v]llm serve'; sleep 5
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len 16384 --gpu-memory-utilization 0.9 --port 8000 "${@:3}" \
    > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; return 1
}
stop() { pkill -f '[v]llm serve'; sleep 10; }

# 格式对比实验占着 GPU：等它的进程退出、日志最后一行是 ALL_DONE（exp.log 里有上一轮留下的 ALL_DONE，只看 grep 不够）
sleep 60
until ! pgrep -f '[r]un_format_exp' >/dev/null && tail -1 /root/autodl-tmp/exp.log | grep -q ALL_DONE; do sleep 30; done
stop
mkdir -p outputs/rt/synth

if [ ! -f outputs/rt/synth/r1.jsonl ]; then
  log "synth start"
  serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 gen --limit-mm-per-prompt '{"image":0,"video":0}' || exit 1
  python -m ragtruth.synth --n ${SYN_N:-3000} --out outputs/rt/synth/r1.jsonl --concurrency 64 \
    > /root/autodl-tmp/synth.log 2>&1 || { log "synth FAILED"; exit 1; }
  log "synth done: $(tail -3 /root/autodl-tmp/synth.log | head -1)"
fi

if [ ! -f outputs/rt/synth/r1_mined.done ]; then
  serve /root/autodl-tmp/ckpt/sft-0.6b-e3 critic || exit 1
  python -m ragtruth.mine mine --synth outputs/rt/synth/r1.jsonl --out outputs/rt/synth/r1_mined.jsonl \
    > /root/autodl-tmp/mine.log 2>&1 || { log "mine FAILED"; exit 1; }
  touch outputs/rt/synth/r1_mined.done
  log "mine done: $(tail -1 /root/autodl-tmp/mine.log)"
fi
stop

for a in direct critic random; do
  python -m ragtruth.mine mix --arm $a --synth outputs/rt/synth/r1.jsonl --mined outputs/rt/synth/r1_mined.jsonl \
    --out outputs/rt/claims/syn_r1_$a.jsonl >> $LOG 2>&1
done

for a in direct critic random; do
  if [ -f outputs/rt/eval/sft-0.6b-syn-$a/metrics.json ]; then log "skip $a (done)"; continue; fi
  log "train syn-$a"
  python -u -m train.sft --config configs/rt/exp/sft_0.6b_syn_$a.yaml --mode coldstart \
    > /root/autodl-tmp/sft-0.6b-syn-$a.log 2>&1 || { log "train syn-$a FAILED"; continue; }
  serve /root/autodl-tmp/ckpt/sft-0.6b-syn-$a sft-0.6b-syn-$a && \
    python -m ragtruth.eval --model sft-0.6b-syn-$a --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
      --style full --max_tokens 512 --out outputs/rt/eval/sft-0.6b-syn-$a > /root/autodl-tmp/eval-sft-0.6b-syn-$a.log 2>&1
  stop
  log "evaluated syn-$a"
done
log "SYN_DONE"
