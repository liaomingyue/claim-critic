#!/bin/bash
# 5090 上的格式诊断实验：等合成实验跑完 → ④ e4（0.6B 先判定后证据和理由）→ B（9B 只判定）。每一步写 /root/autodl-tmp/diag.log。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/diag.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

serve() {  # serve <ckpt> <name> <extra args...>
  pkill -f '[v]llm serve'; sleep 5
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 "${@:3}" \
    > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; return 1
}
stop() { pkill -f '[v]llm serve'; sleep 10; }

# 同一台机器上排在前面的实验占着 GPU：等它的进程退出。NOWAIT=1（新开的机器，见 bootstrap_worker.sh）直接开始
if [ -z "${NOWAIT:-}" ]; then
  sleep 60
  until ! pgrep -f '[r]un_synth_exp' >/dev/null; do sleep 30; done
fi

# name config ckpt eval-args extra-serve-args
run() {
  local name=$1 cfg=$2 ckpt=$3 ev=$4; shift 4
  if [ -f outputs/rt/eval/$name/metrics.json ]; then log "skip $name (done)"; return; fi
  log "train $name"
  python -u -m train.sft --config $cfg --mode coldstart > /root/autodl-tmp/$name.log 2>&1 || { log "train $name FAILED"; return; }
  if [[ $name == *9b* ]]; then   # Qwen3.5：合并后的权重缺分词器 / 处理器配置，从基座补齐
    B=/root/autodl-tmp/models/Qwen3.5-9B
    cp $B/tokenizer_config.json $B/tokenizer.json $B/vocab.json $B/merges.txt $B/chat_template.jinja \
       $B/preprocessor_config.json $B/video_preprocessor_config.json $ckpt/ 2>/dev/null
  fi
  serve $ckpt $name "$@" && \
    python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think $ev \
      --out outputs/rt/eval/$name > /root/autodl-tmp/eval-$name.log 2>&1
  stop
  log "evaluated $name"
}

run sft-0.6b-e4 configs/rt/exp/sft_0.6b_e4.yaml /root/autodl-tmp/ckpt/sft-0.6b-e4 "--style vfirst --max_tokens 512"
run sft-9b-verdict configs/rt/exp/sft_9b_verdict.yaml /root/autodl-tmp/ckpt/sft-9b-verdict "--style verdict --max_tokens 64" \
  --limit-mm-per-prompt '{"image":0,"video":0}'
log "DIAG_DONE"
