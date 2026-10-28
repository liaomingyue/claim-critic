#!/bin/bash
# 5090 上的 9B 合成错误样本实验：等格式诊断跑完 → 9B v2 critic 判合成错误句的识破率（难样本由它自己挑）
# → 三组训练集（direct / critic / random，在 9B v2 训练集上成对加入）→ 依次训练并评测。每一步写 /root/autodl-tmp/syn9b.log。
# 合成样本复用 0.6B 实验的 outputs/rt/synth/r1.jsonl。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/syn9b.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/root/autodl-tmp/models/Qwen3.5-9B
MM='{"image":0,"video":0}'

serve() {  # serve <ckpt> <name> <extra args...>
  pkill -f '[v]llm serve'; sleep 5
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 "${@:3}" \
    > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; return 1
}
stop() { pkill -f '[v]llm serve'; sleep 10; }
fix_tok() {  # Qwen3.5：合并后的权重缺分词器 / 处理器配置，从基座补齐
  cp $B/tokenizer_config.json $B/tokenizer.json $B/vocab.json $B/merges.txt $B/chat_template.jinja \
     $B/preprocessor_config.json $B/video_preprocessor_config.json "$1"/ 2>/dev/null
}

# ARMS：这台机器跑哪几组（多台机器分工时，direct 不需要识破率，可以单独放到别的机器上）
ARMS=${ARMS:-critic random direct}
# 同一台机器上排在前面的实验占着 GPU：等它们的进程退出。NOWAIT=1（新开的机器）直接开始
if [ -z "${NOWAIT:-}" ]; then
  sleep 60
  until ! pgrep -f '[r]un_(synth|diag)_exp' >/dev/null; do sleep 30; done
fi

if [[ "$ARMS" =~ critic|random ]] && [ ! -f outputs/rt/synth/r1_mined_9b.done ]; then
  log "mine with 9B v2"
  serve /root/autodl-tmp/ckpt/sft-9b-v2-notips critic --limit-mm-per-prompt "$MM" || exit 1
  python -m ragtruth.mine mine --synth outputs/rt/synth/r1.jsonl --out outputs/rt/synth/r1_mined_9b.jsonl \
    > /root/autodl-tmp/mine9b.log 2>&1 || { log "mine9b FAILED"; exit 1; }
  touch outputs/rt/synth/r1_mined_9b.done
  log "mine9b done: $(tail -1 /root/autodl-tmp/mine9b.log)"
fi
stop

for a in $ARMS; do
  MINED=""; [ $a != direct ] && MINED="--mined outputs/rt/synth/r1_mined_9b.jsonl"
  python -m ragtruth.mine mix --arm $a --base outputs/rt/claims/train_sub35_r3.jsonl --synth outputs/rt/synth/r1.jsonl \
    $MINED --out outputs/rt/claims/syn9b_r1_$a.jsonl >> $LOG 2>&1
done

# 默认先跑 critic / random 两组核心对照，direct 最后
for a in $ARMS; do
  if [ -f outputs/rt/synth/STOP_9B ]; then log "STOP_9B found, stop before syn9b-$a"; break; fi
  name=sft-9b-syn-$a ckpt=/root/autodl-tmp/ckpt/sft-9b-syn-$a
  if [ -f outputs/rt/eval/$name/metrics.json ]; then log "skip $name (done)"; continue; fi
  log "train $name"
  python -u -m train.sft --config configs/rt/exp/sft_9b_syn_$a.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 \
    || { log "train $name FAILED"; continue; }
  fix_tok $ckpt
  serve $ckpt $name --limit-mm-per-prompt "$MM" && \
    python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
      --style full --max_tokens 512 --out outputs/rt/eval/$name > /root/autodl-tmp/eval-$name.log 2>&1
  stop
  log "evaluated $name"
done
log "SYN9B_DONE"
