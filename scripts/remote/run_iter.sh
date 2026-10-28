#!/bin/bash
# critic 循环迭代实验的一条序列（9B）：bash run_iter.sh <random|critic> [轮数，默认 3]
# 每轮：等本轮合成池 → 用本序列当前 critic 判识破率 → 选样本（ragtruth.iterate select）→ 累积建训练集
# → 从基座重新训练 → 评测；本轮训出的模型就是下一轮的 critic。第 1 轮的 critic 是 9B v2。
# POOL_SRC=<主实例 ssh 主机>:<端口> 时（加开的机器），合成池从主实例经同地区网络拉取。日志 /root/autodl-tmp/iter_<序列>.log。
set -u
SER=$1; ROUNDS=${2:-3}
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/iter_$SER.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/root/autodl-tmp/models/Qwen3.5-9B; CK=/root/autodl-tmp/ckpt; S=outputs/rt/synth; I=outputs/rt/iter
MM='{"image":0,"video":0}'
mkdir -p $I $S

serve() {  # serve <ckpt> <name>
  pkill -f '[v]llm serve'; sleep 5; pkill -9 -f '[V]LLM::EngineCore'; sleep 3
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; return 1
}
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }   # 强杀父进程后 EngineCore 会成孤儿继续占显存
fix_tok() { cp $B/tokenizer_config.json $B/tokenizer.json $B/vocab.json $B/merges.txt $B/chat_template.jinja \
              $B/preprocessor_config.json $B/video_preprocessor_config.json "$1"/ 2>/dev/null; }
wait_pool() {  # 等第 $1 轮合成池；加开的机器从主实例拉
  until [ -f $S/r$1_ft.done ]; do
    if [ -n "${POOL_SRC:-}" ]; then
      h=${POOL_SRC%:*}; p=${POOL_SRC#*:}
      rsync -a -e "ssh -p $p -o StrictHostKeyChecking=no" "root@$h:/root/autodl-tmp/critic_model/$S/r$1_ft.*" $S/ 2>/dev/null
    fi
    [ -f $S/r$1_ft.done ] || sleep 60
  done
}
# 同一台机器上排在前面的 GPU 任务（合成池）先跑完
[ -z "${NOWAIT:-}" ] && { sleep 30; until ! pgrep -f '[r]un_pools' >/dev/null; do sleep 30; done; }

prev=$CK/sft-9b-v2-notips
SELS=""
for r in $(seq 1 $ROUNDS); do
  name=sft-9b-iter-$SER-r$r
  SELS="$SELS $I/${SER}_r${r}_sel.jsonl"
  if [ -f outputs/rt/eval/$name/metrics.json ]; then log "skip round $r (done)"; prev=$CK/$name; continue; fi
  wait_pool $r
  log "round $r: mine with $(basename $prev)"
  fix_tok $prev
  serve $prev critic || exit 1
  python -m ragtruth.mine mine --synth $S/r${r}_ft.jsonl --out $I/${SER}_r${r}_mined.jsonl \
    > /root/autodl-tmp/iter_${SER}_mine_r$r.log 2>&1 || { log "mine r$r FAILED"; exit 1; }
  stop
  python -m ragtruth.iterate select --series $SER --pool $S/r${r}_ft.jsonl --mined $I/${SER}_r${r}_mined.jsonl \
    --n ${PER_ROUND:-600} --seed $r --out $I/${SER}_r${r}_sel.jsonl >> $LOG 2>&1
  python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl --sel $SELS \
    --out outputs/rt/claims/iter_${SER}_r$r.jsonl >> $LOG 2>&1
  sed -e "s|data: .*|data: outputs/rt/claims/iter_${SER}_r$r.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" \
    configs/rt/sft_9b_v2.yaml > configs/rt/exp/$name.yaml
  log "round $r: train $name"
  python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 \
    || { log "train $name FAILED"; exit 1; }
  fix_tok $CK/$name
  serve $CK/$name $name && \
    python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
      --style full --max_tokens 512 --out outputs/rt/eval/$name > /root/autodl-tmp/eval-$name.log 2>&1
  stop
  log "round $r: evaluated $name"
  prev=$CK/$name
done
log "ITER_DONE"
