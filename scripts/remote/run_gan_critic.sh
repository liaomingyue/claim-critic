#!/bin/bash
# GAN 式两轮循环的 critic 部分（加开的机器上跑，与主实例上的生成器 DPO 并行）：
# 每轮等主实例放出「本轮选样就绪」→ 拉选样 → 基础数据 + 各轮选样累积建训练集 → 从基座训 critic → 真实测试集评测
# → 把 critic 权重和评测结果推回主实例，并打 .synced 标记（主实例下一轮打难度要用它）。
# 用法：bash run_gan_critic.sh <主实例 ssh 主机> <端口>。日志 /root/autodl-tmp/gan_critic.log，结束写 CRITIC_DONE。
set -u
SRC=$1 PORT=$2
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/gan_critic.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/root/autodl-tmp/ckpt; O=outputs/rt/gan; B9=/root/autodl-tmp/models/Qwen3.5-9B
E="ssh -p $PORT -o StrictHostKeyChecking=no"
MM='{"image":0,"video":0}'
R=/root/autodl-tmp/critic_model
mkdir -p $O
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }

SELS=""
for r in 1 2; do
  name=critic-gan-r$r
  until [ -f $O/r$r/critic_sel.ready ]; do   # 等主实例的选样
    mkdir -p $O/r$r
    rsync -a -e "$E" "root@$SRC:$R/$O/r$r/critic_sel.*" $O/r$r/ 2>/dev/null
    [ -f $O/r$r/critic_sel.ready ] || sleep 60
  done
  SELS="$SELS $O/r$r/critic_sel.jsonl"
  log "round $r: build and train $name"
  python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl --sel $SELS \
    --out outputs/rt/claims/gan_r$r.jsonl >> $LOG 2>&1
  sed -e "s|data: .*|data: outputs/rt/claims/gan_r$r.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" \
    configs/rt/sft_9b_v2.yaml > configs/rt/exp/$name.yaml
  python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 \
    || { log "train FAILED"; exit 1; }
  cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} $CK/$name/ 2>/dev/null
  stop
  nohup vllm serve $CK/$name --served-model-name $name --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$name.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$name" && break; sleep 10; done
  python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
    --style full --max_tokens 512 --out outputs/rt/eval/sft-9b-$name > /root/autodl-tmp/eval-$name.log 2>&1
  stop
  log "round $r: push $name to main"
  rsync -a -e "$E" $CK/$name root@$SRC:$CK/ && \
  rsync -a -e "$E" outputs/rt/eval/sft-9b-$name root@$SRC:$R/outputs/rt/eval/ && \
  rsync -a -e "$E" outputs/rt/claims/gan_r$r.jsonl configs/rt/exp/$name.yaml root@$SRC:$R/outputs/rt/claims/ && \
  $E root@$SRC "touch $CK/$name/.synced" || { log "push FAILED"; exit 1; }
  log "round $r: $name synced"
done
rsync -a -e "$E" outputs/runs/registry.jsonl root@$SRC:$R/outputs/runs/registry_gancritic.jsonl
log "CRITIC_DONE"
