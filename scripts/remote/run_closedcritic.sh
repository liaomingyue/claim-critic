#!/bin/bash
# 闭环验证的并行部分：在另一台机器上训某一组的 critic（默认 gfree），和 run_closedloop.sh 所在机器上的 g0 组同时训。
# 通过共享代码目录（文件存储）里的标记协调：等 <组>.valid.done 出现 → 先写 <组>.critic.done 占位（主机器上的
# run_closedloop.sh 见到它就跳过这一组）→ 选样、建训练集、训练、评测；失败时删掉占位并写 FAILED，由人接手。
# 用法：ARM=gfree bash run_closedcritic.sh（工作目录 = 共享代码目录）。日志 /root/autodl-tmp/closedcritic.log。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
LOG=/root/autodl-tmp/closedcritic.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/closed; B9=$M/Qwen3.5-9B; n=${ARM:-gfree}
MM='{"image":0,"video":0}'
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
fail() { rm -f $O/$n.critic.done; log "$1 FAILED"; exit 1; }

log "waiting for $O/$n.valid.done"
until [ -f $O/$n.valid.done ]; do sleep 60; done
[ -f $O/$n.critic.done ] && { log "$n critic already done or claimed elsewhere"; exit 0; }
touch $O/$n.critic.done   # 占位
name=critic-closed-$n
python -m ragtruth.gen_dpo select --dir $O/$n --random --n 600 --seed 0 --out $O/$n/critic_sel.jsonl >> $LOG 2>&1 || fail select
python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl --sel $O/$n/critic_sel.jsonl \
  --out outputs/rt/claims/closed_$n.jsonl >> $LOG 2>&1 || fail build
sed -e "s|data: .*|data: outputs/rt/claims/closed_$n.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" \
  configs/rt/sft_9b_v2.yaml > configs/rt/exp/$name.yaml
log "train $name"
python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 || fail train
fix_tok $CK/$name; stop
nohup vllm serve $CK/$name --served-model-name $name --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 \
  --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$name.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$name" && break; sleep 10; done
python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
  --style full --max_tokens 512 --out outputs/rt/eval/sft-9b-$name > /root/autodl-tmp/eval-$name.log 2>&1 || fail eval
stop
log "CLOSEDCRITIC_DONE $n"
