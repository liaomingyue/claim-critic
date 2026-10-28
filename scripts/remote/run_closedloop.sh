#!/bin/bash
# 闭环验证（2026-10-28）：更像真实的合成错误，能不能训出召回更高的 critic？
# 两组，除生成器外完全相同：
#   g0   ：原始 9B，指定版 prompt（随机指定句子、类型、改法）
#   gfree：真实错误 SFT 的自由选择版生成器（ckpt/gen-sft-free）
# 同一批 1,000 条新回答（g0 先抽种子，gfree 用同一批回答），每条采样 4 次 → 35B 判有效
# → 每条回答随机取一条有效改写（不按难度 / 真实性挑），各取 600 条 → 与基础训练集成对合并
# → 从基座训 9B critic（超参同 9B v2）→ RAGTruth 测试集评测。
# 日志 /root/autodl-tmp/closedloop.log，结束写 CLOSEDLOOP_DONE。
set -u
cd ${CODE:-/root/autodl-tmp/critic_model}   # bootstrap_fs.sh 传入共享代码目录
LOG=/root/autodl-tmp/closedloop.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/closed; B9=$M/Qwen3.5-9B
MM='{"image":0,"video":0}'
G="python -m ragtruth.gen_dpo"
mkdir -p $O
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
serve() {
  stop
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len ${3:-8192} --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; exit 1
}
done_mark() { [ -f $O/$1.done ]; }
mark() { touch $O/$1.done; log "$1 done"; }

# 主实例只存了 G_free 的适配器（合并权重按约定不保留）：新机器上先合并回完整权重
if [ ! -f $CK/gen-sft-free/model.safetensors ]; then
  python -u -m train.merge_lora --config configs/rt/sft_gen_free_9b.yaml > /root/autodl-tmp/gen-sft-free-merge.log 2>&1     || { log "G_free merge FAILED"; exit 1; }
fi

# 之前用过的回答 / 句子都不再用作种子
EX="outputs/rt/gan/r1/samples.jsonl outputs/rt/gan/r2/samples.jsonl"
for d in outputs/rt/realgen/eval_*/; do EX="$EX ${d}samples.jsonl"; done

if ! done_mark g0.sample; then
  fix_tok $B9; serve $B9 gen 16384
  $G sample --dir $O/g0 --n_seeds 1000 --k 4 --variants 1 --hard_shots 0 --seed 21 --exclude_samples $EX >> $LOG 2>&1 \
    || { log "g0 sample FAILED"; exit 1; }
  mark g0.sample
fi
if ! done_mark gfree.sample; then
  fix_tok $CK/gen-sft-free; serve $CK/gen-sft-free gen 16384
  $G sample --dir $O/gfree --free --resp_from $O/g0/samples.jsonl --k 4 --variants 1 >> $LOG 2>&1 \
    || { log "gfree sample FAILED"; exit 1; }
  mark gfree.sample
fi
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
for n in g0 gfree; do
  done_mark $n.valid || { $G valid --dir $O/$n --model judge >> $LOG 2>&1 && mark $n.valid; } || { log "$n valid FAILED"; exit 1; }
done
stop

for n in g0 gfree; do
  name=critic-closed-$n
  if ! done_mark $n.critic; then
    $G select --dir $O/$n --random --n 600 --seed 0 --out $O/$n/critic_sel.jsonl >> $LOG 2>&1 || { log "$n select FAILED"; exit 1; }
    python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl --sel $O/$n/critic_sel.jsonl \
      --out outputs/rt/claims/closed_$n.jsonl >> $LOG 2>&1 || { log "$n build FAILED"; exit 1; }
    sed -e "s|data: .*|data: outputs/rt/claims/closed_$n.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" \
      configs/rt/sft_9b_v2.yaml > configs/rt/exp/$name.yaml
    log "train $name"
    python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > /root/autodl-tmp/$name.log 2>&1 \
      || { log "$name train FAILED"; exit 1; }
    fix_tok $CK/$name; serve $CK/$name $name
    python -m ragtruth.eval --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think \
      --style full --max_tokens 512 --out outputs/rt/eval/sft-9b-$name > /root/autodl-tmp/eval-$name.log 2>&1 \
      || { log "$name eval FAILED"; exit 1; }
    stop
    mark $n.critic
  fi
done
log "CLOSEDLOOP_DONE"
