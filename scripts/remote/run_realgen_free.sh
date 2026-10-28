#!/bin/bash
# 自由选择版的真实错误 SFT 生成器（2026-10-28 用户提出：句子、类型、改法都交给生成器自己判断）。
# 接在 run_realgen.sh（指定版）之后，在同一台机器上跑，复用它的修复结果（outputs/rt/realgen/repair.jsonl）。
# 1. build --free：输入修复后的整条回答，输出真实的句号、类型、改法；没修好的句子从上下文删掉（不再整条回答丢掉）；
# 2. SFT 得 G_free（ckpt/gen-sft-free）；
# 3. 评测：G0 和 G_free 都用自由版 prompt，在指定版评测的同一批回答上各采样 4 条 → 35B 判有效 → D1 真实性 → C2 难度；
#    再算各生成器的错误分布（位置、类型、改法）和真实分布的距离。
# 日志 /root/autodl-tmp/realgen.log，结束写 REALGEN_FREE_DONE。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/realgen.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/realgen; B9=$M/Qwen3.5-9B
MM='{"image":0,"video":0}'
G="python -m ragtruth.gen_dpo"
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

rm -rf $CK/gen-dpo-r2   # 本机的 G2 副本评测已用完（主实例上有适配器），给 G_free 腾盘
if ! done_mark free.sft; then
  python -m ragtruth.realgen build --repair $O/repair.jsonl --out_dir $O --free --suffix _free >> $LOG 2>&1 \
    || { log "free build FAILED"; exit 1; }
  python -u -m train.sft --config configs/rt/sft_gen_free_9b.yaml --mode coldstart > /root/autodl-tmp/gen-sft-free.log 2>&1 \
    || { log "free SFT FAILED"; exit 1; }
  fix_tok $CK/gen-sft-free
  mark free.sft
fi

GENS="g0free:$B9 gfree:$CK/gen-sft-free"
for gp in $GENS; do
  n=${gp%%:*}; p=${gp#*:}; D=$O/eval_$n
  if ! done_mark eval_$n.sample; then
    fix_tok $p; serve $p gen 16384
    $G sample --dir $D --free --resp_from $O/eval_gsft/samples.jsonl --k 4 --variants 1 >> $LOG 2>&1 \
      || { log "sample $n FAILED"; exit 1; }
    mark eval_$n.sample
  fi
done
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
for n in g0free gfree; do
  done_mark eval_$n.valid || { $G valid --dir $O/eval_$n --model judge >> $LOG 2>&1 && mark eval_$n.valid; }
done
fix_tok $CK/discrim-9b; serve $CK/discrim-9b discrim
for n in g0free gfree; do
  done_mark eval_$n.d1 || { $G realism --dir $O/eval_$n --model discrim --tag d1 >> $LOG 2>&1 && mark eval_$n.d1; }
done
fix_tok $CK/critic-gan-r2; serve $CK/critic-gan-r2 critic
for n in g0free gfree; do
  done_mark eval_$n.hard || { $G hard --dir $O/eval_$n --model critic --tag c2 >> $LOG 2>&1 && mark eval_$n.hard; }
done
stop
python -m ragtruth.realgen dist --repair $O/repair.jsonl --out $O/dist.json \
  --dirs $O/eval_g0 $O/eval_g2 $O/eval_gsft $O/eval_g0free $O/eval_gfree >> $LOG 2>&1
log "REALGEN_FREE_DONE"
