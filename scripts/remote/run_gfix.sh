#!/bin/bash
# 改错生成器 G_fix（大循环里「改错」方向的生成器）：9B LoRA SFT，替换 35B 零样本修复。
# 训练数据：35B 修复成功的真实错句 + 反过来的合成错误（合成错句 → 原句）。
# 评测：另选 300 句没修过、原文不在训练集里的真实错句，G_fix 和 35B 各修一遍（单次、不重试）
# → 同一个 35B 复核 → 同一个 critic（C2）检测修复句（识破率 ≤ 1/4 才算修好）→ 比较三道关的通过率。
# 用法：bash run_gfix.sh（工作目录 = 共享代码目录）。日志 /root/autodl-tmp/gfix.log，结束写 GFIX_DONE。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
LOG=/root/autodl-tmp/gfix.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; B9=$M/Qwen3.5-9B; O=outputs/rt/gfix
MM='{"image":0,"video":0}'
mkdir -p $O
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
serve() {
  stop; fix_tok "$1"
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len ${3:-8192} --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; exit 1
}
done_mark() { [ -f $O/$1.done ]; }
mark() { touch $O/$1.done; log "$1 done"; }
step() { done_mark $1 || { "${@:2}" >> $LOG 2>&1 && mark $1; } || { log "$1 FAILED"; exit 1; }; }
R="python -m ragtruth.realgen"

GD=""
for d in outputs/rt/gan/r1 outputs/rt/gan/r2 outputs/rt/realgen/eval_g0 outputs/rt/realgen/eval_gfree outputs/rt/closed/g0; do
  [ -f $d/valid.jsonl ] && GD="$GD $d"
done
step data $R fixdata --repair outputs/rt/realgen/repair.jsonl --gen_dirs $GD --max_synthetic 1500 --out_dir $O
if ! done_mark sft; then
  sed -e "s|  data: .*|  data: $O/sft_train.jsonl|" -e "s|  output_dir: .*|  output_dir: $CK/gen-fix|" \
    configs/rt/sft_gen_real_9b.yaml > configs/rt/exp/sft_gen_fix.yaml
  log "train gen-fix"
  python -u -m train.sft --config configs/rt/exp/sft_gen_fix.yaml --mode coldstart > /root/autodl-tmp/gen-fix.log 2>&1 \
    || { log "train FAILED"; exit 1; }
  mark sft
fi

# 评测集：300 句新的真实错句（不在 35B 修复集里，原文不在 G_fix 训练集里）；两边同一批
EX="--fresh 300 --exclude outputs/rt/realgen/repair.jsonl --exclude_sources $O/sft_train.jsonl --seed 7 --retries 0 --no_verify"
serve $CK/gen-fix gfix 16384
step fix.gfix $R repair $EX --model gfix --out $O/eval_gfix.jsonl
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
step fix.35b $R repair $EX --model judge --out $O/eval_35b.jsonl
step verify.gfix $R verify --repair $O/eval_gfix.jsonl --out $O/eval_gfix_v.jsonl
step verify.35b $R verify --repair $O/eval_35b.jsonl --out $O/eval_35b_v.jsonl
serve $CK/critic-gan-r2 critic
step critic.gfix $R criticcheck --repair $O/eval_gfix_v.jsonl --out $O/eval_gfix_c.jsonl
step critic.35b $R criticcheck --repair $O/eval_35b_v.jsonl --out $O/eval_35b_c.jsonl
stop
log "G_fix: $(cat $O/eval_gfix_c.summary.json | tr -d '\n ') | 35B: $(cat $O/eval_35b_c.summary.json | tr -d '\n ')"
log "GFIX_DONE"
