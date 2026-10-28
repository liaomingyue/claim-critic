#!/bin/bash
# 统一判别器 D（2026-10-28 用户提出：正确判别器和错误判别器合成一个，核心都是「模型自己写的 vs 被改过的」）。
# 自然类 = RAGTruth 正确句 + 真实错句；合成类 = 35B 修复出的正确句 + 各生成器造的错句；四类配平，中性 prompt。
# 训练（同 D1 超参）→ 留出集打分（标签词 logprob），按「正确句 / 错句」分开报 AUC。
# 用法：bash run_unidisc.sh（工作目录 = 共享代码目录）。日志 /root/autodl-tmp/unidisc.log，结束写 UNIDISC_DONE。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
LOG=/root/autodl-tmp/unidisc.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/root/autodl-tmp/ckpt; B9=/root/autodl-tmp/models/Qwen3.5-9B; O=outputs/rt/discrim_uni
MM='{"image":0,"video":0}'
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }

GD=""
for d in outputs/rt/gan/r1 outputs/rt/gan/r2 outputs/rt/realgen/eval_g0 outputs/rt/realgen/eval_g2 outputs/rt/realgen/eval_gsft \
         outputs/rt/realgen/eval_gfree outputs/rt/realgen/eval_g0free outputs/rt/closed/g0 outputs/rt/closed/gfree; do
  [ -f $d/valid.jsonl ] && GD="$GD $d"
done
python -m ragtruth.discrim build --unified --repair outputs/rt/realgen/repair.jsonl --gen_dirs $GD --cap 2000 \
  --out_dir $O >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
sed -e "s|data: .*|data: $O/train.jsonl|" -e "s|output_dir: .*|output_dir: $CK/discrim-uni|" \
  configs/rt/discrim_9b.yaml > configs/rt/exp/discrim_uni.yaml
log "train discrim-uni"
python -u -m train.sft --config configs/rt/exp/discrim_uni.yaml --mode coldstart > /root/autodl-tmp/discrim-uni.log 2>&1 \
  || { log "train FAILED"; exit 1; }
cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} $CK/discrim-uni/ 2>/dev/null
stop
nohup vllm serve $CK/discrim-uni --served-model-name discrim --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 \
  --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-discrim-uni.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q discrim && break; sleep 10; done
python -m ragtruth.discrim score --rows $O/heldout.jsonl --out $O/heldout_scores.jsonl --eval --logprob >> $LOG 2>&1 \
  || { log "score FAILED"; exit 1; }
stop
log "UNIDISC_DONE"
