#!/bin/bash
# 放大评测（a6 评测是否骗过 critic，核心看离 0.5 附近的样本是不是多了，这个最重要）：
# 每组快评只有 144 个 prompt × 4 次，同一起点第 0 步的边界条数在 4–16 之间，比不出差别。这里每个 prompt 采 16 次：
#   CKPTS="v0=v0 a6=outputs/rt/gentarget/grpo_seed5_a6/ckpt/step50" TAG=p1 bash run_bigeval.sh
# 2B 服务上的 rm1 / rm2 / rm3 和训练时一样打分（P(真错) = A + B），再用 9B 金标 critic（GPU0）和第一版 RM2（GPU1）补独立分，
# 最后按边界口径（2B P(判出) 0.3–0.7、9B 判得出）出表。名字的序号就是 quick_samples 里的 step。
# 双卡（评测都用多卡跑，更快）：评测 prompt 按 id 哈希对半分，两张卡各起一个同样的服务各评一半，最后把 quick_samples 拼起来。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/bigeval.log; S=$(dirname $(readlink -f $0))
MC=$FS/models/MiniCPM5-2B; Q9=$FS/models/Qwen3.5-9B; O=outputs/rt/gentarget; SEED=${SEED:-seed5}; INIT=$FS/adapters/gen-$SEED-mc/adapter
OUT=$O/bigeval_${TAG:-p1}; K=${EVAL_K:-16}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
mkdir -p $A/s0 $A/s1 $OUT/h0 $OUT/h1
free_gpus
python - "$O/eval_prompts.jsonl" "$OUT" <<'EOF'
import json, sys, zlib
src, out = sys.argv[1], sys.argv[2]
fs = [open(f"{out}/h{g}/eval_prompts.jsonl", "w", encoding="utf-8") for g in (0, 1)]
for l in open(src, encoding="utf-8"):
    fs[zlib.crc32(json.loads(l)["id"].encode()) % 2].write(l)
EOF
for g in 0 1; do
  A=$A/s$g bash $S/serve_loras.sh --gpu $g --port 800$g --base $MC --log $A/vllm-be$g.log \
    rm1=$FS/adapters/rm1__mcgold-mc/adapter rm2=$FS/adapters/rm2v2__mcgen3-mc/adapter rm3=$FS/adapters/rm3__mc-mc/adapter policy_v0=$INIT >> $LOG 2>&1 & eval "SV$g=\$!"
done
wait $SV0 && wait $SV1 || { log "bigeval vllm FAILED"; exit 1; }
log "bigeval $TAG: $CKPTS (k=$K, 2 GPUs)"
for g in 0 1; do
  python -u -m train.gen_grpo --causal 1 --base $MC --prompts $O/prompts.jsonl --eval_prompts $OUT/h$g/eval_prompts.jsonl --reward target --vllm_url http://127.0.0.1:800$g/v1 \
    --temps rm1=rm1__mcgold,rm2=rm2v2__mcgen3,rm3=rm3__mc --err_mode ab --gate 0.7 --eval_k $K --eval_minutes 0 --version_prefix big_ --out $OUT/h$g \
    --eval_ckpts $CKPTS >> $A/bigeval_${TAG}_h$g.log 2>&1 & eval "EV$g=\$!"
done
wait $EV0 || log "bigeval eval h0 FAILED"; wait $EV1 || log "bigeval eval h1 FAILED"
N=$(echo $CKPTS | wc -w); STEPS=$(seq -s, 0 $((N-1)))
for k in $(seq 0 $((N-1))); do cat $OUT/h0/quick_samples_big_s$k.jsonl $OUT/h1/quick_samples_big_s$k.jsonl > $OUT/quick_samples_big_s$k.jsonl; done
grep -h QUICK_EVAL $A/bigeval_${TAG}_h*.log | cut -c1-300 | sed 's/^/(half) /' | tee -a $LOG
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --base $Q9 --log $A/vllm-ie9.log critic=$FS/adapters/rm1__rm1_gold_9b-9b/adapter >> $LOG 2>&1 & P9=$!
A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8002 --base $MC --log $A/vllm-ie2.log rm2v1=$FS/adapters/rm2v2__mcgen-mc/adapter >> $LOG 2>&1 & P2=$!
wait $P9 && wait $P2 || { log "indep vllm FAILED"; exit 1; }
python -u -m ragtruth.gen_indep_eval --run $OUT --prefix big_ --steps $STEPS --url9b http://127.0.0.1:8001/v1 --url2b http://127.0.0.1:8002/v1 >> $LOG 2>&1 || log "bigeval indep FAILED"
free_gpus
python -u -m ragtruth.boundary_metric $OUT | sed "s/^/BOUNDARY /" | tee -a $LOG
log "BIGEVAL_DONE $TAG"
