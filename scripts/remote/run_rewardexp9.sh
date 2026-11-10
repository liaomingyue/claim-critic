#!/bin/bash
# 第九批（a12 从第 50 步接着训到 100 步，RESUME=<out>/state/step50）；第八批（a12）；第七批（a11）；第六批（a10，100 步）；第五批（a9，100 步）；第四批（a8，150 步）；第三批（a7 + c3 / c4，另存一份免得改正在跑的脚本）；第二批（c3 / c4，接在 run_rewardexp.sh 后面跑，单独一份免得改正在跑的脚本）。奖励设计对比（先把可能的实验列上去；奖励先跑 50 步，怎么改要基于数据分析；跑完先不关机）。
# 起点 gen-seed4-mc（种子里的错句要用 2B 筛一下，只留 2B 不知道是错的；同步做）：第三版 172 条种子里只留 hardpool
# （2B critic 在原回答里判不出、P < 0.5）的那些，单卡 SFT 1 轮。
# 每组：同一起点 gen-seed4-mc、同一提示池（hardpool 示例、指定类型、结合上下句）、50 步、同一批快评提示，只改奖励的一处；
# 训完用两个奖励外的裁判打快评合格样本：9B 金标 critic（骗过 = P < 0.5）、第一版 RM2 rm2v2__mcgen（自然 = P > 0.5）。
#   EXPS="a0 a1 a4 a5 b3" bash run_rewardexp.sh      # a0 = 基线（soft，系数下限 0.4）
# 进奖励的模型（各组相同，除 b3 多一个 rm2b）：rm2=rm2v2__mcgen3（真实性）、rm1=rm1__mcgold（critic）、rm3=rm3__mc（P(真错) = A + B ≥ 0.7 过滤）；
# 原句有没有错用预先算好的 9B 分数。9B 金标 critic 和 rm2v2__mcgen 只做评测、不进奖励。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/rewardexp.log; S=$(dirname $(readlink -f $0))
MC=$FS/models/MiniCPM5-2B; Q9=$FS/models/Qwen3.5-9B; O=outputs/rt/gentarget; SEED=${SEED:-seed5}; INIT=$FS/adapters/gen-$SEED-mc/adapter; CK=$A/ckpt
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
mkdir -p $A/s0 $A/s1
if [ ! -f $INIT/adapter_model.safetensors ] && [ "$SEED" = seed4 ]; then
  python - <<'EOF' | tee -a $LOG
import json
hard = {json.loads(l)["id"] for l in open("outputs/rt/gentarget/hard_pool.jsonl", encoding="utf-8")}
rows = [json.loads(l) for l in open("outputs/rt/gentarget/seed2b/sft_train_v3.jsonl", encoding="utf-8")]
keep = [r for r in rows if r["id"] in hard]
open("outputs/rt/gentarget/seed2b/sft_train_v4.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in keep))
import collections
print("SEED4", json.dumps({"v3": len(rows), "kept_2b_missed": len(keep), "type": dict(collections.Counter(r["type"] for r in keep)),
                           "task": dict(collections.Counter(r["task_type"] for r in keep)), "mechanism": dict(collections.Counter(r["mechanism"] for r in keep).most_common())}))
EOF
  log "train gen-seed4-mc"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_seed4_mc.yaml --mode coldstart > $A/gen-seed4-mc.log 2>&1 || { log "gen-seed4-mc FAILED"; log "REWARDEXP_DONE"; exit 1; }
  mkdir -p $FS/adapters/gen-seed4-mc && rsync -a $CK/gen-seed4-mc/adapter $FS/adapters/gen-seed4-mc/ || { log "save FAILED"; log "REWARDEXP_DONE"; exit 1; }
  log "trained gen-seed4-mc $(grep -o "'train_loss': '[0-9.]*'" $A/gen-seed4-mc.log | tail -1)"
fi
if [ ! -f $INIT/adapter_model.safetensors ]; then   # 第五版种子（151 条，本地组装好传上来）
  log "train gen-$SEED-mc"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_${SEED}_mc.yaml --mode coldstart > $A/gen-$SEED-mc.log 2>&1 || { log "gen-$SEED-mc FAILED"; log "REWARDEXP_DONE"; exit 1; }
  mkdir -p $FS/adapters/gen-$SEED-mc && rsync -a $CK/gen-$SEED-mc/adapter $FS/adapters/gen-$SEED-mc/ || { log "save FAILED"; log "REWARDEXP_DONE"; exit 1; }
  log "trained gen-$SEED-mc $(grep -o "'train_loss': '[0-9.]*'" $A/gen-$SEED-mc.log | tail -1)"
fi
BASE="--causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --eval_prompts $O/eval_prompts.jsonl --reward target --div_mode place --err_mode ab --gate 0.7 --w_div 0.1
  --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --raw_ratio 4 --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1
  --steps ${STEPS:-50} --lr 5e-6 --n 8 --prompts_per_step 16 --eval_k 4 --save_every 50"
TEMPS=rm1=rm1__mcgold,rm2=rm2v2__mcgen3,rm3=rm3__mc
declare -A ARGS=(
  [a0]="--shape soft --crit_floor 0.4"                       # 基线
  [a1]="--shape soft --crit_floor 0.2"                       # critic 没骗过罚得更重
  [a4]="--shape filter --crit_max 0.5"                      # 只奖励骗过 critic 的
  [a5]="--shape mult"                                       # 自然度 × 骗过 critic，都连续
  [b3]="--shape soft --crit_floor 0.4 --rm2_ens rm2b"       # 两个 RM2 取平均
  [a2]="--shape soft --crit_floor 0.7"
  [c1]="--shape soft --crit_floor 0.4 --gate 0.5"
  [c2]="--shape soft --crit_floor 0.4 --gate 0.85"
  [a6]="--shape boundary --crit_floor 0.05"                                       # critic 越接近 0.5 越好（优先级最高）
  [a7]="--shape boundary2 --base_fail 0.1"                                        # 没过门槛给基线 0.1，明显的错低于基线
  [a8]="--shape boundary3 --base_fail 0.1"                                        # 峰值 0.3、P ≥ 0.9 记 0（在 a7 基础上继续优化）
  [a9]="--shape boundary4 --base_fail 0.1"                                        # 保底按离真错多近给，0.9–0.98 给 0.03
  [a10]="--shape boundary5 --base_fail 0.1 --n 16 --prompts_per_step 8"          # a7 系数 + a9 保底；每 prompt 16 次、每步 8 个 prompt
  [a11]="--shape boundary6 --base_fail 0.1 --n 16 --prompts_per_step 16"         # 4P(1−P) 只在 0.1–0.9 给，两头记 0（阈值 0.1 / 0.9）；保底同 a9；每 prompt 16 次、每步仍 16 个 prompt（耗时增加可以接受）
  [a12]="--shape boundary7 --base_fail 0.1 --obvious_pen 0.1 --n 16 --prompts_per_step 16"   # 明显的错 −0.1，保底按离真错多近给
  [c3]="--shape soft --crit_floor 0.4 --gate 0.7 --gate_lo 0.3"                  # 正确性连续系数
  [c4]="--shape soft --crit_floor 0.4 --gate 0.7 --gate_lo 0.3 --min_sim 0.4"    # 再加：整句换掉的记 0
)
indep() {   # $1 = 目录 $2 = 版本前缀
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --base $Q9 --log $A/vllm-ie9.log critic=$FS/adapters/rm1__rm1_gold_9b-9b/adapter >> $LOG 2>&1 & P9=$!
  A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8002 --base $MC --log $A/vllm-ie2.log rm2v1=$FS/adapters/rm2v2__mcgen-mc/adapter >> $LOG 2>&1 & P2=$!
  wait $P9 && wait $P2 || { log "indep vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_indep_eval --run $1 --prefix $2 --steps 0,${STEPS:-50} --url9b http://127.0.0.1:8001/v1 --url2b http://127.0.0.1:8002/v1 >> $LOG 2>&1 || log "indep $1 FAILED"
  free_gpus
}
for e in ${EXPS:-a0 a1 a4 a5 b3}; do
  R=$O/grpo_${SEED}_$e
  if [ ! -f $R/quick_eval.jsonl ] || [ "$(grep -c '"step": '${STEPS:-50}',' $R/quick_eval.jsonl)" = 0 ]; then
    free_gpus
    EXTRA=""; [[ "${ARGS[$e]}" == *rm2_ens* ]] && EXTRA="rm2b=$FS/adapters/rm2v2__mcgen2-mc/adapter"
    A=$A bash $S/serve_loras.sh --gpu 1 --port 8000 --base $MC --log $A/vllm-rx.log --priority \
      rm1=$FS/adapters/rm1__mcgold-mc/adapter rm2=$FS/adapters/rm2v2__mcgen3-mc/adapter rm3=$FS/adapters/rm3__mc-mc/adapter policy_v0=$INIT $EXTRA >> $LOG 2>&1 || { log "exp $e vllm FAILED"; continue; }
    T=$TEMPS; [ -n "$EXTRA" ] && T=$T,rm2b=rm2v2__mcgen2
    log "exp $e ($SEED): ${ARGS[$e]}"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE --temps $T ${ARGS[$e]} --version_prefix ${SEED}${e}_ --out $R ${RESUME:+--resume $RESUME} >> $A/rewardexp_${SEED}_$e.log 2>&1 \
      && log "exp $e done" || log "exp $e FAILED"
    grep -h QUICK_EVAL $A/rewardexp_${SEED}_$e.log | cut -c1-450 | sed "s/^/[$e] /" | tee -a $LOG
    cp $A/rewardexp_${SEED}_$e.log $R/ 2>/dev/null
  fi
  indep $R ${SEED}${e}_
done
log "REWARDEXP_DONE"
