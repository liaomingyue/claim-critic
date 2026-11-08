#!/bin/bash
# 2026-11-06 用户：入库样本要 Claude 确认是真错。rm1__core_r2 = 种子集 4604 行 + Claude 核验为真错的盲区（critic P < 0.3）/ 边界（0.3–0.7）改写句 581 条
# （类型：Claude 判解释对时用改写模型自报的类型，否则按 Claude 的机制换算）+ 每条配同位置原句（A）。名单 outputs/rt/core/r2_ids.tsv（本地生成、随代码上传）。
#   ① critic_harvest --ids → rm1__core_r2 → run_mcrm.sh（校准 + RAGTruth++）
#   ② 三套验收回答打分（本地 critic_cap3 算）+ 接着训第 100 步回答上的盲区召回
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/r2.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
if [ ! -f outputs/rt/rm/rm1__core_r2/eval/report.json ]; then
  python -u -m ragtruth.critic_harvest --harvest outputs/rt/core/harvest_r1.jsonl --base rm1__core --name rm1__core_r2 --ids outputs/rt/core/r2_ids.tsv \
    --gate 0 --min_nat 0 --max_add 1000 --per_prompt 8 >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
  grep -a CRITIC_HARVEST $LOG | tail -1 | cut -c1-600
  RM=rm1__core_r2 GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/rm1__core_r2/eval/report.json ] || { log "train FAILED"; exit 1; }
fi
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__core_r2/eval/report.json'))['temperature'])")
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2.log c2=$FS/adapters/rm1__core_r2-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
  [ -f $d/critic_rm1__core_r2.json ] || python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c2 --temp $T --out $d/critic_rm1__core_r2.json >> $LOG 2>&1
done
K=outputs/rt/core/loop2_cont/eval
for m in v0 s100; do for s in a b c; do [ -f $K/${m}_${s}_r2.json ] || python -u -m ragtruth.critic_score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model c2 --temp $T --out $K/${m}_${s}_r2.json >> $LOG 2>&1; done; done
free_gpu
log "R2_DONE"
