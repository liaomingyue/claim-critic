#!/bin/bash
# 2026-11-09 用户：试试不配原句。rm1__core_r2 + Claude 核实的 285 条范围偏移错句（不加同位置原句）→ rm1__core_r2_sccl_no → 三套验收打分（本地 critic_cap3 两个口径 + 按类型）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/scopeno.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
name=rm1__core_r2_sccl_no
python -u -m ragtruth.scope_critic build --no_orig --arms cl >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "BUILT $(grep -a SCOPE_BUILD $LOG | tail -1)"
if [ ! -f outputs/rt/rm/$name/eval/report.json ]; then
  free_gpu; RM=$name GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/$name/eval/report.json ] || { log "train FAILED"; exit 1; }
fi
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$name/eval/report.json'))['temperature'])"); log "TRAINED T=$T"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-scno.log c=$FS/adapters/$name-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
PIDS=""
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
  python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$name.json >> $LOG 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
free_gpu
log "SCOPENO_DONE"
