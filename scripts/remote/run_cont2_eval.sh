#!/bin/bash
# cont2 训完后的评测（2026-11-06）：等 R3FULL_DONE → rm1__core_r2 给 cont2 的 v0/s50/s100 打分 → 盲区召回 + 9B 配对比较
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/cont2_eval.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
until grep -q 'R3FULL_DONE\|FAILED' $A/r3full_core_cont2.log; do sleep 30; done
grep -q R3FULL_DONE $A/r3full_core_cont2.log || { log "cont2 FAILED, skip"; exit 1; }
free_gpu
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__core_r2/eval/report.json'))['temperature'])")
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-c2e.log c2=$FS/adapters/rm1__core_r2-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
K=outputs/rt/core/loop2_cont2/eval
for m in v0 s50 s100; do for s in a b c; do python -u -m ragtruth.critic_score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model c2 --temp $T --out $K/${m}_${s}_r2.json >> $LOG 2>&1 & done; done; for p in $(jobs -p); do [ "$p" = "$KA" ] || wait $p; done
free_gpu
python outputs/blind_recall.py $K r2 v0 s50 s100 | tee -a $LOG
K1=outputs/rt/core/loop2_cont/eval; C=outputs/rt/core/loop2_coreonly/eval
python -m ragtruth.pol9b_cmp --arm start=$K:v0 --arm c50=$K:s50 --arm c100=$K:s100 --arm cont1_start=$K1:v0 --arm co200=$C:s200 --arm sft=$C:v0 --sets a b c --base start --out $K/pol9b_cmp.json > /dev/null 2>&1 || log "cmp FAILED"
log "CONT2_EVAL_DONE"
