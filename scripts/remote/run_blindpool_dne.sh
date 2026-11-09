#!/bin/bash
# 2026-11-08：sev__gen_dne（不给生成器解释）判真错最好 → 等 run_blindpool.sh 做完，用 dne 也给全部 9B 样本 + 审计样本打分 → gen_audit/sev_dne_scores_all.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/blindpool.log; G=outputs/rt/core/gen_audit
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
until grep -q BLINDPOOL_DONE $LOG 2>/dev/null; do sleep 30; done
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/sev__gen_dne/eval/report.json'))['temperature'])")
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-dne.log s=$FS/adapters/sev__gen_dne-mc/adapter >> $LOG 2>&1 || { log "serve dne FAILED"; exit 1; }
python -u -m ragtruth.sev_filter_score --model s --prompt dne --temp $T --files items_pair.jsonl items_sevnew.jsonl --extra samples_9b_rest.jsonl --out $G/sev_dne_scores_all.jsonl >> $LOG 2>&1 || log "dne score FAILED"
pkill -f '[v]llm serve'
log "BLINDPOOL_DNE_DONE"
