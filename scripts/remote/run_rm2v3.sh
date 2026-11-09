#!/bin/bash
# 2026-11-08 用户：RM2 也要训一版 → rm2__v3（标签 = Claude 在生成器 v2 采样上标的「自然」）；排在 sev__v3r 之后；训完给 1200 条打分。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/rm2v3.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -qE 'SEV3R_DONE|sev__v3r FAILED|serve FAILED' $A/sev3r.log 2>/dev/null; do sleep 30; done
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
pkill -f '[v]llm serve'; sleep 5
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
[ -f outputs/rt/rm/rm2__v3/eval/report.json ] || RM=rm2__v3 GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1
pkill -f '[v]llm serve'; sleep 5
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm2__v3/eval/report.json'))['temperature'])") || { log "rm2__v3 FAILED"; exit 1; }
log "TRAINED rm2__v3 T=$T"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-rm2v3.log rm2v3=$FS/adapters/rm2__v3-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --temps rm2v3=$T --out gen_v2_label/model_scores_rm2v3.jsonl >> $LOG 2>&1 || log "score FAILED"
pkill -f '[v]llm serve'
log "RM2V3_DONE"
