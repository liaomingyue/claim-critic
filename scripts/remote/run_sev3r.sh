#!/bin/bash
# 2026-11-08 用户：改动描述只用规则（逐词 diff 列出），不用模型写。sev__v3r = sev__v3 去掉 9B 描述、只给规则改动清单；训完和 v3 同批打分比较。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sev3r.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until sed -n '/two-step/,$p' $A/genv2.log | grep -q GENV2_DONE; do sleep 30; done
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
pkill -f '[v]llm serve'; sleep 5
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
[ -f outputs/rt/rm/sev__v3r/eval/report.json ] || RM=sev__v3r GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1
pkill -f '[v]llm serve'; sleep 5
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/sev__v3r/eval/report.json'))['temperature'])") || { log "sev__v3r FAILED"; exit 1; }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-v3r.log v3=$FS/adapters/sev__v3r-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --temps v3=$T --out gen_v2_label/model_scores_v3r.jsonl >> $LOG 2>&1 || log "score FAILED"
pkill -f '[v]llm serve'
log "SEV3R_DONE"
