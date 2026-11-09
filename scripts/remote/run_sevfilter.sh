#!/bin/bash
# 2026-11-08 用户：sev 模型就是判「是不是真错、解释对不对」的 → 量它当过滤器（不当奖励）的精确率。
# sev__core（MiniCPM5-2B）给已审计的 2B / 9B / 种子集样本 + 9B 全部合格样本打分 → outputs/rt/core/gen_audit/sev_scores.jsonl（本地对 Claude 结论算精确率）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sevfilter.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/sev__core/eval/report.json'))['temperature'])") || { log "no sev__core report"; exit 1; }
log "sev__core T=$T"
python -c "import json;r=json.load(open('outputs/rt/rm/sev__core/eval/report.json'));print(json.dumps({k:v for k,v in r.items() if k!='rows'},ensure_ascii=False)[:3000])" | tee -a $LOG
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-sev.log sev=$FS/adapters/sev__core-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.sev_filter_score --url http://127.0.0.1:8000/v1 --model sev --temp $T >> $LOG 2>&1 || { log "score FAILED"; exit 1; }
pkill -f '[v]llm serve'
log "SEVFILTER_DONE"
