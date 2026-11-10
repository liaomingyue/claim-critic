#!/bin/bash
# 9B 评测器和 Claude 金标的一致性：给三套验收回答（grpo300 / grpo300b / grpo300c，都有 Claude 逐句金标）补 9B 金标 critic 的逐句打分，
# 写到各自目录的 critic_scores.json（grpo300 已有的跳过），然后跑 scripts.analysis.eval9b_agreement。
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; LOG=$A/eval9b_agree.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
source scripts/remote/models9b.sh; ensure_9b || { log "9b copy FAILED"; exit 1; }
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__rm1_gold_9b/eval/report.json'))['temperature'])")
J="["; for s in grpo300 grpo300b grpo300c; do J+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"outputs/rt/minicpm/$s/answers.jsonl\",\"out\":\"outputs/rt/minicpm/$s/critic_scores.json\"},"; done
echo "${J%,}]" > $A/eval9b_jobs.json
A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $A/models/Qwen3.5-9B --log $A/vllm-9b.log critic=$FS/adapters/rm1__rm1_gold_9b-9b/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $A/eval9b_jobs.json --concurrency 256 >> $LOG 2>&1 || { log "score FAILED"; exit 1; }
pkill -f '[v]llm serve'
python -m scripts.analysis.eval9b_agreement --out outputs/rt/eval9b_agreement.json >> $LOG 2>&1
log "EVAL9B_AGREE_DONE"
