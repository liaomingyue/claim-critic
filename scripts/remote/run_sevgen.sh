#!/bin/bash
# 2026-11-08 用户：sev 训练集要从生成器真实输出里标（Claude 统一审计），把 sev 效果提上去。
#   ① 训 sev__gen（数据 outputs/rt/rm/sev__gen，随代码上传；超参同 sev__core：configs/rt/rm_mc.yaml）→ 校准 + 报告
#   ② 新旧 sev 逐条打分（同一批审计样本），本地按测试切分比过滤精确率：
#      新 sev__gen（新提示词）→ gen_audit/sev2_scores.jsonl；旧 sev__core 补打 items_sevnew → gen_audit/sev_scores_sevnew.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sevgen.log; G=outputs/rt/core/gen_audit
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
# 机器先开、数据后到（2026-11-08 用户：现在就申请 5090）：等本地标完后单独上传的 sev__gen 数据和 Claude 标签
log "waiting for sev__gen data"
until [ -f outputs/rt/rm/sev__gen/UPLOADED ]; do sleep 30; done
log "data ready: $(cat outputs/rt/rm/sev__gen/meta.json | tr -d '
' | cut -c1-400)"
if [ ! -f outputs/rt/rm/sev__gen/eval/report.json ]; then
  RM=sev__gen GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/sev__gen/eval/report.json ] || { log "sev__gen train FAILED"; exit 1; }
fi
rep() { python -c "import json;r=json.load(open('outputs/rt/rm/$1/eval/report.json'));print(r['temperature'] if '$2'=='T' else json.dumps({k:r[k] for k in ('temperature','calib_calibrated','test_calibrated')},ensure_ascii=False)[:2500])"; }
TN=$(rep sev__gen T); TO=$(rep sev__core T)
log "SEVGEN_TRAINED $(rep sev__gen all)"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-sevgen.log sevnew=$FS/adapters/sev__gen-mc/adapter sev=$FS/adapters/sev__core-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.sev_filter_score --model sevnew --prompt new --temp $TN --files items_pair.jsonl items_sevnew.jsonl --out $G/sev2_scores.jsonl >> $LOG 2>&1 &
python -u -m ragtruth.sev_filter_score --model sev --prompt old --temp $TO --files items_sevnew.jsonl --out $G/sev_scores_sevnew.jsonl >> $LOG 2>&1 &
wait $(jobs -p | tail -2) 2>/dev/null; sleep 2
grep -a SEV_SCORE $LOG | tail -2
free_gpu
log "SEVGEN_DONE"
