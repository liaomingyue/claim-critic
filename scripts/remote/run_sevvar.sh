#!/bin/bash
# sev 输入里把改动明确告诉模型再判，应该容易很多。两个变体（数据 ragtruth.sev_gen data --variant，随代码上传）：
#   sev__gen_diff：现有输入 + 逐词 diff 改动标记（5 选 1）
#   sev__gen_dne ：不给生成器解释，只给原句 / 改写句 / 改动标记，判严重程度 3 选 1（A 显然 / B 细微 / C 没错）
# 训练同 sev__gen（run_mcrm.sh），再给同一批审计样本逐条打分 → gen_audit/sev_{diff,dne}_scores.jsonl（本地 outputs/sev2_an.py 比）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sevvar.log; G=outputs/rt/core/gen_audit
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
for v in diff dne; do
  RM=sev__gen_$v
  if [ ! -f outputs/rt/rm/$RM/eval/report.json ]; then
    free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
    [ -f outputs/rt/rm/$RM/eval/report.json ] || { log "$RM train FAILED"; continue; }
  fi
  T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$RM/eval/report.json'))['temperature'])")
  log "TRAINED $RM T=$T $(python -c "import json;r=json.load(open('outputs/rt/rm/$RM/eval/report.json'));print({k:{m:r[k].get(m) for m in ('auc','recall@0.5','precision@0.5','option_acc')} for k in ('calib_calibrated','test_calibrated')})")"
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-$v.log s=$FS/adapters/$RM-mc/adapter >> $LOG 2>&1 || { log "$RM serve FAILED"; continue; }
  python -u -m ragtruth.sev_filter_score --model s --prompt $v --temp $T --files items_pair.jsonl items_sevnew.jsonl --out $G/sev_${v}_scores.jsonl >> $LOG 2>&1 || log "$RM score FAILED"
  free_gpu
  log "SCORED $RM"
done
log "SEVVAR_DONE"
