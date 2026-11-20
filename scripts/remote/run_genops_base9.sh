#!/bin/bash
# 9B 错误合成的基线打分，直接用推理机（run_genops_infer9.sh）已经起好的 vLLM，不用等训练机：
#   9B 基座（policy_v0 = B 清零的 LoRA）在留出回答 × 20 种操作上各采 4 条，9B 把关模型打分；2B 各版生成器的留出样本也用 9B 把关模型重打（后缀 _rm9）。
# 结果在共享盘上，训练机的 run_genops_grpo9.sh 看到文件已有就跳过这一步。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis; P=$FS/runs/pair/${PAIR:-gog9}; LOG=$A/genops_base9.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8
until [ -f $P/infer_ready ]; do [ -f $P/infer_failed ] && { log "BASE9 FAILED: infer node"; exit 1; }; sleep 20; done
export VLLM_API_KEY=$(cat $P/key) OPENAI_API_KEY=x GENOPS_URL=http://127.0.0.1:6006/v1
t() { python -c "import json;print(json.load(open('$R/${1}_9b/eval/report.json'))['temperature'])"; }
TJ="{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d2)}"
log "temperatures $TJ"
[ -s $C/samples_ops_ev_base9.jsonl ] || python -u -m ragtruth.gen_ops sample --model policy_v0 --tokenizer $A/models/Qwen3.5-9B --card --style free --n 4 --url $GENOPS_URL --out samples_ops_ev_base9.jsonl >> $LOG 2>&1 || { log "BASE9 FAILED: sample"; exit 1; }
log "sampled $(grep -a GENOPS_SAMPLE $LOG | tail -1 | cut -c1-300)"
for x in ev_base9 ${RESCORE-ev_base_free ev_s1c ev_st4 ev_st8}; do
  [ -s $C/samples_ops_$x.jsonl ] || { [ -s $C/samples_ops_$x.jsonl.gz ] && gunzip -k $C/samples_ops_$x.jsonl.gz; }
  [ -s $C/samples_ops_$x.jsonl ] || { log "missing $x"; continue; }
  [ -s $C/samples_ops_${x}_rm9.jsonl ] || GENOPS_SUF=rm9 python -u $T/genops_score.py "$TJ" $x >> $LOG 2>&1 || log "score $x FAILED"
  log "scored $x $(wc -l < $C/samples_ops_${x}_rm9.jsonl 2>/dev/null)"
done
log "BASE9_DONE"
