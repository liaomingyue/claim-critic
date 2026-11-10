#!/bin/bash
# sev 过滤之外再加一道「critic 判不出」的筛选（r2 的做法，核实换成 sev__gen）。
#   ① 9B 错误改写模型在剩下 631 个提示（genpool_rest.jsonl）上每个采 8 条 → samples_9b_rest.jsonl（扩大候选池）
#   ② sev__gen 给全部 9B 样本 + 审计样本打分 → gen_audit/sev2_scores_all.jsonl
#   ③ 基线 critic rm1__core 给同一批打 P(判出) → gen_audit/critic_core_scores.jsonl（本地先量 sev 在 critic 盲区里准不准）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/blindpool.log; C=outputs/rt/core; G=$C/gen_audit
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
until grep -q SEVVAR_DONE $A/sevvar.log 2>/dev/null; do sleep 30; done
log "start (after sevvar)"
if [ ! -s $C/samples_9b_rest.jsonl ]; then
  source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-gen9b.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve gen FAILED"; exit 1; }
  python -u -m ragtruth.core sample --model gen --tokenizer $B9 --pool genpool_rest.jsonl --out samples_9b_rest.jsonl --n 8 >> $LOG 2>&1 || { log "sample FAILED"; exit 1; }
  free_gpu
fi
grep -a CORE_SAMPLE $LOG | tail -1 | cut -c1-400
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
TS=$(python -c "import json;print(json.load(open('outputs/rt/rm/sev__gen/eval/report.json'))['temperature'])")
TC=$(python -c "import json;print(json.load(open('outputs/rt/rm/rm1__core/eval/report.json'))['temperature'])")
log "T sev__gen=$TS rm1__core=$TC"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bp.log sev=$FS/adapters/sev__gen-mc/adapter critic=$FS/adapters/rm1__core-mc/adapter >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
python -u -m ragtruth.sev_filter_score --model sev --prompt new --temp $TS --files items_pair.jsonl items_sevnew.jsonl --extra samples_9b_rest.jsonl --out $G/sev2_scores_all.jsonl >> $LOG 2>&1 & P1=$!
python -u -m ragtruth.critic_on_samples --model critic --temp $TC --extra samples_9b_rest.jsonl --out $G/critic_core_scores.jsonl >> $LOG 2>&1 & P2=$!
wait $P1; wait $P2
grep -aE 'SEV_SCORE|CRITIC_ON_SAMPLES' $LOG | tail -2
free_gpu
log "BLINDPOOL_DONE"
