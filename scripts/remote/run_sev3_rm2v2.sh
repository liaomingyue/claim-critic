#!/bin/bash
# 训新版 sev（sev__v3：输入带要求的机制 + 9B 描述 + 逐词改动）和新版 RM2（rm2__v2：真实错误 vs 经 sev / critic 过滤的合成错误），再在新样本集上重跑。
#   ① 9B 基座给 1200 条有 Claude 标注的生成器 v2 样本写改动描述（不给机制）→ labitems_desc9b.jsonl
#   ② sev__v3 数据（ragtruth.sev_gen data_v3）→ run_mcrm；③ rm2__v2（本地建好上传）→ run_mcrm
#   ④ 1200 条样本用 rm1 / rm2 / rm2v2 / dne / v3 打分 → gen_v2_label/model_scores_v3.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sev3_rm2v2.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
if [ ! -s $C/labitems_desc9b.jsonl ]; then
  free_gpu; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-desc.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2 describe --model base --samples labitems.jsonl --no_mech --out labitems_desc9b.jsonl >> $LOG 2>&1 || { log "describe FAILED"; exit 1; }
  free_gpu
fi
log "DESC $(grep -a DESC9B $LOG | tail -1)"
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
python -u -m ragtruth.sev_gen data_v3 --desc labitems_desc9b.jsonl >> $LOG 2>&1 || { log "data_v3 FAILED"; exit 1; }
grep -a SEV_V3_DATA $LOG | tail -1 | cut -c1-400
for RM in sev__v3 rm2__v2; do
  if [ ! -f outputs/rt/rm/$RM/eval/report.json ]; then
    free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  fi
  [ -f outputs/rt/rm/$RM/eval/report.json ] || { log "$RM FAILED"; continue; }
  log "TRAINED $RM $(python -c "import json;r=json.load(open('outputs/rt/rm/$RM/eval/report.json'));print({'T':r['temperature'],**{k:{m:r[k].get(m) for m in ('auc','option_acc')} for k in ('calib_calibrated','test_calibrated')}})")"
done
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
TEMPS="rm1=$(t rm1__core) rm2=$(t rm2__core) rm2v2=$(t rm2__v2) dne=$(t sev__gen_dne) v3=$(t sev__v3)"
log "temps $TEMPS"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-score3.log rm1=$FS/adapters/rm1__core-mc/adapter rm2=$FS/adapters/rm2__core-mc/adapter rm2v2=$FS/adapters/rm2__v2-mc/adapter dne=$FS/adapters/sev__gen_dne-mc/adapter v3=$FS/adapters/sev__v3-mc/adapter >> $LOG 2>&1 || { log "serve score FAILED"; exit 1; }
python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --desc labitems_desc9b.jsonl --temps $TEMPS --out gen_v2_label/model_scores_v3.jsonl >> $LOG 2>&1 || log "score FAILED"
free_gpu
log "SEV3_RM2V2_DONE"
