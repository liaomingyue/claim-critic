#!/bin/bash
# 扩充 sev / RM2 训练数据（训练池上 2B + 9B 采样 1444 条 Claude 标注，只进训练集）→ sev__v3b、rm2__v3b，和 v3 在同一测试切分上比。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/v3b.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
if [ ! -s $C/labitems_t1_desc9b.jsonl ]; then
  free_gpu; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-desc.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2 describe --model base --samples labitems_t1.jsonl --no_mech --out labitems_t1_desc9b.jsonl >> $LOG 2>&1 || { log "describe FAILED"; exit 1; }
  free_gpu
fi
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
python -u -m ragtruth.sev_gen data_v3 --name sev__v3b --desc labitems_desc9b.jsonl,labitems_t1_desc9b.jsonl --extra_train gen_v2_label_t1 >> $LOG 2>&1 || { log "data FAILED"; exit 1; }
log "DATA $(grep -a SEV_V3_DATA $LOG | tail -1 | cut -c1-300)"
for RM in sev__v3b rm2__v3b; do
  [ -f outputs/rt/rm/$RM/eval/report.json ] || { free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu; }
  [ -f outputs/rt/rm/$RM/eval/report.json ] && log "TRAINED $RM" || log "$RM FAILED"
done
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-v3b.log v3=$FS/adapters/sev__v3b-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --desc labitems_desc9b.jsonl --temps v3=$(t sev__v3b) rm2v3=$(t rm2__v3b) --out gen_v2_label/model_scores_v3b.jsonl >> $LOG 2>&1 || log "score FAILED"
free_gpu
log "V3B_DONE"
