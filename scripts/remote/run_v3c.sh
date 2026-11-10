#!/bin/bash
# 只补范围偏移（训练池上已采未标的 300 条，Claude 标 294）+ 直接矛盾放宽为「真错即可」。
# sev__v3c = v3b 数据 + t2 范围偏移 + 矛盾放宽；sev__v3c0 = 只加 t2（不放宽），用来拆开两个改动各自的作用。校准 / 测试切分不变。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/v3c.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
# 先拷 2B（前台），9B 后台拷；用 9B 前等它拷完
source $S/model_env.sh minicpm5-2b && log "2B ready $MODEL_LOCAL"
( source $S/model_env.sh qwen3.5-9b && touch $A/9b_copied ) >> $LOG 2>&1 &
CP9=$!
if [ ! -s $C/labitems_t2_desc9b.jsonl ]; then
  wait $CP9; log "9B copied"
  free_gpu; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-desc.log gen=$FS/adapters/gen-core-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2 describe --model base --samples labitems_t2.jsonl --no_mech --out labitems_t2_desc9b.jsonl >> $LOG 2>&1 || { log "describe FAILED"; exit 1; }
  free_gpu
fi
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
DESC=labitems_desc9b.jsonl,labitems_t1_desc9b.jsonl,labitems_t2_desc9b.jsonl
python -u -m ragtruth.sev_gen data_v3 --name sev__v3c --desc $DESC --extra_train gen_v2_label_t1,gen_v2_label_t2 --relax contradicts_source >> $LOG 2>&1 || { log "data v3c FAILED"; exit 1; }
python -u -m ragtruth.sev_gen data_v3 --name sev__v3c0 --desc $DESC --extra_train gen_v2_label_t1,gen_v2_label_t2 >> $LOG 2>&1 || { log "data v3c0 FAILED"; exit 1; }
log "DATA $(grep -a SEV_V3_DATA $LOG | tail -2 | cut -c1-300 | tr '\n' ' ')"
for RM in sev__v3c sev__v3c0; do
  [ -f outputs/rt/rm/$RM/eval/report.json ] || { free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu; }
  [ -f outputs/rt/rm/$RM/eval/report.json ] && log "TRAINED $RM" || log "$RM FAILED"
done
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
for RM in sev__v3c sev__v3c0; do
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-v3c.log v3=$FS/adapters/$RM-mc/adapter >> $LOG 2>&1 || { log "serve $RM FAILED"; continue; }
  python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --desc labitems_desc9b.jsonl --temps v3=$(t $RM) --out gen_v2_label/model_scores_${RM#sev__}.jsonl >> $LOG 2>&1 || log "score $RM FAILED"
done
free_gpu
log "V3C_DONE"
