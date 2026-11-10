#!/bin/bash
# 训 sev 和 RM2。第一轮在 critic 盲区里的 Claude 标注（hard_check 100 + blind_fill 787 = 887 条，outputs/rt/core/sevaug_r1）只进训练集：
#   sev__v3d = sev__v3c0 的配方 + sevaug_r1；rm2__v3d = rm2__v3b 的配方 + sevaug_r1。校准 / 测试切分不变，和 v3c0 / v3b 直接比。
#   盲区里的效果，等第二轮 Claude 标注（blind_fill_r2，没进训练）出来再在上面比。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/sevrm2_r1.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
DESC=labitems_desc9b.jsonl,labitems_t1_desc9b.jsonl,labitems_t2_desc9b.jsonl
python -u -m ragtruth.sev_gen data_v3 --name sev__v3d --desc $DESC --extra_train gen_v2_label_t1,gen_v2_label_t2,sevaug_r1 >> $LOG 2>&1 || { log "sev data FAILED"; exit 1; }
log "$(grep -a SEV_V3_DATA $LOG | tail -1 | cut -c1-400)"
python -u -m ragtruth.rm2_v3 --name rm2__v3d --extra_train gen_v2_label_t1,sevaug_r1 >> $LOG 2>&1 || { log "rm2 data FAILED"; exit 1; }
log "$(grep -a '^RM2_V3' $LOG | tail -1 | cut -c1-400)"
for RM in sev__v3d rm2__v3d; do
  [ -f outputs/rt/rm/$RM/eval/report.json ] || { free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu; }
  [ -f outputs/rt/rm/$RM/eval/report.json ] && log "TRAINED $RM $(python -c "import json;r=json.load(open('outputs/rt/rm/$RM/eval/report.json'));print(r['temperature'], {k:r[k].get('auc') for k in ('calib','test') if isinstance(r.get(k),dict)})")" || log "$RM FAILED"
done
log "SEVRM2_R1_DONE"
