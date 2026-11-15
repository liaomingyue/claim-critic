#!/bin/bash
# 事后分类模型（ragtruth/rm_opc.py，数据本地建好随代码上传）：opl__v1 判错在哪一层，opc__v1 判层内是哪一种操作。
# 训完用它们给几版生成器在留出回答上的样本分类，看各版产出落在哪些类别上。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/rmopc.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
for rm in opl__v1 opc__v1; do
  RM=$rm GPU=0 PORT=8100 MC=$MC CODE=$PWD bash $S/run_mcrm.sh >> $LOG 2>&1
  [ -f $R/$rm/eval/report.json ] || { log "$rm FAILED"; exit 1; }
  log "TRAINED $rm $(python -c "import json;r=json.load(open('$R/$rm/eval/report.json'));print(r['temperature'], r['calib_calibrated']['option_acc'], r['test_calibrated']['option_acc'])")"
done
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opc.log opl=$FS/adapters/opl__v1-mc/adapter opc=$FS/adapters/opc__v1-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
python -u scripts/analysis/genops_classify.py "{\"opl\": $(t opl__v1), \"opc\": $(t opc__v1)}" ${TAGS:-ev_base_free ev_s1c ev_st3 ev_st4} >> $LOG 2>&1 || log "classify FAILED"
pkill -f '[v]llm serve'
mkdir -p $FS/results/opc && cp $R/opl__v1/eval/report.json $FS/results/opc/opl_report.json && cp $R/opc__v1/eval/report.json $FS/results/opc/opc_report.json
tar -czf $FS/results/opc/opc_out.tgz $C/samples_ops_ev_*_opc.jsonl $R/opl__v1/eval/test_scores.jsonl $R/opc__v1/eval/test_scores.jsonl 2>> $LOG
log "RMOPC_DONE"
