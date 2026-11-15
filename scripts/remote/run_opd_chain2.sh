#!/bin/bash
# 多维度分类模型已经训好、适配器已存（adapters/opd__v1-mc），接着做：留出测试集上各维度的准确率 → 给几版生成器的留出样本分类 →
# 起点模型的节点占比 → 准确率够就训生成器 st5（CLS=2）→ 关机。
# run_mcrm.sh 自带的单字母评测不适用于四个字母的标签（训练本身不受影响），这里用 genops_classify_opd.py 按位置评测。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/opd_chain2.log; T=scripts/analysis
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
[ -f $FS/adapters/opd__v1-mc/adapter/adapter_model.safetensors ] || { log "opd adapter missing"; sleep 600; /usr/bin/shutdown; exit 1; }
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opd.log opd=$FS/adapters/opd__v1-mc/adapter >> $LOG 2>&1 || log "serve FAILED"
python -u $T/genops_classify_opd.py eval >> $LOG 2>&1 || log "eval FAILED"
python -u $T/genops_classify_opd.py classify ${TAGS:-ev_base_free ev_s1c ev_st3 ev_st4} >> $LOG 2>&1 || log "classify FAILED"
python -u $T/genops_classify_opd.py base ev_base_free >> $LOG 2>&1 || log "base FAILED"
pkill -f '[v]llm serve'; sleep 8
mkdir -p $FS/results/opd && cp $R/opd__v1/eval/report_multi.json configs/rt/op_node_base_opd.json $FS/results/opd/ 2>> $LOG
tar -czf $FS/results/opd/opd_out.tgz $C/samples_ops_ev_*_opd.jsonl 2>> $LOG
log "RMOPD_DONE"
RP=$R/opd__v1/eval/report_multi.json
if [ -f $RP ] && [ -f configs/rt/op_node_base_opd.json ] && python -c "import json,sys;r=json.load(open('$RP'));sys.exit(0 if r['op_on_errors']>=${MIN_OP:-0.5} and r['node_on_errors']>=${MIN_NODE:-0.5} else 1)"; then
  log "opd accurate enough, training st5"
  CODE=$PWD bash $S/run_genops_st5.sh
else
  log "opd below the bar or missing, st5 not started"
fi
log "OPD_CHAIN2_DONE"
sleep ${GRACE:-1500}; /usr/bin/shutdown
