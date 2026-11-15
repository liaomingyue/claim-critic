#!/bin/bash
# 多维度事后分类模型 opd__v1（ragtruth/rm_opd.py，数据本地建好随代码上传）：一次前向给出层、操作、动作、手段。
# 训完：留出测试集上各维度的准确率 → 给几版生成器的留出样本分类 → 算起点模型各操作的节点占比。
# 节点占比是生成器下一轮训练（run_genops_st5.sh）里多样性项的基准。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/rmopd.log; T=scripts/analysis
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
RM=opd__v1 GPU=0 PORT=8100 MC=$MC CODE=$PWD bash $S/run_mcrm.sh >> $LOG 2>&1
[ -f $R/opd__v1/eval/report.json ] || { log "opd__v1 FAILED"; exit 1; }
log "TRAINED opd__v1"
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opd.log opd=$FS/adapters/opd__v1-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u $T/genops_classify_opd.py eval >> $LOG 2>&1 || log "eval FAILED"
python -u $T/genops_classify_opd.py classify ${TAGS:-ev_base_free ev_s1c ev_st3 ev_st4} >> $LOG 2>&1 || log "classify FAILED"
python -u $T/genops_classify_opd.py base ev_base_free >> $LOG 2>&1 || log "base FAILED"
pkill -f '[v]llm serve'; sleep 8
mkdir -p $FS/results/opd && cp $R/opd__v1/eval/report_multi.json configs/rt/op_node_base_opd.json $FS/results/opd/ 2>> $LOG
tar -czf $FS/results/opd/opd_out.tgz $C/samples_ops_ev_*_opd.jsonl 2>> $LOG
log "RMOPD_DONE"
