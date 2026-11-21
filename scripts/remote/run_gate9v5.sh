#!/bin/bash
# 第二轮：三个 9B 把关模型加 Claude 标签重训（一台一个，三台并行）。
#   数据：rm_ops 原配方 + 第一轮 9B 生成器各阶段 Claude 抽查（留出原文上 6 组）+ 训练原文上的边界样本 Claude 审核；新加的组在训练集里重复 2 次。
#   按原文哈希切分（和原配方同一个盐），新标签落在测试集里的那部分用来和旧把关模型比。
#   WHICH=sev|nat|opm；sev 那台先建数据（三个模型的数据一次建好），写 data_ready，另外两台等它。
cd ${CODE:-/root/autodl-fs/code/critic_model}
export PATH=/root/miniconda3/bin:$PATH PYTHONHASHSEED=0 PYTHONPATH=. PYTHONIOENCODING=utf-8
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/gate9v5.log; F=$FS/runs/g9v5; mkdir -p $F
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
WHICH=${WHICH:?}
case $WHICH in sev) SRC=sev__v5 ;; nat) SRC=rm2__v5 ;; opm) SRC=opm__v2 ;; *) log "bad WHICH"; exit 1 ;; esac
if [ $WHICH = sev ]; then
  rm -f $F/data_ready $F/data_failed
  EX="ops_verify_ev_s1c9b=samples_ops_ev_s1c9b.jsonl ops_verify_ev_s1c9b_zone=samples_ops_ev_s1c9b.jsonl ops_verify_ev_s2c9b=samples_ops_ev_s2c9b.jsonl"
  EX="$EX ops_verify_ev_s2c9b_zone=samples_ops_ev_s2c9b.jsonl ops_verify_ev_s3c9_zone=samples_ops_ev_s3c9.jsonl ops_verify_ev_s4c9_zone=samples_ops_ev_s4c9.jsonl"
  EX="$EX ops_verify_tr9_s4c9_bd=samples_ops_tr9_s4c9.jsonl"
  python -m ragtruth.rm_ops --extra $EX --dup 2 --names sev__v5,rm2__v5,opm__v2 >> $LOG 2>&1 && touch $F/data_ready || { touch $F/data_failed; log "GATE9V5 FAILED data"; exit 1; }
  grep -a "SEV_V4\|RM2_V4\|OPM_V1" $LOG | tail -3 | cut -c1-300
else
  until [ -f $F/data_ready ]; do [ -f $F/data_failed ] && { log "GATE9V5 FAILED data (other node)"; exit 1; }; sleep 30; done
fi
SETS=none SRC=$SRC bash scripts/remote/run_crit9b.sh || { log "GATE9V5 FAILED train $SRC"; exit 1; }
log "GATE9V5_DONE $SRC"
