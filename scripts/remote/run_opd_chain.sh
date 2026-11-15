#!/bin/bash
# 接在两步分类模型（run_rmopc.sh）后面：等它结束 → 训多维度分类模型（run_rmopd.sh）→ 准确率够就接着训生成器 st5（run_genops_st5.sh，CLS=2）→ 关机。
# 准确率门槛：留出测试集的真错样本上，操作判对 ≥ MIN_OP、节点（动作 / 手段）判对 ≥ MIN_NODE。不够就不训生成器：分类不准时按它扣分只会引入噪声。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; S=scripts/remote; LOG=$A/opd_chain.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -qE 'RMOPC_DONE|FAILED' $A/rmopc.log 2>/dev/null && grep -q 'WORKER_DONE' $A/worker.log; do sleep 30; done
sleep 20; pkill -f 'sleep 120[0]'   # 上一个角色结束时挂的关机定时：这条链自己负责关机
log "rmopc finished, shutdown timer cancelled: $(grep -E 'TRAINED|FAILED' $A/rmopc.log | tr '\n' ';' | cut -c1-300)"
CODE=$PWD bash $S/run_rmopd.sh
R=outputs/rt/rm/opd__v1/eval/report_multi.json
if [ -f $R ] && python -c "import json,sys;r=json.load(open('$R'));sys.exit(0 if r['op_on_errors']>=${MIN_OP:-0.5} and r['node_on_errors']>=${MIN_NODE:-0.5} else 1)"; then
  log "opd accurate enough, training st5"
  CODE=$PWD bash $S/run_genops_st5.sh
else
  log "opd below the bar or missing, st5 not started"
fi
log "OPD_CHAIN_DONE"
sleep ${GRACE:-1500}; /usr/bin/shutdown
