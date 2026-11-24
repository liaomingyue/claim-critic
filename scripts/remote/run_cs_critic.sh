#!/bin/bash
# 客服 Agent：训练奖励模型（cs_agent.critic），TASK=step（过程 critic，5 类）| outcome（全流程 reward，3 类），训完在 test / claude 两个评测集上打分。
#   adapter + head → $FS/adapters/cs-critic-<TASK>-9b；分数和指标 → outputs/rt/cs/critic_<TASK>/
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; TASK=${TASK:?step or outcome}; NAME=${NAME:-sample_r0}
D=outputs/rt/cs/$NAME; O=outputs/rt/cs/critic_$TASK; AD=$FS/adapters/cs-critic-$TASK-9b; LOG=$A/cs_critic_$TASK.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $S/setup_tau2.sh || { log "CS_CRITIC FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_CRITIC FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
pkill -f '[v]llm serve'; sleep 10
mkdir -p $O
EPS="$D/base9_train.jsonl $D/sftB_train.jsonl"
if [ ! -f $AD/head.pt ]; then
  $PY -u -m cs_agent.critic train --task $TASK --base $B9 --data $D/steps_cls_v1.jsonl --eps $EPS --out $A/ckpt/cs-critic-$TASK \
    --epochs ${EPOCHS:-2} --max_len ${MAX_LEN:-16384} >> $LOG 2>&1 || { log "CS_CRITIC FAILED: train"; exit 1; }
  mkdir -p $AD && rsync -a $A/ckpt/cs-critic-$TASK/ $AD/ || { log "CS_CRITIC FAILED: save"; exit 1; }
fi
log "trained: $(grep -a '^CRITIC \|CRITIC_DONE' $LOG | tail -2 | cut -c1-400 | tr '\n' ' ')"
$PY -u -m cs_agent.critic eval --task $TASK --base $B9 --ckpt $AD --data $D/steps_cls_v1.jsonl --eps $EPS --out $O/scores.jsonl >> $LOG 2>&1 || log "eval FAILED"
grep -a CRITIC_EVAL $LOG | tail -1 > $O/metrics.txt
log "EVAL $(cut -c1-600 $O/metrics.txt)"
cp $LOG $O/ 2>/dev/null
log "CS_CRITIC_DONE $TASK"
