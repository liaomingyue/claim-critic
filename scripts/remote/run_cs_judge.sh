#!/bin/bash
# 客服 Agent：带特权信息的 9B 逐步判断，和采样（run_cs_sample.sh NO_JUDGE=1）同时在另一台机器上跑。
#   一个 9B 服务（基座，开思考）；循环：对每个策略已经采到的轨迹先做规则 / 标准动作标注（写 <策略>_labels_live.jsonl，不和采样机最后写的文件冲突），
#   再补判还没判过的（cs_agent.judge 续跑）；看到采样机写的 SAMPLE_DONE 后做最后一轮，然后退出。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; NAME=${NAME:-sample_r0}; O=outputs/rt/cs/$NAME; LOG=$A/cs_judge.log; PS_=${JPOLS:-"base9 sftB"}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_JUDGE FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_JUDGE FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
A=$A/s1 MAX_LORAS=1 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $B9 --log $A/vllm-judge.log \
  unused=$FS/adapters/gen-ops-zero-9b/adapter >> $LOG 2>&1 || { log "CS_JUDGE FAILED: vllm"; exit 1; }
last=0
while true; do
  [ -f $O/SAMPLE_DONE ] && last=1
  for p in $PS_; do
    [ -s $O/${p}_train.jsonl ] || continue
    $PY -m cs_agent.label --eps $O/${p}_train.jsonl --split train --out $O/${p}_labels_live.jsonl > /dev/null 2>> $LOG || { log "label $p FAILED"; continue; }
    $PY -u -m cs_agent.judge --eps $O/${p}_train.jsonl --labels $O/${p}_labels_live.jsonl --split train --url http://127.0.0.1:8001/v1 \
      --select neg+${JUDGE_POS:-0.15} --concurrency ${CONC:-32} --out $O/${p}_judge.jsonl >> $A/cs_judge_$p.log 2>&1 || log "judge $p FAILED"
    log "JUDGE $p $(grep -a JUDGE_DONE $A/cs_judge_$p.log | tail -1 | cut -c1-250)"
  done
  [ $last = 1 ] && break
  sleep 120
done
pkill -f '[v]llm serve'
cp $LOG $A/cs_judge_*.log $O/ 2>/dev/null
log "CS_JUDGE_DONE"
