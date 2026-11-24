#!/bin/bash
# 客服 Agent：成功轨迹全部用第六版问法判（含 OVERFIX，写 <策略>_judge6.jsonl），失败轨迹按第三版定好的第一个错误补问 OVERFIX（<策略>_judgeofx.jsonl），
# 然后重建步骤级标签数据集（cs_agent.step_dataset → steps_v1.jsonl）。都可续跑；不关机。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; NAME=${NAME:-sample_r0}; O=outputs/rt/cs/$NAME; LOG=$A/cs_judge6.log; PS_=${JPOLS:-"base9 sftB"}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_JUDGE6 FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_JUDGE6 FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
if ! curl -s -m 5 http://127.0.0.1:8001/v1/models > /dev/null; then
  A=$A/s1 MAX_LORAS=1 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $B9 --log $A/vllm-judge6.log \
    unused=$FS/adapters/gen-ops-zero-9b/adapter >> $LOG 2>&1 || { log "CS_JUDGE6 FAILED: vllm"; exit 1; }
fi
for p in $PS_; do
  L=$O/${p}_labels.jsonl; [ -s $L ] || L=$O/${p}_labels_live.jsonl
  $PY -u -m cs_agent.judge2 --prompt v6 --eps $O/${p}_train.jsonl --labels $L --split train --url http://127.0.0.1:8001/v1 \
    --tok $B9 --concurrency ${CONC:-12} --out $O/${p}_judge6.jsonl >> $A/cs_judge6_$p.log 2>&1 || log "judge6 $p FAILED"
  log "JUDGE6 $p $(grep -a JUDGE2_DONE $A/cs_judge6_$p.log | tail -1 | cut -c1-300)"
  $PY -u -m cs_agent.judge2 --prompt ofx --ref $O/${p}_judge2.jsonl --eps $O/${p}_train.jsonl --labels $L --split train --url http://127.0.0.1:8001/v1 \
    --tok $B9 --concurrency ${CONC:-12} --out $O/${p}_judgeofx.jsonl >> $A/cs_judgeofx_$p.log 2>&1 || log "judgeofx $p FAILED"
  log "JUDGEOFX $p $(grep -a JUDGE2_DONE $A/cs_judgeofx_$p.log | tail -1 | cut -c1-300)"
done
$PY -m cs_agent.step_dataset --dir $O --out $O/steps_v1.jsonl > $A/steps_v1.txt 2>&1 || log "step_dataset FAILED"
log "STEPS $(tr -d '\n ' < $A/steps_v1.txt | cut -c1-600)"
cp $LOG $A/cs_judge6_*.log $A/cs_judgeofx_*.log $O/ 2>/dev/null
log "CS_JUDGE6_DONE"
