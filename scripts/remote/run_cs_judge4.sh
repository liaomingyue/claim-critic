#!/bin/bash
# 客服 Agent：失败轨迹用第四版问法重判（cs_agent.judge2 --prompt v4：根因步 + 出事步 + 引文定位），写 <策略>_judge4.jsonl，可续跑。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; NAME=${NAME:-sample_r0}; O=outputs/rt/cs/$NAME; LOG=$A/cs_judge${JV:-4}.log; PS_=${JPOLS:-"base9 sftB"}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_JUDGE4 FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_JUDGE4 FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
A=$A/s1 MAX_LORAS=1 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $B9 --log $A/vllm-judge4.log \
  unused=$FS/adapters/gen-ops-zero-9b/adapter >> $LOG 2>&1 || { log "CS_JUDGE4 FAILED: vllm"; exit 1; }
for p in $PS_; do
  L=$O/${p}_labels.jsonl; [ -s $L ] || L=$O/${p}_labels_live.jsonl
  $PY -u -m cs_agent.judge2 --prompt ${JPROMPT:-v4} --eps $O/${p}_train.jsonl --labels $L --split train --url http://127.0.0.1:8001/v1 \
    --tok $B9 --concurrency ${CONC:-12} --out $O/${p}_judge${JV:-4}.jsonl >> $A/cs_judge${JV:-4}_$p.log 2>&1 || log "judge4 $p FAILED"
  log "JUDGE4 $p $(grep -a JUDGE2_DONE $A/cs_judge${JV:-4}_$p.log | tail -1 | cut -c1-250)"
done
pkill -f '[v]llm serve'
cp $LOG $A/cs_judge${JV:-4}_*.log $O/ 2>/dev/null
log "CS_JUDGE4_DONE"
