#!/bin/bash
# 客服 Agent：9B 多轮 SFT（APIGen-MT retail，A = 只按结果筛 / B = 再按规则筛），训完在 τ² retail 测试集上评测（同基线设置：40 任务 × 4 次，策略温度 0.6）。
#   V=A|B bash run_cs_sft.sh   单卡 5090：先训（独占卡），训完起一个 9B 服务同时当策略（SFT 的 LoRA）和用户模拟器（基座）。
#   adapter → $FS/adapters/cs-sft-<V>-9b（epoch1 / 最终）；评测 → outputs/rt/cs/sft<V>_q9b/test.jsonl + 过程违规（cs_agent.rules）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; V=${V:?A or B}; K=${K:-4}; LOG=$A/cs_sft_$V.log
O=outputs/rt/cs/sft${V}_q9b; D=outputs/rt/cs/sft_data; AD=$FS/adapters/cs-sft-$V-9b
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_SFT FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_SFT FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
# 数据（两台机器同时跑时只建一次）
( flock 7; [ -s $D/sft_$V.jsonl ] || $PY -u -m cs_agent.sft_data build --src $FS/external/agent/apigen-mt-5k/apigen-mt_5k.json --out $D --tok $B9 >> $LOG 2>&1 ) 7>$FS/env/.cs_sft_data.lock
[ -s $D/sft_$V.jsonl ] || { log "CS_SFT FAILED: data"; exit 1; }
log "data $(cat $D/stats.json | tr -d '\n ' | cut -c1-300)"
if [ ! -f $AD/adapter_model.safetensors ]; then
  $PY -u -m cs_agent.sft --base $B9 --data $D/sft_$V.jsonl --out $A/ckpt/cs-sft-$V --epochs ${EPOCHS:-1} --lr ${LR:-1e-4} >> $LOG 2>&1 || { log "CS_SFT FAILED: train"; exit 1; }
  mkdir -p $AD && rsync -a $A/ckpt/cs-sft-$V/ $AD/ || { log "CS_SFT FAILED: save adapter"; exit 1; }
fi
log "trained: $(grep -a 'SFT_DONE\|^SFT ' $LOG | tail -2 | tr '\n' ' ')"
mkdir -p $O
A=$A/s1 MAX_LORAS=2 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $B9 --log $A/vllm-sft$V.log policy_v0=$AD >> $LOG 2>&1 || { log "CS_SFT FAILED: vllm"; exit 1; }
$PY -u -m cs_agent.evaluate --policy_url http://127.0.0.1:8001/v1 --policy policy_v0 --tok $B9 --user_url http://127.0.0.1:8001/v1 \
  --split test --k $K --concurrency ${CONC:-32} --out $O/test.jsonl >> $LOG 2>&1 || log "eval FAILED"
$PY -c "import json; from cs_agent.rules import summarize; eps=[json.loads(l) for l in open('$O/test.jsonl')]; s=summarize(eps); open('$O/test_rules.json','w').write(json.dumps(s, indent=1)); print('RULES', json.dumps(s))" >> $LOG 2>&1
log "EVAL $(grep -a EVAL_DONE $LOG | tail -1 | cut -c1-300)"
log "RULES $(grep -a '^RULES' $LOG | tail -1 | cut -c1-300)"
pkill -f '[v]llm serve'
cp $LOG $O/ 2>/dev/null
log "CS_SFT_DONE $V"
