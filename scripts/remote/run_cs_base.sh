#!/bin/bash
# 客服 Agent 第 1 步：Qwen3.5-2B 基座在 τ²-bench retail 上的基线（单卡 5090，不训练、不调 Claude）。
#   一张卡上两个 vLLM：9B 用户模拟器（:8001，基座直接用，τ² 官方的用户模拟提示词）+ 2B 策略（:8000，B 全零的起点 LoRA = 基座）；
#   测试集 40 个任务、训练集 74 个任务各跑 K 次（默认 4）：测试集定下限，训练集的逐任务通过率之后用来挑 GRPO 的任务（0.2–0.7 区间）。
#   结果 outputs/rt/cs/<名>/{test,train}.jsonl 和 _summary.json（直接写在共享盘的代码目录里）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; NAME=${NAME:-base_q2b}; K=${K:-4}; O=outputs/rt/cs/$NAME; LOG=$A/cs_base.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_BASE FAILED: setup_tau2"; exit 1; }
log "tau2 python $PY"
source $S/models9b.sh; start_9b_bg
Q2=$A/models/Qwen3.5-2B
[ -f $Q2/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/Qwen3.5-2B/ $A/models/.Qwen3.5-2B.partial/ && mv $A/models/.Qwen3.5-2B.partial $Q2; } || { log "CS_BASE FAILED: copy 2B"; exit 1; }
Z=$FS/adapters/cs-q2b-zero/adapter
[ -f $Z/adapter_model.safetensors ] || python scripts/make_lora_init.py $Q2 $Z >> $LOG 2>&1 || { log "CS_BASE FAILED: zero lora"; exit 1; }
ensure_9b || { log "CS_BASE FAILED: 9b copy"; exit 1; }
# 同卡两个服务：9B 0.66 时 16k 上下文的 KV 不够（要 0.61 GB 只剩 0.45 GB，实测），改 0.72 + 每批 8192 token（激活峰值跟着降）；2B 0.22。
# 显存小时线性注意力层的状态缓存块也少（9B 0.72 只有 144 块），默认 max_num_seqs 256 起不来；对话并发 32，设 64 够用。
# 9B 用户模拟器：请求里用 model=base（不走 LoRA）；serve_loras 要求至少挂一个 LoRA，挂一个不用的
up() { curl -s -m 3 http://127.0.0.1:$1/v1/models | grep -q "\"$2\""; }   # 服务已经开着（重跑评测时）就不重启
up 8001 unused || A=$A/s1 MAX_LORAS=1 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem ${USER_MEM:-0.72} --max-len 16384 --batch ${BATCH:-8192} --vllm-args "--max-num-seqs ${MAX_SEQS:-64}" --base $A/models/Qwen3.5-9B --log $A/vllm-user.log \
  unused=$FS/adapters/rm1__core_9b-9b/adapter >> $LOG 2>&1 || { log "user vllm FAILED: skipping the 2B baseline"; SKIP2=1; }
up 8000 policy_v0 || A=$A/s0 MAX_LORAS=2 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${POL_MEM:-0.22} --max-len 16384 --batch ${BATCH:-8192} --vllm-args "--max-num-seqs ${MAX_SEQS:-64}" --base $Q2 --log $A/vllm-policy.log \
  policy_v0=$Z >> $LOG 2>&1 || { log "2B policy vllm FAILED on the shared card: skipping the 2B baseline, going on to 9B"; SKIP2=1; }
mkdir -p $O
for sp in $([ "${SKIP2:-0}" = 1 ] || echo test train); do
  $PY -u -m cs_agent.evaluate --policy_url http://127.0.0.1:8000/v1 --policy policy_v0 --tok $Q2 --user_url http://127.0.0.1:8001/v1 \
    --split $sp --k $K --concurrency ${CONC:-32} --out $O/$sp.jsonl >> $LOG 2>&1 || log "eval $sp FAILED"
  log "EVAL $sp $(grep -a EVAL_DONE $LOG | tail -1 | cut -c1-400)"
done
pkill -f '[v]llm serve'
# 9B 基座也测一遍（BASE9=0 关掉）：策略和用户模拟器共用同一个 9B 服务（策略走 B 全零的 LoRA，等于基座），决定先训 2B 还是直接 9B
if [ "${BASE9:-1}" = 1 ]; then
  for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
  O9=outputs/rt/cs/base_q9b; mkdir -p $O9
  A=$A/s1 MAX_LORAS=2 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $A/models/Qwen3.5-9B --log $A/vllm-9b.log \
    policy_v0=$FS/adapters/gen-ops-zero-9b/adapter >> $LOG 2>&1 || { log "CS_BASE FAILED: 9b vllm"; exit 1; }
  for sp in test train; do
    $PY -u -m cs_agent.evaluate --policy_url http://127.0.0.1:8001/v1 --policy policy_v0 --tok $A/models/Qwen3.5-9B --user_url http://127.0.0.1:8001/v1 \
      --split $sp --k $K --concurrency ${CONC:-32} --out $O9/$sp.jsonl >> $LOG 2>&1 || log "eval 9b $sp FAILED"
    log "EVAL 9b $sp $(grep -a EVAL_DONE $LOG | tail -1 | cut -c1-400)"
  done
  pkill -f '[v]llm serve'
fi
cp $LOG $O/ 2>/dev/null
log "CS_BASE_DONE"
