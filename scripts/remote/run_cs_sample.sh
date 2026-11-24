#!/bin/bash
# 客服 Agent：在训练任务上大规模采样轨迹并自动标注（结果模型 / 步骤模型的训练数据，第 0 轮）。单卡 5090，不训练、不调 Claude。
#   并发：每个策略 8 条（两个策略共 16 条长对话；每条对话占两份上下文：策略看到的和用户模拟器看到的）。12 条开局 KV 43%，对话变长后也到 98%、抢占 528 次。实测 24 条/策略开局就占满 KV、反复抢占，18 条随对话变长也会占满，吞吐反而降到 1/3。
#   一个 9B 服务：基座（用户模拟器）+ 几个策略 LoRA（POLS="名=adapter ..."，默认 9B 基座和 SFT-B），几个策略同时采，共用一张卡。
#   每个策略：训练集 74 个任务 × K 条（默认 16），策略温度 1.0（要多样）；采完用 cs_agent.label 打轨迹级 / 步骤级标签。
#   采完再用带特权信息的 9B 逐步判断（cs_agent.judge）。
#   结果 outputs/rt/cs/<NAME>/<策略>_train.jsonl、<策略>_labels.jsonl、<策略>_label_stats.txt、<策略>_judge.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; NAME=${NAME:-sample_r0}; K=${K:-16}; O=outputs/rt/cs/$NAME; LOG=$A/cs_sample.log
POLS=${POLS:-"base9=$FS/adapters/gen-ops-zero-9b/adapter sftB=$FS/adapters/cs-sft-B-9b"}   # 名字不能叫 base（vLLM 里基座的名字，用户模拟器用它）
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
source $S/setup_tau2.sh || { log "CS_SAMPLE FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_SAMPLE FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
for kv in $POLS; do [ -f ${kv#*=}/adapter_model.safetensors ] || { log "CS_SAMPLE FAILED: no adapter ${kv#*=}"; exit 1; }; done
A=$A/s1 MAX_LORAS=4 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.9 --max-len 32768 --base $B9 --log $A/vllm-sample.log $POLS >> $LOG 2>&1 || { log "CS_SAMPLE FAILED: vllm"; exit 1; }
mkdir -p $O
PIDS=""
for kv in $POLS; do p=${kv%%=*}
  $PY -u -m cs_agent.evaluate --policy_url http://127.0.0.1:8001/v1 --policy $p --tok $B9 --user_url http://127.0.0.1:8001/v1 \
    --split train --k $K --temp ${TEMP:-1.0} --concurrency ${CONC:-8} --out $O/${p}_train.jsonl >> $A/cs_sample_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done   # 只等采样进程（保活进程也是后台任务，不能一起等）
for kv in $POLS; do p=${kv%%=*}
  log "EVAL $p $(grep -a EVAL_DONE $A/cs_sample_$p.log | tail -1 | cut -c1-300)"
  $PY -m cs_agent.label --eps $O/${p}_train.jsonl --split train --out $O/${p}_labels.jsonl > $O/${p}_label_stats.txt 2>> $LOG || log "label $p FAILED"
  log "LABEL $p $(tr -d '\n ' < $O/${p}_label_stats.txt | cut -c1-400)"
done
touch $O/SAMPLE_DONE   # 判断机（run_cs_judge.sh）看到它就做最后一轮补判
[ "${NO_JUDGE:-0}" = 1 ] && { pkill -f '[v]llm serve'; cp $LOG $A/cs_sample_*.log $O/ 2>/dev/null; log "CS_SAMPLE_DONE (judge on another machine)"; exit 0; }
# 带特权信息的 9B 逐步判断（开思考）：全部负样本轨迹 + 每个策略随机 JUDGE_POS 条正样本；步骤模型的主要标签来源，和上面的规则 / 标准动作标签对照
for kv in $POLS; do p=${kv%%=*}
  $PY -u -m cs_agent.judge --eps $O/${p}_train.jsonl --labels $O/${p}_labels.jsonl --split train --url http://127.0.0.1:8001/v1 \
    --select neg+${JUDGE_POS:-0.15} --concurrency ${CONC:-24} --out $O/${p}_judge.jsonl >> $A/cs_judge_$p.log 2>&1 &
done
sleep 10; while pgrep -f 'cs_agent.judge' > /dev/null; do sleep 30; done
for kv in $POLS; do p=${kv%%=*}; log "JUDGE $p $(grep -a JUDGE_DONE $A/cs_judge_$p.log | tail -1 | cut -c1-300)"; done
pkill -f '[v]llm serve'
cp $LOG $A/cs_sample_*.log $A/cs_judge_*.log $O/ 2>/dev/null
log "CS_SAMPLE_DONE"
