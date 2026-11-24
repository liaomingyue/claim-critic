#!/bin/bash
# 客服 Agent GRPO 的训练机任务（由 run_train_node.sh 设好 RM_URL / LORA_SYNC_URL / VLLM_API_KEY 后调用；JOB=run_cs_grpo.sh）。
#   推理机：run_infer_node.sh，INFER_MODEL=qwen3.5-9b MAX_LEN=32768，runs/pair/<配对>/loras 写 policy_v0=$FS/adapters/cs-zero-9b
#   策略起点 = 9B 原版（B 全零 LoRA）；奖励 = 全局（结果模型 P(正确) + S1 规则一票否决）+ 步骤（过程 critic，去底噪）。
#   训完在推理机上评测 τ² 测试集（40 任务 × 4 次，策略温度 0.6，同基线），EVAL_STEPS 列出的中间版本一起评。
#   SMOKE=1：1 个任务 × 2 条、1 步，只验证流程（不评测）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; RUN=${RUN:-cs_grpo_r1}; O=$A/$RUN; LOG=$A/cs_grpo.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $S/setup_tau2.sh || { log "CS_GRPO FAILED: setup_tau2"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "CS_GRPO FAILED: 9b copy"; exit 1; }; B9=$A/models/Qwen3.5-9B
INIT=$FS/adapters/cs-zero-9b
[ -f $INIT/adapter_model.safetensors ] || $PY scripts/make_lora_init.py $B9 $INIT >> $LOG 2>&1 || { log "CS_GRPO FAILED: zero lora"; exit 1; }
[ -s outputs/rt/cs/critic_step/scores.jsonl ] || { log "CS_GRPO FAILED: missing step critic scores (floor caps)"; exit 1; }
[ -f $FS/adapters/cs-critic-step-9b/head.pt ] || { log "CS_GRPO FAILED: missing step critic"; exit 1; }
[ -f $FS/adapters/cs-critic-outcome-9b/head.pt ] || { log "CS_GRPO FAILED: missing outcome critic"; exit 1; }
OUTRM="--critic_outcome $FS/adapters/cs-critic-outcome-9b"   # 结果模型（三分类第 1 个 epoch，奖励用 GOOD / FAIL 重新归一的 P(正确)）
URL=${RM_URL:?need RM_URL (inference node)}
EXTRA=""; [ "${SMOKE:-0}" = 1 ] && { EXTRA="--steps 1 --tasks_per_step 1 --n 2"; O=$A/${RUN}_smoke; }
log "start $RUN -> $O (policy/user $URL)"
$PY -u -m cs_agent.grpo --base $B9 --init $INIT --critic_step $FS/adapters/cs-critic-step-9b $OUTRM \
  --floor_caps outputs/rt/cs/critic_step/scores.jsonl --policy_url $URL --user_url $URL --user_key "$VLLM_API_KEY" --out $O --steps ${STEPS:-60} --concurrency ${CONC:-16} $EXTRA ${GRPO_ARGS:-} >> $LOG 2>&1 \
  || { log "CS_GRPO FAILED: train"; exit 1; }
F=$O/final; [ -d $F/policy ] && F=$F/policy
mkdir -p $FS/runs/$RUN && rsync -a --exclude 'lora/' $O/ $FS/runs/$RUN/ && rsync -a $F/ $FS/adapters/$RUN-9b/
for v in ${EVAL_STEPS:-}; do d=$O/lora/v$v; [ -d $d/policy ] && d=$d/policy; [ -d $d ] && rsync -a $d/ $FS/adapters/$RUN-v$v-9b/; done
log "CS_GRPO_TRAINED $RUN"
[ "${SMOKE:-0}" = 1 ] && { log "CS_GRPO_DONE $RUN (smoke)"; exit 0; }
# 评测：推理机还开着（train_done 之后才收尾），把要评的版本推上去，用和基线相同的设置跑测试集
export USER_API_KEY=$VLLM_API_KEY
for tag in final ${EVAL_STEPS:-}; do
  [ $tag = final ] && AD=$FS/adapters/$RUN-9b NAME=${RUN}_final || { AD=$FS/adapters/$RUN-v$tag-9b; NAME=${RUN}_v$tag; }
  [ -d $AD ] || continue
  $PY -c "from train.gen_grpo import Server; Server('$URL').load_lora('$NAME', '$AD')" >> $LOG 2>&1 || { log "push $NAME FAILED"; continue; }
  EO=outputs/rt/cs/${NAME}_q9b; mkdir -p $EO
  $PY -u -m cs_agent.evaluate --policy_url $URL --policy $NAME --tok $B9 --user_url $URL --split test --k ${K:-4} --concurrency ${ECONC:-16} \
    --out $EO/test.jsonl >> $LOG 2>&1 || log "eval $NAME FAILED"
  $PY -c "import json; from cs_agent.rules import summarize; eps=[json.loads(l) for l in open('$EO/test.jsonl')]; s=summarize(eps); open('$EO/test_rules.json','w').write(json.dumps(s, indent=1)); print('RULES', json.dumps(s))" >> $LOG 2>&1
  log "EVAL $NAME $(grep -a EVAL_DONE $LOG | tail -1 | cut -c1-400)"
done
cp $LOG $FS/runs/$RUN/ 2>/dev/null
log "CS_GRPO_DONE $RUN"
