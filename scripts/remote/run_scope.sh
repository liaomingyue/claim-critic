#!/bin/bash
# 2026-11-09 用户：给 critic 补范围偏移，比「过 Claude」和「不过 Claude」（ragtruth/scope_critic.py）。接在 run_v3c.sh 后面（同一台机器）。
#   一、9B 生成器 v2 在 genpool_div 上按范围偏移采样（每条回答 4 条）→ 9B 基座写描述
#   二、2B：sev（v3c / v3c0 里范围偏移更好的）+ RM2 v3b + critic r2 打分 → 过滤 → 等本地 Claude 核实（scope_label/labels.jsonl 上传后继续）
#   三、建 cl / nc 两组 → 训 critic → 给三套验收回答打分（本地 critic_cap3 对比 r2）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/scope.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
until grep -aq V3C_DONE $A/v3c.log 2>/dev/null; do sleep 60; done
log "v3c done, start scope"

# 一、采样 + 描述（9B）
if [ ! -s $C/scope_desc9b.jsonl ]; then
  free_gpu; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-scope9.log gen=$FS/adapters/gen-v2-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  [ -s $C/samples_scope_9b.jsonl ] || python -u -m ragtruth.gen_v2 sample --model gen --tokenizer $B9 --pool genpool_div.jsonl --mechs certainty_or_scope_shift --n 4 --out samples_scope_9b.jsonl >> $LOG 2>&1 || { log "sample FAILED"; exit 1; }
  log "SAMPLED $(grep -a GENV2_SAMPLE $LOG | tail -1 | cut -c1-300)"
  python -u -m ragtruth.scope_critic items >> $LOG 2>&1 || { log "items FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2 describe --model base --samples samples_scope_9b.jsonl --no_mech --out scope_desc9b.jsonl >> $LOG 2>&1 || { log "describe FAILED"; exit 1; }
  log "DESC $(grep -a DESC9B $LOG | tail -1)"
  free_gpu
fi

# 二、打分 + 过滤（2B）
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
python -u -m ragtruth.scope_critic pick_sev >> $LOG 2>&1 || { log "pick FAILED"; exit 1; }
PICK=$(cat $C/scope_pool/sev_pick.txt); log "PICK $(grep -a PICK_SEV $LOG | tail -1)"
if [ ! -s $C/scope_pool/scores.jsonl ]; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-scope2.log v3=$FS/adapters/sev__$PICK-mc/adapter \
    rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
  python -u -m ragtruth.score_samples --items scope_pool/items.jsonl --desc scope_desc9b.jsonl \
    --temps v3=$(t sev__$PICK) rm2v3=$(t rm2__v3b) rm1=$(t rm1__core_r2) --out scope_pool/scores.jsonl >> $LOG 2>&1 || { log "score FAILED"; exit 1; }
  free_gpu
fi
python -u -m ragtruth.scope_critic filter >> $LOG 2>&1 || { log "filter FAILED"; exit 1; }
log "SCOPE_READY_FOR_CLAUDE $(grep -a SCOPE_FILTER $LOG | tail -1)"

# 等本地 Claude 核实结果上传（最多 4 小时）
for i in $(seq 1 240); do [ -s $C/scope_label/labels.jsonl ] && break; sleep 60; done
[ -s $C/scope_label/labels.jsonl ] || { log "no claude labels, stop"; exit 1; }
python -u -m ragtruth.scope_critic build >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "BUILT $(grep -a SCOPE_BUILD $LOG | tail -1)"

# 三、训 critic + 打分
for arm in cl nc; do
  name=rm1__core_r2_sc$arm
  if [ ! -f outputs/rt/rm/$name/eval/report.json ]; then
    free_gpu; RM=$name GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
    [ -f outputs/rt/rm/$name/eval/report.json ] || { log "$arm train FAILED"; continue; }
  fi
  T=$(t $name); log "TRAINED $arm T=$T"
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-sc$arm.log c=$FS/adapters/$name-mc/adapter >> $LOG 2>&1 || { log "$arm serve FAILED"; continue; }
  PIDS=""
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
    [ -f $d/critic_$name.json ] && continue
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$name.json >> $LOG 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait $p; done
  free_gpu
  log "SCORED $arm"
done
log "SCOPE_DONE"
