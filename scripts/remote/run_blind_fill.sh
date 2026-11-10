#!/bin/bash
# 多类型补 critic 盲区（ragtruth/blind_fill.py）。范围偏移 / 跨段拼接按 sev 排序各取 300 交 Claude；近义 / 张冠李戴各 100 做「核实 vs 不核实」对照。
#   一、v2h3（本机）+ g1（第二台机器，run_blind_fill2.sh，多开卡并行）各在大提示池（RAG GRPO 版 + SFT 版回答 × 干净句，只这 4 种类型）每个提示温度 0 采 1 条并打分
#   二、select → items.jsonl，等本地 Claude 标完上传 labels.jsonl
#   三、build cl / nc → 本机训 cl、第二台训 nc → 三套验收打分（本地 critic_cap3：显然错 / 任意错 / 按类型）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/blind_fill.log; O=outputs/rt/gen_v2_grpo; B=$O/blind_fill
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
source $S/bg_sync.sh
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
RMS="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
G1=$A/runs/g1/ckpt/step50; [ -d $G1/policy ] && G1=$G1/policy
mkdir -p $B
if [ ! -s $B/items.jsonl ]; then
  if [ ! -s $B/samples_v2h3.jsonl ]; then
    if pgrep -f 'blind_fill sample --model v2h3' > /dev/null; then   # 上一次启动留下的采样还在跑 → 等它
      while pgrep -f 'blind_fill sample --model v2h3' > /dev/null; do sleep 30; done
    else
      free_gpu
      A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bf.log $RMS v2h3=$A/ckpt/gen-v2h3-mc/adapter >> $LOG 2>&1 || { log "vllm FAILED"; exit 1; }
      python -u -m ragtruth.blind_fill sample --model v2h3 --tag v2h3 --tokenizer $MC >> $LOG 2>&1 || { log "sample v2h3 FAILED"; exit 1; }
    fi
  fi
  [ -s $B/samples_v2h3.jsonl ] || { log "sample v2h3 FAILED (no file)"; exit 1; }
  log "v2h3 sampled; waiting for g1 (second machine)"
  free_gpu
  for i in $(seq 1 240); do [ -f $B/G1_DONE ] && break; sleep 60; done
  [ -f $B/G1_DONE ] || log "no g1 samples after 4h, select with v2h3 only"
  python -u -m ragtruth.blind_fill select >> $LOG 2>&1 || { log "select FAILED"; exit 1; }
  log "$(grep -a '^BLIND_FILL_SELECT' $LOG | tail -1)"
fi
log "BLIND_FILL_READY_FOR_CLAUDE"
for i in $(seq 1 480); do [ -s $B/labels.jsonl ] && break; sleep 60; done
[ -s $B/labels.jsonl ] || { log "no claude labels, stop"; exit 1; }
python -u -m ragtruth.blind_fill build >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "BUILT $(grep -a '^BLIND_FILL_BUILD' $LOG | tail -1 | cut -c1-1500)"
for arm in cl; do   # nc 在第二台训（run_blind_fill2.sh）
  name=rm1__core_r2_bf$arm
  if [ ! -f outputs/rt/rm/$name/eval/report.json ]; then
    free_gpu; RM=$name GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
    [ -f outputs/rt/rm/$name/eval/report.json ] || { log "$arm train FAILED"; continue; }
  fi
  T=$(t $name); log "TRAINED $arm T=$T"
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bf$arm.log c=$FS/adapters/$name-mc/adapter >> $LOG 2>&1 || { log "$arm serve FAILED"; continue; }
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
log "BLIND_FILL_DONE (cl)"
