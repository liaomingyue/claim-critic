#!/bin/bash
# 2026-11-09 用户：第二轮闭环的造样本 + critic 这一步（第二台机器）。
#   ① 等第二轮 RAG（mc-grpo-r3_core_cont4 第 100 步）→ 在 690 个原文上各采 8 个回答（T 1.0）→ 第一轮 critic（bfcl）挑干净句（P < 0.1）→ rag_pool_r2
#   ② v2h3 / g1 在新提示池上（范围 / 近义 / 张冠李戴 / 拼接）每个提示温度 0 采 1 条，打分的 critic 换成 bfcl
#   ③ 每类 300：一半 sev ≥ 0.9、一半 sev 0.5–0.9（难段）→ 全部交 Claude（本地）→ 只留真错 + 自然 → 加进 bfcl → rm1__core_rd2 → 三套验收打分
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/round2_fill.log; O=outputs/rt/gen_v2_grpo; B=$O/blind_fill_r2
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
CR=rm1__core_r2_bfcl; NEW=rm1__core_rd2
TEMPS="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=$CR"
RMS="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/$CR-mc/adapter"
MECHS="certainty_or_scope_shift=300 near_synonym_shift=300 wrong_attribution=300 merged_across_passages=300"
BF="--round blind_fill_r2 --pools g=rag_pool_r2/prompts.jsonl --verify $MECHS --control"
mkdir -p $A/lora && rsync -a $FS/adapters/gen-v2h3-mc/adapter/ $A/lora/v2h3/ && G1=$O/g1/ckpt/step50 && { [ -d $G1/policy ] && G1=$G1/policy; } && rsync -a $G1/ $A/lora/g1/
RAGD=$FS/adapters/mc-grpo-r3_core_cont4/step100
log "waiting for round-2 RAG ($RAGD)"
for i in $(seq 1 300); do [ -f $RAGD/policy/adapter_model.safetensors ] || [ -f $RAGD/adapter_model.safetensors ] && break; sleep 60; done
[ -d $RAGD/policy ] && RAGD=$RAGD/policy
[ -f $RAGD/adapter_model.safetensors ] || { log "no round-2 RAG after 5h, stop"; exit 1; }
rsync -a $RAGD/ $A/lora/rag/
if [ ! -s $O/rag_pool_r2/prompts.jsonl ]; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2a.log $RMS rag=$A/lora/rag >> $LOG 2>&1 || { log "vllm FAILED"; exit 1; }
  [ -s $O/rag_pool_r2/answers.jsonl ] || python -u -m ragtruth.gen_pool_rag answers --out_dir rag_pool_r2 --model rag --tokenizer $MC --k 8 --temperature 1.0 >> $LOG 2>&1 || { log "answers FAILED"; exit 1; }
  log "$(grep -a '^RAG_POOL_ANSWERS' $LOG | tail -1 | cut -c1-600)"
  python -u -m ragtruth.gen_pool_rag clean --out_dir rag_pool_r2 --tokenizer $MC --thr 0.1 --temps $TEMPS >> $LOG 2>&1 || { log "clean FAILED"; exit 1; }
  log "$(grep -a '^RAG_POOL_CLEAN' $LOG | tail -1 | cut -c1-600)"
fi
if [ ! -s $B/items.jsonl ]; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2b.log $RMS v2h3=$A/lora/v2h3 g1=$A/lora/g1 >> $LOG 2>&1 || { log "vllm FAILED"; exit 1; }
  for g in v2h3 g1; do
    [ -s $B/samples_$g.jsonl ] || python -u -m ragtruth.blind_fill sample $BF --model $g --tag $g --tokenizer $MC --temps $TEMPS >> $LOG 2>&1 || { log "sample $g FAILED"; exit 1; }
    log "$(grep -a '^BLIND_FILL_SAMPLE' $LOG | tail -1)"
  done
  free_gpu
  python -u -m ragtruth.blind_fill select $BF --strat >> $LOG 2>&1 || { log "select FAILED"; exit 1; }
  log "$(grep -a '^BLIND_FILL_SELECT' $LOG | tail -1)"
fi
log "ROUND2_READY_FOR_CLAUDE"
for i in $(seq 1 600); do [ -s $B/labels.jsonl ] && break; sleep 60; done
[ -s $B/labels.jsonl ] || { log "no claude labels, stop"; exit 1; }
python -u -m ragtruth.blind_fill build $BF --base $CR --name $NEW >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "BUILT $(grep -a '^BLIND_FILL_BUILD' $LOG | tail -1 | cut -c1-1500)"
if [ ! -f outputs/rt/rm/$NEW/eval/report.json ]; then
  free_gpu; RM=$NEW GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/$NEW/eval/report.json ] || { log "train FAILED"; exit 1; }
fi
T=$(t $NEW); log "TRAINED $NEW T=$T"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-r2c.log c=$FS/adapters/$NEW-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
PIDS=""
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
  [ -f $d/critic_$NEW.json ] && continue
  python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$NEW.json >> $LOG 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
free_gpu
log "ROUND2_FILL_DONE"
