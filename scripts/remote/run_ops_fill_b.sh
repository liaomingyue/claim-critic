#!/bin/bash
# 针对性补采：只做挂接层 8 种操作 + 近义替换 + 翻转结果（对应 RAG 剩余显然错最集中的张冠李戴、直接矛盾），
# 在训练集原文的回答上每个任务采 N 条（默认 6），把关模型 + 当前 critic 打分。只采样和打分，挑样、核实、训练另做。
# 新旧卡片对比：HARD=1 TAG=trc_st8 N=3 再跑一遍，其余不变。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis; LOG=$A/ops_fill_b.log
E=${E:-st8}; TAG=${TAG:-trb_$E}; CR=${CR:-rm1__core_r2_bfcl}; N=${N:-6}
CARD="--card"; [ "${HARD:-0}" = 1 ] && CARD="--card --card_hard"   # HARD=1：卡片里加「真实的助手怎么犯这个错、怎样改才难认」（配 TAG=trc_st8）
OPSL="change_speaker move_modifier reverse_direction swap_referent change_owner move_number move_time_place merge_events swap_term flip_outcome"
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
GEN=$FS/adapters/gen-ops-grpo-$E/step100/policy
[ -s $GEN/adapter_model.safetensors ] || { log "OPSFILLB FAILED: missing generator"; exit 1; }
pkill -f '[v]llm serve'; sleep 3
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ofb.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/$CR-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$GEN >> $LOG 2>&1 || { log "OPSFILLB FAILED: serve"; exit 1; }
python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC $CARD --style free --n $N --pool genpool_train.jsonl --ops $OPSL --out samples_ops_$TAG.jsonl >> $LOG 2>&1 || { log "OPSFILLB FAILED: sample"; exit 1; }
python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t $CR), \"disc\": $(t rm2__d1)}" $TAG >> $LOG 2>&1 || { log "OPSFILLB FAILED: score"; exit 1; }
pkill -f '[v]llm serve'
gzip -kf $C/samples_ops_$TAG.jsonl $C/samples_ops_${TAG}_rm4.jsonl
log "$(grep -a '^GENOPS_SAMPLE\|^SCORE4_DONE' $LOG | tail -2 | cut -c1-600 | tr '\n' ' ')"
log "OPSFILLB_DONE"
