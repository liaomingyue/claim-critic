#!/bin/bash
# 用 RAG 模型自己写的、逐句核实过的句子补 critic（ragtruth/own_fill.py）。只用验收原文的 fit 那一半，final 那一半留着验收。
#   ① 基线训练集 + 要加的句子（本地拼好、传到 outputs/rt/own_fill/<轮>/add_{e,en}.jsonl）→ 两组训练集：e = 只加错句，en = 错句 + 难负例
#   ② 各训一个 critic
#   ③ 两个新 critic 给 cont8 / cont9 的起点和第 100 步的回答逐句打分（本地只看 final 那一半，和标注对）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; R=outputs/rt/rm; LOG=$A/own_fill.log
RD=${RD:-own1}; CR=${CR:-rm1__core_r2_bfcl}; NAME=${NAME:-rm1__core_$RD}; B=outputs/rt/own_fill/$RD
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
for f in $R/$CR/train.jsonl $R/$CR/calib.jsonl $R/$CR/meta.json $B/add_e.jsonl $B/add_en.jsonl configs/rt/accept_split.json; do [ -s $f ] || { log "OWNFILL FAILED: missing $f"; exit 1; }; done
python -u -m ragtruth.own_fill build --round $RD --base $CR --name $NAME --arms e en >> $LOG 2>&1 || { log "OWNFILL FAILED: build"; exit 1; }
log "$(grep -a '^OWN_FILL_BUILD' $LOG | tail -1)"
for a in e en; do
  [ -f $R/$NAME$a/eval/report.json ] && continue
  free_gpu; RM=$NAME$a GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f $R/$NAME$a/eval/report.json ] || { log "OWNFILL FAILED: train $NAME$a"; exit 1; }
  log "TRAINED $NAME$a"
done
CRITICS="${NAME}e ${NAME}en" CODE=$PWD bash $S/run_critic_on_rag.sh >> $LOG 2>&1 || { log "OWNFILL FAILED: score"; exit 1; }
mkdir -p $FS/results/own_fill_$RD && cp $LOG $FS/results/own_fill_$RD/ 2>/dev/null
for a in e en; do mkdir -p $FS/results/own_fill_$RD/$NAME$a; cp $R/$NAME$a/meta.json $R/$NAME$a/eval/report.json $FS/results/own_fill_$RD/$NAME$a/; done
log "OWNFILL_DONE"
