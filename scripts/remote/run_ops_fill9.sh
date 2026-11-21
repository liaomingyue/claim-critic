#!/bin/bash
# 全 9B 链路的第 3 步：9B 错误合成模型在训练集原文的回答上造错 → 9B 把关模型和 9B critic 打分 → 取盲区（三项都过且 critic < 0.5，不经 Claude）
# → 补进 critic 训练集，在 9B 底座上重训（run_crit9b.sh）→ 给已标注的回答打分（9B RAG 回答 + 2B 各组回答），本地和基线 9B critic 比。
#   GEN_E=错误合成模型的运行名（默认 s3c9，第 100 步）  RD=轮次目录（默认 of9）  NAME=新 critic 数据名前缀（默认 rm1__core_of9，组名 all）
#   基线 critic = rm1__core_r2_bfcl（训练集同 rm1__core_r2_bfcl_9b）。单卡 5090。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis; LOG=$A/ops_fill9.log
GEN_E=${GEN_E:-s3c9}; RD=${RD:-of9}; NAME=${NAME:-rm1__core_of9}; CR=rm1__core_r2_bfcl; TAG=tr9_$GEN_E; B=outputs/rt/ops_fill/$RD
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
source $S/models9b.sh; ensure_9b || { log "OF9 FAILED: 9b model"; exit 1; }; B9=$A/models/Qwen3.5-9B
ad() { echo $FS/adapters/${1}_9b-9b/adapter; }
t() { python -c "import json;print(json.load(open('$R/${1}_9b/eval/report.json'))['temperature'])"; }
GEN=$FS/adapters/gen-ops-grpo-$GEN_E/step100/policy
for f in $GEN/adapter_model.safetensors $C/genpool_train.jsonl $R/$CR/train.jsonl; do [ -s $f ] || { log "OF9 FAILED: missing $f"; exit 1; }; done
if [ ! -s $B/zone.jsonl ]; then
  if [ ! -s $C/samples_ops_${TAG}_rm9.jsonl ]; then
    free_gpu
    A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-of9.log sev4=$(ad sev__v4) nat4=$(ad rm2__v4) opm=$(ad opm__v1) rm1=$(ad $CR) disc=$(ad rm2__d2) g=$GEN >> $LOG 2>&1 || { log "OF9 FAILED: serve"; exit 1; }
    [ -s $C/samples_ops_$TAG.jsonl ] || python -u -m ragtruth.gen_ops sample --model g --tokenizer $B9 --card --style free --n 1 --pool genpool_train.jsonl --out samples_ops_$TAG.jsonl >> $LOG 2>&1 || { log "OF9 FAILED: sample"; exit 1; }
    GENOPS_SUF=rm9 python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t $CR), \"disc\": $(t rm2__d2)}" $TAG >> $LOG 2>&1 || { log "OF9 FAILED: score"; exit 1; }
    free_gpu
  fi
  python -u -m ragtruth.ops_fill select --tag $TAG --suf rm9 --round $RD >> $LOG 2>&1 || { log "OF9 FAILED: select"; exit 1; }
fi
log "$(grep -a '^OPS_FILL_SELECT' $LOG | tail -1 | cut -c1-1500)"
[ -s $R/${NAME}all/train.jsonl ] || python -u -m ragtruth.ops_fill build --round $RD --base $CR --name $NAME --arms all >> $LOG 2>&1 || { log "OF9 FAILED: build"; exit 1; }
log "$(grep -a '^OPS_FILL_BUILD' $LOG | tail -1 | cut -c1-600)"
free_gpu
SETS=${SETS:-q9b_p9r9:s200 q9b_p9r2:s200 cont8:v0 cont8:s100 cont9:v0 cont9:s100 cont9dyn:s100 cont8dyn:s100} SRC=${NAME}all CODE=$PWD bash $S/run_crit9b.sh >> $LOG 2>&1 || { log "OF9 FAILED: crit9b"; exit 1; }
gzip -kf $C/samples_ops_$TAG.jsonl $C/samples_ops_${TAG}_rm9.jsonl
log "OF9_DONE"
