#!/bin/bash
# 定位用：补过盲区样本的 critic 训出来的 RAG 模型，它自己写错的句子，critic 认不认得出。
#   回答：cont8（critic = rm1__core_of1all）和 cont9（核实组）的起点、第 100 步在三套验收原文上的回答，这些句子 Claude 已经逐句标过。
#   critic：基线 rm1__core_r2_bfcl、两组各自训练时用的 rm1__core_of1all / rm1__core_of1cl。
#   输出 <eval>/<版本>_<集>_critic_<名>.json（和 critic_score 同格式），本地和 Claude 的标注对。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/critic_on_rag.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
CRITICS=${CRITICS:-rm1__core_r2_bfcl rm1__core_of1all rm1__core_of1cl}
SPEC=""; for c in $CRITICS; do
  [ -d $FS/adapters/$c-mc/adapter ] && [ -f outputs/rt/rm/$c/eval/report.json ] || { log "missing critic $c"; exit 1; }
  SPEC="$SPEC $c=$FS/adapters/$c-mc/adapter"
done
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-cor.log $SPEC >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
for arm in ${ARMS:-cont8 cont9}; do
  K=outputs/rt/core/loop2_$arm/eval
  for m in v0 s100; do for s in a b c; do
    [ -f $K/${m}_$s.jsonl ] || { log "missing $K/${m}_$s.jsonl"; continue; }
    for c in $CRITICS; do
      T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$c/eval/report.json'))['temperature'])")
      [ -f $K/${m}_${s}_critic_$c.json ] || python -u -m ragtruth.critic_score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model $c --temp $T --out $K/${m}_${s}_critic_$c.json >> $LOG 2>&1
    done
  done; done
  log "scored $arm: $(ls $K/*_critic_*.json 2>/dev/null | wc -l) files"
done
pkill -f '[v]llm serve'
log "CRITIC_ON_RAG_DONE"
