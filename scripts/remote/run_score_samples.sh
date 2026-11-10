#!/bin/bash
# rm1 / rm2 / rm3(sev) 分别给有 Claude 标注的生成器 v2 样本打分，本地对照 Claude 看各模型靠不靠得住。
# 等 genv2（9B-only 描述版）做完 + 本地上传 gen_v2_label_r2/items.jsonl（UPLOADED 标记）→ 2B 服务挂 rm1 / rm2 / dne / gen 四个适配器 → gen_v2_label/model_scores.jsonl
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/score_samples.log; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until sed -n '/9B-only descriptions/,$p' $A/genv2.log | grep -q GENV2_DONE && [ -f $C/gen_v2_label_r2/UPLOADED ]; do sleep 30; done
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
pkill -f '[v]llm serve'; sleep 5
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
TEMPS="rm1=$(t rm1__core) rm2=$(t rm2__core) dne=$(t sev__gen_dne) gen=$(t sev__gen)"
log "temps $TEMPS"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-score.log rm1=$FS/adapters/rm1__core-mc/adapter rm2=$FS/adapters/rm2__core-mc/adapter dne=$FS/adapters/sev__gen_dne-mc/adapter gen=$FS/adapters/sev__gen-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
python -u -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --temps $TEMPS --out gen_v2_label/model_scores.jsonl >> $LOG 2>&1 || log "score FAILED"
pkill -f '[v]llm serve'
log "SCORE_SAMPLES_DONE"
