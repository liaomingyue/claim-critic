#!/bin/bash
# 补打分（2026-11-05：swap_scale 前缀匹配误把 nohA 当成 noh，提前停了旧脚本，noh 的训练还在跑但打分没人做）。
# 等 R 的 run_mcrm 跑完（report.json + adapter），用小显存 vLLM（训练还在同卡上）给 grpo300 / grpo300b / cmp_h2 打分。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; R=${R:?}; LOG=$A/scale.log; PORT=${PORT:-8011}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
until [ -f outputs/rt/rm/$R/eval/report.json ] && [ -f $FS/adapters/$R-mc/adapter/adapter_model.safetensors ] && ! pgrep -f "[c]onfigs/rt/exp/$R-mc.yaml" > /dev/null; do sleep 30; done
sleep 60
log "$R report $(python -c "import json;r=json.load(open('outputs/rt/rm/$R/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"
mkdir -p $A/s9
A=$A/s9 bash scripts/remote/serve_loras.sh --gpu 0 --port $PORT --mem 0.3 --base $A/models/MiniCPM5-2B --log $A/vllm-score1.log c1=$FS/adapters/$R-mc/adapter >> $LOG 2>&1 || { log "score_one serve $R FAILED"; exit 1; }
T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$R/eval/report.json'))['temperature'])")
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
  python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:$PORT/v1 --model c1 --temp $T --out $d/critic_$R.json >> $LOG 2>&1 || log "score $R $d FAILED"
done
for s in old_a old_b new_a new_b; do
  python -u -m ragtruth.critic_score --answers outputs/rt/minicpm/cmp_h2/$s.jsonl --url http://127.0.0.1:$PORT/v1 --model c1 --temp $T --out outputs/rt/minicpm/cmp_h2/${s}_$R.json >> $LOG 2>&1 || log "score $R $s FAILED"
done
pkill -f "[v]llm serve.*--port $PORT"
log "SCALE_ONE_DONE $R (score_one)"
