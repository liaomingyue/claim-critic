#!/bin/bash
# 阶段三（st3）：从阶段二续训的第 100 步出发，难度只在 critic 拿不准的区间给分（0.1–0.9 的三角带，0.3–0.5 最高），难度权重固定不变。
# 约束的基准线 = 阶段二续训最后 30 步里真错 / 自然 / 操作一致的平均水平；哪一项掉下去就给哪一项加补偿权重。训完在留出回答上采样打分。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/st3.log
E=${E:-st3}; W=${W:-1.5}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
G=$FS/adapters/gen-ops-grpo
log "stage 3 ($E) from ${FROM:-st2c}: fixed difficulty weight $W, baselines of stage-3 entry"
E=$E STAGE=3 STYLE=free LR=2e-5 STEPS=${STEPS:-100} N=32 PPS=8 DISC=rm2__d2 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-${FROM:-st2c}/step100/policy \
  EXTRA="--ops_disc 1 --ops_ref 0.569 --ops_refs 0.738,0.821,0.686 --ops_w_hard $W --ops_w_fixed 1" CODE=$PWD bash $S/run_genops_grpo.sh >> $LOG 2>&1
[ -d $G-$E/step${STEPS:-100}/policy ] || { log "$E FAILED"; exit 1; }
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ev-$E.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$G-$E/step${STEPS:-100}/policy >> $LOG 2>&1 || { log "ev serve FAILED"; exit 1; }
python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 4 --out samples_ops_ev_$E.jsonl >> $LOG 2>&1 || log "ev sample FAILED"
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
python -u $A/score6.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d1)}" ev_$E >> $LOG 2>&1 || log "ev score FAILED"
pkill -f '[v]llm serve'
python $A/ev_cmp2.py ev_base_free ev_s1c ev_s2d ev_st2 ev_st2c ev_$E > $A/st3_cmp.txt 2>&1
log "ST3_DONE $E"
