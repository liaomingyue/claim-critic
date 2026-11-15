#!/bin/bash
# 阶段三再训一轮（st4），纠 Claude 抽查发现的两个偏向：难度分乘明显错系数（--ops_clear），同一任务里同一种改法分摊奖励（--ops_div）。
# 难度项乘了系数后幅度变小，权重从 1.5 提到 3.0，让它在组内的波动仍然不低于阶段一的分。基准线 = 阶段三最后 30 步的平均水平。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/st4.log; T=scripts/analysis
E=${E:-st4}; W=${W:-3.0}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
G=$FS/adapters/gen-ops-grpo
log "stage 3 with clear-error weighting and group diversity ($E) from ${FROM:-st3}: fixed difficulty weight $W"
E=$E STAGE=3 STYLE=free LR=2e-5 STEPS=${STEPS:-100} N=32 PPS=8 DISC=rm2__d2 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-${FROM:-st3}/step100/policy \
  EXTRA="--ops_disc 1 --ops_ref 0.592 --ops_refs 0.746,0.833,0.707 --ops_clear 0.5 --ops_div 0.8 --ops_w_hard $W --ops_w_fixed 1" CODE=$PWD bash $S/run_genops_grpo.sh >> $LOG 2>&1
[ -d $G-$E/step${STEPS:-100}/policy ] || { log "$E FAILED"; exit 1; }
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ev-$E.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$G-$E/step${STEPS:-100}/policy >> $LOG 2>&1 || { log "ev serve FAILED"; exit 1; }
python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 4 --out samples_ops_ev_$E.jsonl >> $LOG 2>&1 || log "ev sample FAILED"
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d1)}" ev_$E >> $LOG 2>&1 || log "ev score FAILED"
pkill -f '[v]llm serve'
python $T/genops_cmp.py ev_base_free ev_s1c ev_s2d ev_st2 ev_st2c ev_st3 ev_$E > $A/st4_cmp.txt 2>&1
python $T/genops_steps.py $E > $A/st4_steps.txt 2>&1
gzip -kf $C/samples_ops_ev_$E.jsonl $C/samples_ops_ev_${E}_rm4.jsonl
mkdir -p $FS/results/$E && cp $A/st4_cmp.txt $A/st4_steps.txt $LOG $FS/results/$E/
log "ST4_DONE $E"
