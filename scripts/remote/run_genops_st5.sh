#!/bin/bash
# 再训一轮（st5），在 st4 的两项纠偏之外加跨任务的多样性：手段节点占比（--ops_node_base）和事后分类的类别占比（CLS=1，--ops_cls）。
# 其余同 st4：难度只奖励 critic 拿不准的区间并乘明显错系数，权重固定 3.0。基准线 = st4 最后 30 步的平均水平。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/st5.log; T=scripts/analysis
E=${E:-st5}; W=${W:-3.0}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
G=$FS/adapters/gen-ops-grpo
log "stage 3 with pattern-node and category diversity, CLS=${CLS:-1} ($E) from ${FROM:-st4}: fixed difficulty weight $W"
CLS=${CLS:-1} E=$E STAGE=3 STYLE=free LR=2e-5 STEPS=${STEPS:-100} N=32 PPS=8 DISC=rm2__d2 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-${FROM:-st4}/step100/policy \
  EXTRA="--ops_disc 1 --ops_ref 0.610 --ops_refs 0.756,0.848,0.706 --ops_clear 0.5 --ops_div 0.8 --ops_node_base configs/rt/op_node_base.json --ops_cls ${CLS:-1} --ops_w_hard $W --ops_w_fixed 1" CODE=$PWD bash $S/run_genops_grpo.sh >> $LOG 2>&1
[ -d $G-$E/step${STEPS:-100}/policy ] || { log "$E FAILED"; exit 1; }
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ev-$E.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$G-$E/step${STEPS:-100}/policy >> $LOG 2>&1 || { log "ev serve FAILED"; exit 1; }
python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 4 --out samples_ops_ev_$E.jsonl >> $LOG 2>&1 || log "ev sample FAILED"
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d1)}" ev_$E >> $LOG 2>&1 || log "ev score FAILED"
pkill -f '[v]llm serve'; sleep 8
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opc-$E.log opl=$FS/adapters/opl__v1-mc/adapter opc=$FS/adapters/opc__v1-mc/adapter >> $LOG 2>&1 && python -u $T/genops_classify.py "{\"opl\": $(t opl__v1), \"opc\": $(t opc__v1)}" ev_$E >> $LOG 2>&1 || log "classify FAILED"
pkill -f '[v]llm serve'
python $T/genops_cmp.py ev_base_free ev_s1c ev_s2d ev_st2 ev_st2c ev_st3 ev_st4 ev_$E > $A/st5_cmp.txt 2>&1
python $T/genops_steps.py $E > $A/st5_steps.txt 2>&1
gzip -kf $C/samples_ops_ev_$E.jsonl $C/samples_ops_ev_${E}_rm4.jsonl $C/samples_ops_ev_${E}_opc.jsonl
mkdir -p $FS/results/$E && cp $A/st5_cmp.txt $A/st5_steps.txt $LOG $FS/results/$E/
log "ST5_DONE $E"
