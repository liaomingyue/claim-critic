#!/bin/bash
# 多样性再训一轮（st8），从 st4 出发，和 st5–st7 同一个起点。在 st7（和真实错误的分布比）的基础上补四处（--ops_real_v 2）：
# 近义词并成一组再算一次占比（改数字时约数词轮着用）；乘一项操作系数，按分类模型给「要求的操作」的概率（躲开套路词的样本常做成了别的操作）；
# 没有目标的操作按固定上限算词面（加限定语的 may）；单个词的门槛 10% → 8%。其余同 st7。
# XTRA：追加的奖励参数（st9 = st8 再加按手段节点算的组内多样性和全局累计分布两项），E 换名字。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/st8.log; T=scripts/analysis
E=${E:-st8}; W=${W:-3.0}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
G=$FS/adapters/gen-ops-grpo
TG=${TG:-configs/rt/op_real_target.json}
pkill -f '[v]llm serve'; sleep 3
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opd-real.log opd=$FS/adapters/opd__v1-mc/adapter >> $LOG 2>&1 && python -u $T/genops_classify_opd.py real >> $LOG 2>&1 || log "real target FAILED"
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
[ -s $TG ] || { log "$E FAILED: no real target"; exit 1; }
mkdir -p $FS/results/$E && cp $TG $FS/results/$E/
log "stage 3 with diversity against the real-error distribution, CLS=${CLS:-2} ($E) from ${FROM:-st4}: fixed difficulty weight $W"
CLS=${CLS:-2} E=$E STAGE=3 STYLE=free LR=2e-5 STEPS=${STEPS:-100} N=32 PPS=8 DISC=rm2__d2 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-${FROM:-st4}/step100/policy \
  EXTRA="--ops_disc 1 --ops_ref 0.610 --ops_refs 0.756,0.848,0.706 --ops_clear 0.5 --ops_div 0.8 --ops_cls ${CLS:-2} --ops_real_target $TG --ops_real_v 2 --ops_div_cap 2.0 --ops_w_hard $W --ops_w_fixed 1 ${XTRA:-}" CODE=$PWD bash $S/run_genops_grpo.sh >> $LOG 2>&1
[ -d $G-$E/step${STEPS:-100}/policy ] || { log "$E FAILED"; exit 1; }
pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ev-$E.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$G-$E/step${STEPS:-100}/policy >> $LOG 2>&1 || { log "ev serve FAILED"; exit 1; }
python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 4 --out samples_ops_ev_$E.jsonl >> $LOG 2>&1 || log "ev sample FAILED"
t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
python -u $T/genops_score.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d1)}" ev_$E >> $LOG 2>&1 || log "ev score FAILED"
pkill -f '[v]llm serve'; sleep 8
if [ "${CLS:-2}" = 2 ]; then
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opd-$E.log opd=$FS/adapters/opd__v1-mc/adapter >> $LOG 2>&1 && python -u $T/genops_classify_opd.py classify ev_$E >> $LOG 2>&1 || log "classify FAILED"
else
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-opc-$E.log opl=$FS/adapters/opl__v1-mc/adapter opc=$FS/adapters/opc__v1-mc/adapter >> $LOG 2>&1 && python -u $T/genops_classify.py "{\"opl\": $(t opl__v1), \"opc\": $(t opc__v1)}" ev_$E >> $LOG 2>&1 || log "classify FAILED"
fi
pkill -f '[v]llm serve'
python $T/genops_cmp.py ev_base_free ev_s1c ev_s2d ev_st2 ev_st2c ev_st3 ev_st4 ev_st5 ev_st6 ev_st7 $([ $E = st8 ] || echo ev_st8) ev_$E > $A/st8_cmp.txt 2>&1
python $T/genops_steps.py $E > $A/st8_steps.txt 2>&1
gzip -kf $C/samples_ops_ev_$E.jsonl $C/samples_ops_ev_${E}_rm4.jsonl $C/samples_ops_ev_${E}_op[cd].jsonl
mkdir -p $FS/results/$E && cp $A/st8_cmp.txt $A/st8_steps.txt $LOG $FS/results/$E/
log "ST8_DONE $E"
