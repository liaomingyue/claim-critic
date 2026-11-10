#!/bin/bash
# 用定向造错生成器的数据重训 2B critic：
#   GEN=outputs/rt/gentarget/grpo_seed5_a7/ckpt/step50 TAG=h1 bash run_harvest.sh
# 1) 双卡造数据：两张卡各起一个 2B 服务（生成器 + rm1 / rm2 / rm3），训练 prompt 每个采 SAMPLES 条，打分全存（gen_target_data harvest）
# 2) 建两份 critic 数据（都只往训练集加，校准 / 测试集 = rm1__mcgold 的，测试集是 RAGTruth++）：
#      rm1__mcgold_$TAG      ：边界样本（2B P(判出) 0.1–0.7）+ 配对原句
#      rm1__mcgold_${TAG}ctl ：对照，同样条数、不限区间（合格的都行）+ 配对原句——区分「加了生成数据」和「加的是边界数据」
# 3) 两张卡各训一个（run_mcrm.sh：合并、校准、RAGTruth++ 报告）
# 4) 按错误类型的召回：新 critic 打 4 组验收回答（Claude 逐句标过），和基线 rm1__mcgold 比（critic_types）
# 机器不关（后面还要做实验）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/harvest.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B; O=outputs/rt/gentarget
GEN=${GEN:?}; TAG=${TAG:?}; SAMPLES=${SAMPLES:-8}; MAXADD=${MAXADD:-1500}; NEW=rm1__mcgold_$TAG; CTL=rm1__mcgold_${TAG}ctl
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
mkdir -p $A/s0 $A/s1
P=$GEN; [ -d $GEN/policy ] && P=$GEN/policy

if [ ! -f $O/harvest_$TAG.jsonl ]; then
  free_gpus
  for g in 0 1; do
    A=$A/s$g bash $S/serve_loras.sh --gpu $g --port 800$g --base $MC --log $A/vllm-hv$g.log \
      gen=$P rm1=$FS/adapters/rm1__mcgold-mc/adapter rm2=$FS/adapters/rm2v2__mcgen3-mc/adapter rm3=$FS/adapters/rm3__mc-mc/adapter >> $LOG 2>&1 & eval "SV$g=\$!"
  done
  wait $SV0 && wait $SV1 || { log "harvest vllm FAILED"; exit 1; }
  log "harvest $TAG: $GEN, $SAMPLES per prompt"
  python -u -m ragtruth.gen_target_data harvest --url http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 --model gen --inp prompts.jsonl --samples $SAMPLES \
    --rm1 rm1 --rm1_temp $(temp rm1__mcgold) --rm2 rm2 --rm2_temp $(temp rm2v2__mcgen3) --rm3 rm3 --rm3_temp $(temp rm3__mc) --gate 0.7 --out harvest_$TAG.jsonl >> $LOG 2>&1 \
    || { log "harvest FAILED"; exit 1; }
  free_gpus
fi
python -u -m ragtruth.critic_harvest --harvest $O/harvest_$TAG.jsonl --name $NEW --lo 0.1 --hi 0.7 --max_add $MAXADD >> $LOG 2>&1 || { log "build $NEW FAILED"; exit 1; }
N=$(python -c "import json;print(json.load(open('outputs/rt/rm/$NEW/meta.json'))['added_errors'])")
python -u -m ragtruth.critic_harvest --harvest $O/harvest_$TAG.jsonl --name $CTL --lo 0.0 --hi 1.01 --max_add $N >> $LOG 2>&1 || { log "build $CTL FAILED"; exit 1; }
log "built $NEW and $CTL ($N errors each)"
RM=$NEW GPU=0 PORT=8100 bash $S/run_mcrm.sh & T0=$!
RM=$CTL GPU=1 PORT=8101 bash $S/run_mcrm.sh & T1=$!
wait $T0; wait $T1
for r in rm1__mcgold $NEW $CTL; do log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"; done

# 按错误类型的召回：4 组验收回答（minicpm sys / grpo300、base9b、ansgrpo accept_sat）
free_gpus
ANS="outputs/rt/minicpm/sys/answers.jsonl outputs/rt/minicpm/grpo300/answers.jsonl outputs/rt/base9b/answers_accept.jsonl outputs/rt/ansgrpo/accept_sat/answers.jsonl"
for pair in "0 $NEW" "1 $CTL"; do
  set -- $pair; g=$1; r=$2
  ( A=$A/s$g bash $S/serve_loras.sh --gpu $g --port 800$g --base $MC --log $A/vllm-ct$g.log critic=$FS/adapters/$r-mc/adapter >> $LOG 2>&1 || exit 1
    for f in $ANS; do
      [ -f $f ] || continue
      python -u -m ragtruth.critic_score --answers $f --url http://127.0.0.1:800$g/v1 --temp $(temp $r) --out $(dirname $f)/critic_$r.json >> $LOG 2>&1 || exit 1
    done ) & eval "CT$g=\$!"
done
wait $CT0 || log "critic_score $NEW FAILED"; wait $CT1 || log "critic_score $CTL FAILED"
free_gpus
RUNS=""
for grp in "2b_base outputs/rt/minicpm/sys" "2b_grpo outputs/rt/minicpm/grpo300" "9b_base outputs/rt/base9b" "9b_grpo outputs/rt/ansgrpo/accept_sat"; do
  set -- $grp
  RUNS="$RUNS $1|base=$2:critic2b_scores.json $1|$TAG=$2:critic_$NEW.json $1|${TAG}ctl=$2:critic_$CTL.json"
done
python -u -m ragtruth.critic_types --runs $RUNS --out outputs/rt/critic_types_$TAG.json 2>&1 | tee -a $LOG
log "HARVEST_DONE $TAG"
