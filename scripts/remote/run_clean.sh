#!/bin/bash
# 标签噪声检验（2026-11-03，用户：用清洗后的数据再训一个 critic，看边界区的误报能不能降下来）：
#   rm1__mcgold_h2  ：h1 的错句里 2B P 0.5–0.7 全留（抽查 9% 不是错）+ 0.3–0.5 经 Claude 判为真错的；0.1–0.3（24% 不是错）去掉
#   rm1__mcgold_h2n ：同条数，从 h1 的 1500 条里随机抽、不清洗——区分「标签干净」和「数据变少」
# 都配同位置原句；两张卡各训一个；训完给两批 2B GRPO 验收回答（grpo300、grpo300b）逐句打分。机器不关。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/clean.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B; O=outputs/rt/gentarget
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
mkdir -p $A/s0 $A/s1
for t in h2 h2n; do
  python -u -m ragtruth.critic_harvest --harvest $O/harvest_h1.jsonl --name rm1__mcgold_$t --ids $O/h1clean/ids_$t.txt --max_add 5000 >> $LOG 2>&1 || { log "build $t FAILED"; exit 1; }
done
log "built h2 / h2n"
free_gpus
RM=rm1__mcgold_h2 GPU=0 PORT=8100 bash $S/run_mcrm.sh & T0=$!
RM=rm1__mcgold_h2n GPU=1 PORT=8101 bash $S/run_mcrm.sh & T1=$!
wait $T0; wait $T1
for r in rm1__mcgold rm1__mcgold_h1 rm1__mcgold_h1ctl rm1__mcgold_h2 rm1__mcgold_h2n; do log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"; done
free_gpus
for pair in "0 rm1__mcgold_h2" "1 rm1__mcgold_h2n"; do
  set -- $pair; g=$1; r=$2
  ( A=$A/s$g bash $S/serve_loras.sh --gpu $g --port 800$g --base $MC --log $A/vllm-cl$g.log critic=$FS/adapters/$r-mc/adapter >> $LOG 2>&1 || exit 1
    for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
      python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:800$g/v1 --temp $(temp $r) --out $d/critic_$r.json >> $LOG 2>&1 || exit 1
    done ) & eval "CT$g=\$!"
done
wait $CT0 || log "score h2 FAILED"; wait $CT1 || log "score h2n FAILED"
free_gpus
log "CLEAN_DONE"
