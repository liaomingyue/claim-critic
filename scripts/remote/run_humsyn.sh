#!/bin/bash
# 纯人工 vs 纯合成 critic（对比一下纯合成数据和人工数据，判断一下效果）。ARM=hum|syn，一台单卡跑一组：
#   hum = RAGTruth 原版人工 2517 + 拒答 513 = 3030；syn = 早期合成 907 + 生成器 805 对（1610，RM2 ≥ 0.3，取自 n1500 池前 805 对）+ 拒答 513 = 3030
#   两组都不含 b9 / gold（Claude 标的真实回答）；校准 / 测试集相同；训练配方同 ref（configs/rt/rm_mc.yaml）。
#   ① 建数据 → ② 训 critic（run_mcrm.sh：校准 + RAGTruth++）→ ③ 扩充 Claude 集打分（grpo300 / b / c，本地 critic_cap3 算）
#   ④ R3 200 步（同四档：TRIP_KL 0.2，评第 50 / 100 / 200 步，不重评 v0）→ ⑤ 被钻空子：自己的 critic 打第 200 步回答和 v0
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; ARM=${ARM:?hum|syn}; LOG=$A/humsyn_$ARM.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg
MC=$A/models/MiniCPM5-2B; M=outputs/rt/minicpm; R=rm1__mcgold_$ARM
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
# ①
case $ARM in
  hum) python -u -m ragtruth.critic_subset --base rm1__mcgold_ref --name $R --keep human refuse_ >> $LOG 2>&1 ;;
  syn) python -u -m ragtruth.critic_subset --base rm1__mcgold_ref_n1500 --name $R --keep synth harvest_ refuse_ --cap_groups harvest_ 805 >> $LOG 2>&1 ;;
esac || { log "build FAILED"; exit 1; }
log "built $R: $(python -c "import json;m=json.load(open('outputs/rt/rm/$R/meta.json'));print(m['train'], m['by_origin'])")"
# ②
if [ ! -f outputs/rt/rm/$R/eval/report.json ]; then
  rm -rf $FS/runs/claims/mcrm_$R; free_gpu
  RM=$R GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh || { log "train $R FAILED"; exit 1; }
fi
log "$R report $(python -c "import json;r=json.load(open('outputs/rt/rm/$R/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"
# ③
free_gpu; mkdir -p $A/s0
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-humsyn.log c=$FS/adapters/$R-mc/adapter >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
T=$(temp $R)
j="["; for d in grpo300 grpo300b grpo300c; do j+="{\"model\":\"c\",\"temp\":$T,\"answers\":\"$M/$d/answers.jsonl\",\"out\":\"$M/$d/critic_$R.json\"},"; done
echo "${j%,}]" > $A/jobs_cap_$ARM.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $A/jobs_cap_$ARM.json --concurrency 384 >> $LOG 2>&1 || log "cap score FAILED"
free_gpu; log "cap scored"
# ④
TRIP_KL=0.2 CRITIC=$R TAG=_$ARM STEPS=200 EVAL_STEPS="50 100 200" EVAL_V0=0 bash $S/run_r3full.sh || log "r3full FAILED"
# ⑤
X=$M/r3cmp_exploit; mkdir -p $X; free_gpu
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-humsynx.log c=$FS/adapters/$R-mc/adapter >> $LOG 2>&1 || { log "serve x FAILED"; exit 1; }
j="["; for s in a b c; do j+="{\"model\":\"c\",\"temp\":$T,\"answers\":\"$M/r3full_$ARM/eval/s200_$s.jsonl\",\"out\":\"$X/${ARM}_s200_$s.json\"},{\"model\":\"c\",\"temp\":$T,\"answers\":\"$M/r3full/eval/v0_$s.jsonl\",\"out\":\"$X/${ARM}_v0_$s.json\"},"; done
echo "${j%,}]" > $A/jobs_x_$ARM.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $A/jobs_x_$ARM.json --concurrency 384 >> $LOG 2>&1 || log "exploit score FAILED"
free_gpu
log "HUMSYN_DONE $ARM"
