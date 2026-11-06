#!/bin/bash
# Claude 复核后的对照（2026-11-05，用户：错误生成器的样本不管边界还是随机都要 Claude 复核再进 critic；方案一）。训练机单卡：
#   ① 建数据（基于 rm1__mcgold_ref，只加训练集）：白名单 = Claude 判「原句无误 + 显然错」的 harvest_h3 样本，类型按 Claude mechanism 改正
#        rm1__mcgold_h3v    ：边界组（critic 0.3–0.7）
#        rm1__mcgold_h3ctlv ：随机组（h3ctl 中与 h3 不重合的，critic ≥ 0.7 或 < 0.3），同条数
#   ② 依次训两个 critic（run_mcrm.sh：校准 + RAGTruth++ 报告）
#   ③ 能力评测打分：200 条 Claude 标过的 2B GRPO 回答（grpo300 / grpo300b）+ 回答 GRPO 策略回答（cmp_h2）；critic_cap 在本地算
# 白名单：outputs/rt/sevlabel/ids_h3v.txt、ids_h3ctlv.txt（本地按 Claude 标签生成后上传）。跑完 GPU 空闲，看门狗关机。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/h3v.log; S=$(dirname $(readlink -f $0)); O=outputs/rt/gentarget
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
export MC=$A/models/MiniCPM5-2B
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
NV=rm1__mcgold_h3v; CV=rm1__mcgold_h3ctlv

# ① 建数据
for pair in "$NV ids_h3v" "$CV ids_h3ctlv"; do
  set -- $pair
  python -u -m ragtruth.critic_harvest --harvest $O/harvest_h3.jsonl --base rm1__mcgold_ref --name $1 --gate 0.5 --ids outputs/rt/sevlabel/$2.txt \
    --per_prompt 99 --max_add 99999 >> $LOG 2>&1 || { log "build $1 FAILED"; exit 1; }
  log "built $1: $(python -c "import json;m=json.load(open('outputs/rt/rm/$1/meta.json'));print(m['added_errors'],m['added_by_type'],m['added_bins'])")"
done

# ② 训两个 critic
for r in $NV $CV; do
  rm -rf $FS/runs/claims/mcrm_$r
  RM=$r GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh || log "train $r FAILED"
done
for r in rm1__mcgold rm1__mcgold_ref rm1__mcgold_h3 rm1__mcgold_h3ctl $NV $CV; do log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"; done

# ③ 能力评测打分
free_gpu
mkdir -p $A/s0
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-h3v.log \
  h3v=$FS/adapters/$NV-mc/adapter h3ctlv=$FS/adapters/$CV-mc/adapter >> $LOG 2>&1 || { log "serve critics FAILED"; exit 1; }
for pair in "h3v $NV" "h3ctlv $CV"; do
  set -- $pair; m=$1; r=$2; T=$(temp $r)
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model $m --temp $T --out $d/critic_$r.json >> $LOG 2>&1 || log "score $r $d FAILED"
  done
  for s in old_a old_b new_a new_b; do
    python -u -m ragtruth.critic_score --answers outputs/rt/minicpm/cmp_h2/$s.jsonl --url http://127.0.0.1:8000/v1 --model $m --temp $T --out outputs/rt/minicpm/cmp_h2/${s}_$r.json >> $LOG 2>&1 || log "score $r $s FAILED"
  done
done
free_gpu
log "H3V_DONE"
