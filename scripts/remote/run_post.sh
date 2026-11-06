#!/bin/bash
# 奖励对照领完后的收尾（2026-11-05）。两台都跑，按共享盘占位分工：
#   ① acceptc（用户：Claude 显然错误扩到约 200 句）：2B 回答 GRPO 第 300 步策略在 132 篇没用过的测试原文上各生成 3 条（温度 0.7、种子 0/1/2，同 grpo300b），
#      再用全部 critic 给这批回答逐句打分（不依赖标签；标签由本地 accept_more label_batch 调 Claude 补）→ outputs/rt/minicpm/grpo300c/
#   ② gbl（用户：先做 A）：ref 数据 + group_by_length 重训，比耗时和指标
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/post.log; Q=$FS/runs/scale_queue; MC=$A/models/MiniCPM5-2B; D=outputs/rt/minicpm/grpo300c
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
until grep -q RWQUEUE_DONE $A/rwqueue.log 2>/dev/null; do sleep 30; done

if mkdir $Q/claim_acceptc 2>/dev/null; then
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_acceptc/by
  free_gpu; mkdir -p $A/s0
  A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-acceptc.log g300=$(ad $FS/adapters/mc-grpo/step300) >> $LOG 2>&1 || { log "serve policy FAILED"; exit 1; }
  [ -f $D/picks.jsonl ] || { log "missing $D/picks.jsonl"; exit 1; }
  python -u -m ragtruth.accept_more gen3 --base_url http://127.0.0.1:8000/v1 --model g300 --seeds 3 --concurrency 32 >> $LOG 2>&1 || { log "gen3 FAILED"; exit 1; }
  free_gpu
  # critic 打分：一次挂 8 个 LoRA（vLLM --max-loras 8），两轮
  CR="rm1__mcgold rm1__mcgold_ref rm1__mcgold_h2 rm1__mcgold_h3 rm1__mcgold_h3ctl rm1__mcgold_h3v rm1__mcgold_h3ctlv rm1__mcgold_h3ctlu \
      rm1__mcgold_ref_z1500 rm1__mcgold_ref_z6000 rm1__mcgold_ref_divhi rm1__mcgold_ref_divlo rm1__mcgold_ref_n1500 rm1__mcgold_ref_nohA rm1__mcgold_ref_noh rm1__mcgold_ref_nohs"
  set -- $CR
  while [ $# -gt 0 ]; do
    grp=(); mods=(); k=0
    while [ $# -gt 0 ] && [ $k -lt 8 ]; do grp+=($1); mods+=("c$k=$FS/adapters/$1-mc/adapter"); k=$((k+1)); shift; done
    free_gpu
    A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-acceptc.log "${mods[@]}" >> $LOG 2>&1 || { log "serve critics FAILED"; continue; }
    for k in "${!grp[@]}"; do
      r=${grp[$k]}; f=$D/critic_$r.json; [ $r = rm1__mcgold ] && f=$D/critic2b_scores.json
      [ -f $f ] || python -u -m ragtruth.critic_score --answers $D/answers.jsonl --url http://127.0.0.1:8000/v1 --model c$k --temp $(temp $r) --out $f >> $LOG 2>&1 || log "score $r FAILED"
    done
  done
  free_gpu
  log "ACCEPTC_DONE $(ls $D | grep -c critic)"
fi

if mkdir $Q/claim_gbl 2>/dev/null; then
  echo "$(hostname) $(date '+%F %T')" > $Q/claim_gbl/by
  R=rm1__mcgold_ref_gbl
  mkdir -p outputs/rt/rm/$R && cp outputs/rt/rm/rm1__mcgold_ref/{train,calib,test}.jsonl outputs/rt/rm/rm1__mcgold_ref/meta.json outputs/rt/rm/$R/
  rm -rf $FS/runs/claims/mcrm_$R; free_gpu
  t0=$(date +%s)
  GBL=1 RM=$R GPU=0 PORT=8100 MC=$MC bash scripts/remote/run_mcrm.sh || { log "train $R FAILED"; exit 1; }
  log "$R train+eval $(( ($(date +%s) - t0) / 60 )) min; runtime $(grep -ao "'train_runtime': '[0-9.]*'" $A/$R-mc.log | tail -n 1)"
  R=$R bash scripts/remote/score_one.sh
  log "GBL_DONE"
fi
log "POST_DONE"
