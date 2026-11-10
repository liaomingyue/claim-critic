#!/bin/bash
# 数量 / 多样性 / 自然度对照（接替 run_scale.sh 的数量曲线部分）。发现：RM2 自然度 ≥ 0.3 后 harvest_h3 只剩 2413 条（605 篇原文），
# n3000 / n6000 凑不够 → 数量曲线改成不加自然度门槛（13071 条、724 篇原文、4938 个 prompt）。另加多样性实验，排在 n6000 前面；再开一张卡。
# 底座 rm1__mcgold_ref（底座对照：删人工 / 早期合成都不更好）。门槛同 h3ctl：sev P(显然) ≥ 0.5、原句 P(F) ≤ 0.5、每 prompt ≤ 2、seed 0。
#   nat1500  RM2 ≥ 0.3，1500 对（自然度门槛那一点；与 z1500 比 = 自然度过滤有没有用）
#   z1500 / z3000 / z6000  不加自然度门槛，嵌套（数量曲线）
#   divhi    不加自然度门槛，1400 对，每篇原文 ≤ 2（覆盖约 700 篇）
#   divlo    不加自然度门槛，1400 对，只用合格样本最多的 100 篇原文（多样性低）
# TASKS = 这台跑的任务（空格分隔）。每个：建数据 → run_mcrm.sh 训练（校准 + RAGTruth++）→ grpo300 / grpo300b / cmp_h2 打分；结束写 SCALE_DONE（run_rwqueue 等它）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/scale.log; S=$(dirname $(readlink -f $0)); O=outputs/rt/gentarget
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
export MC=$A/models/MiniCPM5-2B
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
H=$O/harvest_h3.jsonl; BASE=rm1__mcgold_ref

score() {   # $1 = 数据目录名（已训练）
  local r=$1
  free_gpu; mkdir -p $A/s0
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-scale.log c=$FS/adapters/$r-mc/adapter >> $LOG 2>&1 || { log "serve $r FAILED"; return 1; }
  local T=$(temp $r)
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$r.json >> $LOG 2>&1 || log "score $r $d FAILED"
  done
  for s in old_a old_b new_a new_b; do
    python -u -m ragtruth.critic_score --answers outputs/rt/minicpm/cmp_h2/$s.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out outputs/rt/minicpm/cmp_h2/${s}_$r.json >> $LOG 2>&1 || log "score $r $s FAILED"
  done
  free_gpu
}

train_score() {   # $1 = 数据目录名（已建好）
  local r=$1
  rm -rf $FS/runs/claims/mcrm_$r
  free_gpu
  RM=$r GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh || { log "train $r FAILED"; return 1; }
  log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"
  score $r
  log "SCALE_ONE_DONE $r"
}

# 接手旧脚本时，正在训的那个已经训完但没打分：SCORE_FIRST 给它补打
for r in ${SCORE_FIRST:-}; do score $r && log "SCALE_ONE_DONE $r (scored by run_scale2)"; done

G="--harvest $H --base $BASE --gate 0.5 --lo 0.0 --hi 1.01"
# 共享队列（跑完后有空闲就把其他排队的放到空闲机器上跑）：队列文件里有任务就按顺序领（mkdir 原子占位），
# 哪台先空哪台领；没有队列文件才用 TASKS。领完写 SCALE_DONE，接着 run_rwqueue 领奖励对照。
QD=$FS/runs/scale_queue
[ -f $QD/tasks ] && LIST=$(cat $QD/tasks) || LIST=${TASKS:?}
for t in $LIST; do
  if [ -f $QD/tasks ]; then
    mkdir $QD/claim_$t 2>/dev/null || { log "$t taken"; continue; }
    echo "$(hostname) $(date '+%F %T')" > $QD/claim_$t/by
  fi
  case $t in
    nat1500) r=${BASE}_n1500; args="--min_nat 0.3 --max_add 1500" ;;
    z1500|z3000|z6000) r=${BASE}_$t; args="--min_nat 0 --max_add ${t#z}" ;;
    divhi) r=${BASE}_divhi; args="--min_nat 0 --per_source 2 --max_add 1400" ;;
    divlo) r=${BASE}_divlo; args="--min_nat 0 --top_sources 100 --max_add 1400" ;;
    *) log "unknown task $t"; continue ;;
  esac
  python -u -m ragtruth.critic_harvest $G --name $r $args >> $LOG 2>&1 || { log "build $r FAILED"; continue; }
  log "built $r: $(python -c "import json;m=json.load(open('outputs/rt/rm/$r/meta.json'));print(m['added_errors'],'sources',m['sources'],m['added_by_type'],m['added_bins'],m['train']['n'])")"
  train_score $r
done
log "SCALE_DONE"
