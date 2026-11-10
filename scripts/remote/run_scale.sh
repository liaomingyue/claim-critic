#!/bin/bash
# 把结论做实（大概率不需要 Claude 重新标注，可能是样本数量要足够多、不是非要边界样本；
# RAGTruth 原版人工漏标多，最好不要了；早期合成只抽查过 → 三组底座对照，插到数量曲线前面）。训练机单卡，接在 run_h3v.sh 后面：
#   ⓪ 底座对照（在 ref 上只删来源，不加生成器数据）：
#        rm1__mcgold_ref_nohA  删人工里标 A 的 1755（保留人工 B / C 762）
#        rm1__mcgold_ref_noh   人工 2517 全删
#        rm1__mcgold_ref_nohs  人工 + 早期合成（907）都删
#   ① 复核要不要：rm1__mcgold_h3ctlu = 与 h3ctlv 同池（h3ctl 中与 h3 不重合的 427 条）随机 265、不复核、类型用生成器的（Claude 合格率 60.7%），底座 ref
#   ② 数量曲线：等本地按 Claude 200 条选定底座（写 outputs/rt/rm/scale_base.txt；GPU 空着最多等 8 分钟，免得看门狗关机——
#        超时就用 ref），不限 critic 分数、不复核（同 h3ctl 门槛：sev ≥ 0.5、原句 P(F) ≤ 0.5、RM2 ≥ 0.3、每 prompt ≤ 2、seed 0），
#        <底座>_n1500 / n3000 / n6000，嵌套
#   每个：run_mcrm.sh 训练（校准 + RAGTruth++）→ grpo300 / grpo300b / cmp_h2 打分（critic_cap 和 bootstrap 在本地算）
# 跑完 GPU 空闲，看门狗关机。
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
H=$O/harvest_h3.jsonl

train_score() {   # $1 = 数据目录名（已建好）
  local r=$1
  rm -rf $FS/runs/claims/mcrm_$r
  free_gpu
  RM=$r GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh || { log "train $r FAILED"; return 1; }
  log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"
  free_gpu
  mkdir -p $A/s0
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-scale.log c=$FS/adapters/$r-mc/adapter >> $LOG 2>&1 || { log "serve $r FAILED"; return 1; }
  local T=$(temp $r)
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$r.json >> $LOG 2>&1 || log "score $r $d FAILED"
  done
  for s in old_a old_b new_a new_b; do
    python -u -m ragtruth.critic_score --answers outputs/rt/minicpm/cmp_h2/$s.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out outputs/rt/minicpm/cmp_h2/${s}_$r.json >> $LOG 2>&1 || log "score $r $s FAILED"
  done
  free_gpu
  log "SCALE_ONE_DONE $r"
}

# 两台单卡分工：ARMS = 这台跑的底座对照（nohA / noh / nohs）、DO_U = 跑不跑 h3ctlu、NS = 这台跑的数量点
ARMS=${ARMS:-"nohA noh nohs"}; DO_U=${DO_U:-1}
if [ "${WAIT_H3V:-1}" = 1 ]; then log "waiting for H3V_DONE"; until grep -q H3V_DONE $A/h3v.log 2>/dev/null; do sleep 30; done; fi

# ⓪ 底座对照
for arm in $ARMS; do
  case $arm in nohA) set -- rm1__mcgold_ref_nohA human:A ;; noh) set -- rm1__mcgold_ref_noh human ;; nohs) set -- rm1__mcgold_ref_nohs human synth ;; *) continue ;; esac
  r=$1; shift
  python -u -m ragtruth.critic_subset --base rm1__mcgold_ref --name $r --drop "$@" >> $LOG 2>&1 || { log "build $r FAILED"; continue; }
  log "built $r: $(python -c "import json;m=json.load(open('outputs/rt/rm/$r/meta.json'));print(m['dropped'],m['train'])")"
  train_score $r
done

# ① 复核要不要
r=rm1__mcgold_h3ctlu
[ "$DO_U" = 1 ] && python -u -m ragtruth.critic_harvest --harvest $H --base rm1__mcgold_ref --name $r --gate 0.5 --ids outputs/rt/sevlabel/ids_h3ctlu.txt --per_prompt 99 --max_add 99999 >> $LOG 2>&1 \
  && log "built $r: $(python -c "import json;m=json.load(open('outputs/rt/rm/$r/meta.json'));print(m['added_errors'],m['added_by_type'],m['added_bins'])")" && train_score $r || [ "$DO_U" != 1 ] || log "h3ctlu FAILED"

# ② 数量曲线
python - $H <<'EOF' 2>&1 | tee -a $LOG
import json, sys, collections
q = [x for x in map(json.loads, open(sys.argv[1], encoding="utf-8")) if x.get("p_halluc") is not None and x["p_correct"] >= 0.5 and x["orig_p"] <= 0.5 and (x.get("p_natural") or 0) >= 0.3]
per = collections.Counter(x["prompt_id"] for x in q)
print("SCALE_POOL", {"qualified_nat0.3": len(q), "prompts": len(per), "cap_per_prompt2": sum(min(2, v) for v in per.values())}, flush=True)
EOF
log "waiting for outputs/rt/rm/scale_base.txt (max 8 min)"
for i in $(seq 1 ${BASE_WAIT:-48}); do [ -s outputs/rt/rm/scale_base.txt ] && break; sleep 10; done
BASE=$(cat outputs/rt/rm/scale_base.txt 2>/dev/null || echo rm1__mcgold_ref); [ -f outputs/rt/rm/$BASE/train.jsonl ] || BASE=rm1__mcgold_ref
log "scale base: $BASE"
for n in ${NS:-1500 3000 6000}; do
  r=${BASE}_n$n
  python -u -m ragtruth.critic_harvest --harvest $H --base $BASE --name $r --gate 0.5 --lo 0.0 --hi 1.01 --min_nat 0.3 --max_add $n >> $LOG 2>&1 || { log "build $r FAILED"; continue; }
  log "built $r: $(python -c "import json;m=json.load(open('outputs/rt/rm/$r/meta.json'));print(m['added_errors'],m['added_by_type'],m['added_bins'],m['train']['n'])")"
  train_score $r
done
log "SCALE_DONE"
