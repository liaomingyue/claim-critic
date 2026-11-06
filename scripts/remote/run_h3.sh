#!/bin/bash
# 循环第 2–3 步（2026-11-05，用户：把要跑的实验都跑下）。训练机单卡，等区间奖励的生成器训完（ckpt/step100）后依次：
#   ① 放大评测：v0 / 第 50 步 / 第 100 步，评测 prompt × 16，口径同训练（sev 门槛 0.5、critic = rm1__mcgold_ref、rm2v2__mcgen3）
#   ② 造数据：第 100 步生成器在训练 prompt 上每个采 8 条，sev 门槛（不用 RM3、不用 9B），存全部打分 → harvest_h3.jsonl
#   ③ 建 critic 数据（基于 rm1__mcgold_ref，只加训练集）：
#        rm1__mcgold_h3    ：P(显然) ≥ 0.5、sev P(F) ≤ 0.5、RM2 P(自然) ≥ MIN_NAT、critic P 0.3–0.7（边界）+ 配对原句
#        rm1__mcgold_h3ctl ：同条数、同门槛，critic P 不限——区分「加了显然错误」和「加的是边界样本」
#   ④ 依次训两个 critic（run_mcrm.sh：校准 + RAGTruth++ 报告）
#   ⑤ 能力评测（用户：主要看这两个能力）：基线 / ref / h2 / h3 / h3ctl 给 200 条 Claude 标过的 2B GRPO 回答、回答 GRPO 策略自己的回答（cmp_h2，9B 只当评测标签）打分
#      → ragtruth.critic_cap：显然错误 AUC、误报 5% / 10% 下的召回、细节多的正确句子误报
# 基座用本地盘副本（共享盘加载太慢，GPU 空着超过 10 分钟会被看门狗关机）。跑完 GPU 空闲，看门狗关机。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/h3.log; S=$(dirname $(readlink -f $0)); O=outputs/rt/gentarget
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
export MC=$A/models/MiniCPM5-2B
R=$O/grpo_seed5_sev; MIN_NAT=${MIN_NAT:-0.3}; N3=rm1__mcgold_h3; C3=rm1__mcgold_h3ctl
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }

log "waiting for $R/ckpt/step100"
until [ -f $(ad $R/ckpt/step100)/adapter_model.safetensors ] && ! pgrep -f '[t]rain.gen_grpo' > /dev/null; do sleep 30; done; sleep 20
G100=$(ad $R/ckpt/step100); G50=$(ad $R/ckpt/step50)
log "generator step100: $G100"
mkdir -p $A/s0
free_gpu
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-h3gen.log \
  gen=$G100 rm1=$FS/adapters/rm1__mcgold_ref-mc/adapter rm2=$FS/adapters/rm2v2__mcgen3-mc/adapter sev=$FS/adapters/sev__mc-mc/adapter policy_v0=$FS/adapters/gen-seed5-mc/adapter >> $LOG 2>&1 \
  || { log "serve gen FAILED"; exit 1; }

# ① 放大评测
BE=$O/bigeval_sev; mkdir -p $BE
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --prompts $O/prompts.jsonl --eval_prompts $O/eval_prompts.jsonl --reward target \
  --vllm_url http://127.0.0.1:8000/v1 --temps rm1=rm1__mcgold_ref,rm2=rm2v2__mcgen3,sev=sev__mc --shape sev --gate 0.5 --err_mode ab \
  --eval_k 16 --eval_minutes 0 --version_prefix big_ --out $BE --eval_ckpts v0=v0 s50=$G50 s100=$G100 >> $A/h3_bigeval.log 2>&1 || log "bigeval FAILED"
python - $BE <<'EOF' 2>&1 | tee -a $LOG
import json, sys, collections
d = sys.argv[1]
for k, nm in enumerate(("v0", "s50", "s100")):
    try:
        rows = [json.loads(l) for l in open(f"{d}/quick_samples_big_s{k}.jsonl", encoding="utf-8")]
    except OSError:
        continue
    ss = [s for r in rows for s in r["scores"]]
    q = [s for s in ss if s.get("p_halluc") is not None and s["p_correct"] >= 0.5 and s.get("p_orig_halluc", 0) <= 0.5]
    band = [s for s in q if 0.3 <= s.get("p_halluc", 1) <= 0.7]
    low = [s for s in q if s.get("p_halluc", 1) < 0.3]
    print("BIGEVAL", nm, json.dumps({"samples": len(ss), "qualified": round(len(q) / max(1, len(ss)), 4), "band_0.3-0.7": round(len(band) / max(1, len(ss)), 4),
          "low_<0.3": round(len(low) / max(1, len(ss)), 4), "band_n": len(band), "band_prompts": len({r["id"] for r in rows for s in r["scores"] if s.get("qualified") and 0.3 <= s.get("p_halluc", 1) <= 0.7}),
          "band_natural>0.5": round(sum(s.get("p_natural", 0) > 0.5 for s in band) / max(1, len(band)), 3),
          "band_tactic": dict(collections.Counter(s.get("tactic") for s in band).most_common(8))}, ensure_ascii=False), flush=True)
EOF

# ② 造数据
if [ ! -f $O/harvest_h3.jsonl ]; then
  log "harvest h3 ($G100, 8 per prompt, sev gate)"
  python -u -m ragtruth.gen_target_data harvest --base $MC --url http://127.0.0.1:8000/v1 --model gen --inp prompts.jsonl --samples 8 \
    --rm1 rm1 --rm1_temp $(temp rm1__mcgold_ref) --rm2 rm2 --rm2_temp $(temp rm2v2__mcgen3) --sev sev --sev_temp $(temp sev__mc) --gate 0.5 --out harvest_h3.jsonl >> $LOG 2>&1 \
    || { log "harvest FAILED"; exit 1; }
fi
free_gpu
python - $O/harvest_h3.jsonl <<'EOF' 2>&1 | tee -a $LOG
import json, sys
q = [x for x in map(json.loads, open(sys.argv[1], encoding="utf-8")) if x.get("p_halluc") is not None]
for nat in (0.0, 0.2, 0.3, 0.5):
    b = [x for x in q if (x.get("p_natural") or 0) >= nat]
    print("H3_POOL", f"nat>={nat}", {"qualified": len(b), "band_0.3-0.7": sum(0.3 <= x["p_halluc"] < 0.7 for x in b), "low_<0.3": sum(x["p_halluc"] < 0.3 for x in b)}, flush=True)
EOF

# ③ 建数据：边界样本不够 200 条就放宽自然度门槛
NB=$(python -c "
import json
q=[x for x in map(json.loads, open('$O/harvest_h3.jsonl', encoding='utf-8')) if x.get('p_halluc') is not None and x['p_correct']>=0.5 and x['orig_p']<=0.5]
print(sum(0.3<=x['p_halluc']<0.7 and (x.get('p_natural') or 0)>=$MIN_NAT for x in q))")
[ "$NB" -lt 200 ] && { log "only $NB band samples at nat>=$MIN_NAT, using nat>=0"; MIN_NAT=0; }
python -u -m ragtruth.critic_harvest --harvest $O/harvest_h3.jsonl --base rm1__mcgold_ref --name $N3 --lo 0.3 --hi 0.7 --gate 0.5 --min_nat $MIN_NAT --max_add 1500 >> $LOG 2>&1 || { log "build h3 FAILED"; exit 1; }
N=$(python -c "import json;print(json.load(open('outputs/rt/rm/$N3/meta.json'))['added_errors'])")
python -u -m ragtruth.critic_harvest --harvest $O/harvest_h3.jsonl --base rm1__mcgold_ref --name $C3 --lo 0.0 --hi 1.01 --gate 0.5 --min_nat $MIN_NAT --max_add $N >> $LOG 2>&1 || { log "build h3ctl FAILED"; exit 1; }
log "built $N3 / $C3: $N errors each (min_nat $MIN_NAT); bins $(python -c "import json;print(json.load(open('outputs/rt/rm/$C3/meta.json'))['added_bins'])")"

# ④ 训两个 critic
for r in $N3 $C3; do
  rm -rf $FS/runs/claims/mcrm_$r
  RM=$r GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh || log "train $r FAILED"
done
for r in rm1__mcgold rm1__mcgold_ref rm1__mcgold_h2 $N3 $C3; do log "$r report $(python -c "import json;r=json.load(open('outputs/rt/rm/$r/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"; done

# ⑤ 能力评测
free_gpu
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-h3cap.log \
  c0=$FS/adapters/rm1__mcgold-mc/adapter ref=$FS/adapters/rm1__mcgold_ref-mc/adapter h2=$FS/adapters/rm1__mcgold_h2-mc/adapter \
  h3=$FS/adapters/$N3-mc/adapter h3ctl=$FS/adapters/$C3-mc/adapter >> $LOG 2>&1 || { log "serve critics FAILED"; exit 1; }
for pair in "c0 rm1__mcgold" "ref rm1__mcgold_ref" "h2 rm1__mcgold_h2" "h3 $N3" "h3ctl $C3"; do
  set -- $pair; m=$1; r=$2; T=$(temp $r)
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
    f=$d/critic_$r.json; [ $r = rm1__mcgold ] && f=$d/critic2b_scores.json
    [ -f $f ] || python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model $m --temp $T --out $f >> $LOG 2>&1 || log "score $r $d FAILED"
  done
  for s in old_a old_b new_a new_b; do
    f=outputs/rt/minicpm/cmp_h2/${s}_$r.json
    [ -f $f ] || python -u -m ragtruth.critic_score --answers outputs/rt/minicpm/cmp_h2/$s.jsonl --url http://127.0.0.1:8000/v1 --model $m --temp $T --out $f >> $LOG 2>&1 || log "score $r $s FAILED"
  done
done
free_gpu
python -u -m ragtruth.critic_cap --critics rm1__mcgold rm1__mcgold_ref rm1__mcgold_h2 $N3 $C3 --out outputs/rt/critic_cap_h3.json 2>&1 | tee -a $LOG
log "H3_DONE"
