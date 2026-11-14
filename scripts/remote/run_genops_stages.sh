#!/bin/bash
# 生成器 GRPO 的阶段二、阶段三连着跑，每个阶段跑完在同一批留出回答上采样打分，和前面的阶段放在一张表里比。
#   起点 = s2d（阶段一：真错 × 自然 × 操作一致 × critic「不明显」系数，再加判别器项）
#   阶段二 st2：过门槛的样本，critic 给分越低分越高（0.9 以下都算）。阶段一的分和判别器项原样留着
#   阶段三 st3：从 st2 接着训，难度只在 critic 拿不准的区间给分
#   难度项是加法（奖励 = 阶段一的分 + w × 难度），权重上限 WHARD 默认 2.0，从一半起步，三项守得住才放大。
#   权重小的时候总奖励和阶段一的分几乎是同一个东西（0.5 时相关 0.95），训练会继续往 critic 认得出的方向走
#   每个阶段的约束 = 进入该阶段时真错、自然、操作一致三项各自的平均水平，任何一项掉下去难度权重自动调小
#   任务 = 普查过的回答里，8 条至少有一条达标的任务（outputs/rt/gen_ops_grpo/census_tasks.json）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; LOG=$A/stages.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
G=$FS/adapters/gen-ops-grpo
python - <<'EOF' >> $LOG 2>&1 || { log "prompts FAILED"; exit 1; }
import json
keep = {r["id"] for r in json.load(open("outputs/rt/gen_ops_grpo/census_tasks.json")) if r["n_pass"] > 0}
rows = [l for l in open("outputs/rt/gen_ops_grpo/prompts_free.jsonl", encoding="utf-8") if json.loads(l)["id"] in keep]
open("outputs/rt/gen_ops_grpo/prompts_free_c.jsonl", "w", encoding="utf-8").write("".join(rows))
print("PROMPTS_C", len(rows), "of", len(keep))
EOF
refs() {   # 一次训练最后 10 步的样本：阶段一的平均分 + 三项各自的平均概率
  python - "$1" <<'EOF'
import gzip, json, sys, glob, re
fs = sorted(glob.glob(f"outputs/rt/gen_ops_grpo/{sys.argv[1]}/samples/step*.jsonl.gz"), key=lambda f: int(re.search(r"step(\d+)", f).group(1)))[-10:]
xs = [json.loads(l) for f in fs for l in gzip.open(f, "rt", encoding="utf-8")]
m = lambda k: sum((x.get(k) or 0.0) if x.get("why") == "ok" else 0.0 for x in xs) / len(xs)
print("%.3f %.3f,%.3f,%.3f" % (m("r1"), m("p_err"), m("p_natural"), m("p_op")))
EOF
}
ev() {   # ev <运行名>：留出回答上采样 + 打分。判别器用第一版（rm2__d1）：前面几个阶段的留出样本都是它打的分，同一把尺子才能比；训练里用的是第二版
  local e=$1
  [ -s $C/samples_ops_ev_${e}_rm4.jsonl ] && return 0
  pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ev-$e.log sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter disc=$FS/adapters/rm2__d1-mc/adapter g=$G-$e/step100/policy >> $LOG 2>&1 || { log "ev serve $e FAILED"; return 1; }
  python -u -m ragtruth.gen_ops sample --model g --tokenizer $MC --card --style free --n 4 --out samples_ops_ev_$e.jsonl >> $LOG 2>&1 || log "ev sample $e FAILED"
  t() { python -c "import json;print(json.load(open('$R/$1/eval/report.json'))['temperature'])"; }
  python -u $A/score6.py "{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d1)}" ev_$e >> $LOG 2>&1 || log "ev score $e FAILED"
  pkill -f '[v]llm serve'; sleep 5
  log "EV_DONE $e"
}
prev=s2d
for v in "st2|2" "st3|3"; do
  IFS='|' read -r e stage <<< "$v"
  read -r REF REFS <<< "$(refs $prev)"
  log "stage $stage ($e) from $prev: ref score $REF, refs real/natural/op $REFS"
  E=$e STAGE=$stage STYLE=free LR=2e-5 STEPS=100 N=32 PPS=8 DISC=rm2__d2 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-$prev/step100/policy \
    EXTRA="--ops_disc 1 --ops_ref $REF --ops_refs $REFS --ops_w_hard ${WHARD:-2.0}" CODE=$PWD bash $S/run_genops_grpo.sh >> $LOG 2>&1
  [ -d $G-$e/step100/policy ] || { log "$e FAILED"; break; }
  ev $e
  prev=$e
done
python $A/ev_cmp2.py ev_base_free ev_s1c ev_s2d ev_st2 ev_st3 > $A/stages_cmp.txt 2>&1
log "STAGES_DONE"
