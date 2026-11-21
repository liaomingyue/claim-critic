#!/bin/bash
# 第二轮错误合成 GRPO（两台单卡布局的训练机 JOB）：把关模型换成加了 Claude 标签重训的 sev__v5 / rm2__v5 / opm__v2，critic 换成第一轮补过盲区样本的 rm1__core_of9all（盲区按它算）。
#   只跑第一轮的阶段三、四：s3r2 从第一轮终点（s4c9 第 100 步）出发、难度权重 1.5；s4r2 从 s3r2 出发、加明显错系数和组内多样性、权重 3.0。
#   s3r2 的基准线：新把关模型给第一轮终点的留出样本重打分（ev_s4c9 → _rm9v5）的三项平均概率；s4r2 的基准线同第一轮（上一段最后 30 步）。
#   推理机：G_* 同上、INIT = s4c9 第 100 步（run_genops_infer9.sh，配对 gog9r2）；阶段四前把推理机上的 policy_v0 换成 s3r2 终点。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; G=$FS/adapters/gen-ops-grpo; LOG=$A/r2_9.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 PYTHONPATH=. PYTHONIOENCODING=utf-8
export G_SEV=sev__v5 G_NAT=rm2__v5 G_OPM=opm__v2 G_CR=rm1__core_of9all GSUF=rm9v5 SKIP_BASE=1
refs_train() {   # 上一段最后 30 步训练样本：阶段一的平均分 + 三项平均概率（未通过格式检查的记 0）
  python - "$1" <<'P'
import gzip, json, sys, glob, re
fs = sorted(glob.glob(f"outputs/rt/gen_ops_grpo/{sys.argv[1]}/samples/step*.jsonl*"), key=lambda f: int(re.search(r"step(\d+)", f).group(1)))[-30:]
op = lambda f: gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")
xs = [json.loads(l) for f in fs for l in op(f)]
m = lambda k: sum((x.get(k) or 0.0) if x.get("why") == "ok" else 0.0 for x in xs) / len(xs)
print("%.3f %.3f,%.3f,%.3f" % (m("r1"), m("p_err"), m("p_natural"), m("p_op")))
P
}
refs_ev() {   # 新把关模型在留出样本上的三项平均概率（未通过格式检查的记 0）
  python - "$1" <<'P'
import json, sys
C = "outputs/rt/core"; t = sys.argv[1]
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{C}/samples_ops_{t}_rm9v5.jsonl", encoding="utf-8")}
xs = [json.loads(l) for l in open(f"{C}/samples_ops_{t}.jsonl", encoding="utf-8")]
m = lambda k: sum((P[x["id"]][k] or 0.0) if x.get("why") == "ok" and x["id"] in P else 0.0 for x in xs) / len(xs)
print("%.3f,%.3f,%.3f" % (m("p_err"), m("p_nat"), m("p_op")))
P
}
stage() {   # stage <运行名> <起点 adapter> <REF> <REFS> <难度权重> <额外参数>
  log "stage 3 ($1) from $2: ref $3 refs $4 weight $5 extra '$6'"
  E=$1 STAGE=3 N=32 PPS=8 STEPS=100 LR=2e-5 PROMPTS=$O/prompts_free_c.jsonl INIT=$2 RESCORE="" \
    EXTRA="--ops_disc 1 --ops_ref $3 --ops_refs $4 $6 --ops_w_hard $5 --ops_w_fixed 1" bash scripts/remote/run_genops_grpo9.sh
  [ -f $G-$1/step100/policy/adapter_model.safetensors ]
}
swap_v0() {
  rm -rf $A/v0_$2 && cp -r $1 $A/v0_$2 || return 1
  python - "${POLICY_URL:-$RM_URL}" "$A/v0_$2" <<'P'
import sys
from train.gen_grpo import Server
s = Server(sys.argv[1].split(",")[0]); s.unload_lora("policy_v0"); s.load_lora("policy_v0", sys.argv[2]); print("SWAPPED policy_v0", sys.argv[2])
P
}
S4=$G-s4c9/step100/policy
[ -f $S4/adapter_model.safetensors ] || { log "R2 FAILED: no s4c9"; exit 1; }
# ① 新把关模型重打第一轮第三、四段终点的留出样本（本机卡，推理机此时空着）
E=s3r2 ONLY_BASE=1 RESCORE="ev_s3c9 ev_s4c9" bash scripts/remote/run_genops_grpo9.sh || { log "R2 FAILED: rescore"; exit 1; }
[ -s $C/samples_ops_ev_s4c9_rm9v5.jsonl ] || { log "R2 FAILED: no rescored s4c9"; exit 1; }
REFS=$(refs_ev ev_s4c9); read -r REF _ <<< "$(refs_train s4c9)"
log "round-1 endpoint under new gates: refs $REFS (old-gate stage-1 ref $REF)"
stage s3r2 $S4 $REF $REFS 1.5 "" || { log "R2 FAILED: s3r2"; exit 1; }
swap_v0 $G-s3r2/step100/policy s3r2 >> $LOG 2>&1 || { log "R2 FAILED: swap"; exit 1; }
read -r REF REFS <<< "$(refs_train s3r2)"
stage s4r2 $G-s3r2/step100/policy $REF $REFS 3.0 "--ops_clear 0.5 --ops_div 0.8" || { log "R2 FAILED: s4r2"; exit 1; }
log "R2_DONE"
