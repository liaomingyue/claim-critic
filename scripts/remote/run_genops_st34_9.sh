#!/bin/bash
# 9B 错误合成 GRPO 阶段三 + 阶段三再训（多样性），两台单卡布局的训练机 JOB（run_train_node.sh 调用），设置同 2B 的 st3 → st4：
#   阶段三（s3c9）：从第二段（s2c9b）第 100 步出发，难度只在 critic 拿不准的区间给分（0.1–0.9 三角带，0.3–0.5 最高），权重固定 1.5；
#   阶段四（s4c9）：从 s3c9 第 100 步出发，难度分乘明显错系数（--ops_clear 0.5），同一任务里同一种改法分摊奖励（--ops_div 0.8，组内多样性），权重 3.0。
#   两段都有判别器项；基准线 = 上一段最后 30 步的平均分和真错 / 自然 / 操作一致三项的平均概率。每段训完在留出回答上采样打分。
#   推理机起点 = 第二段终点（DISC=1 INIT=<s2c9b step100> run_genops_infer9.sh）；阶段四开始前把推理机上的 policy_v0 换成阶段三终点（第 0 版的采样用它）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; O=outputs/rt/gen_ops_grpo; G=$FS/adapters/gen-ops-grpo; LOG=$A/st34_9.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 PYTHONPATH=. PYTHONIOENCODING=utf-8
refs() {   # 上一段最后 30 步的样本：阶段一的平均分 + 三项各自的平均概率
  python - "$1" <<'P'
import gzip, json, sys, glob, re
fs = sorted(glob.glob(f"outputs/rt/gen_ops_grpo/{sys.argv[1]}/samples/step*.jsonl*"), key=lambda f: int(re.search(r"step(\d+)", f).group(1)))[-30:]
op = lambda f: gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")
xs = [json.loads(l) for f in fs for l in op(f)]
m = lambda k: sum((x.get(k) or 0.0) if x.get("why") == "ok" else 0.0 for x in xs) / len(xs)
print("%.3f %.3f,%.3f,%.3f" % (m("r1"), m("p_err"), m("p_natural"), m("p_op")))
P
}
stage() {   # stage <运行名> <起点运行名> <难度权重> <额外参数>
  local e=$1 prev=$2 w=$3 x=$4
  [ -f $G-$prev/step100/policy/adapter_model.safetensors ] || { log "ST34 FAILED: no start $prev"; return 1; }
  read -r REF REFS <<< "$(refs $prev)"
  log "stage 3 ($e) from $prev: ref $REF refs $REFS weight $w extra '$x'"
  E=$e STAGE=3 N=32 PPS=8 STEPS=100 LR=2e-5 PROMPTS=$O/prompts_free_c.jsonl INIT=$G-$prev/step100/policy \
    EXTRA="--ops_disc 1 --ops_ref $REF --ops_refs $REFS $x --ops_w_hard $w --ops_w_fixed 1" bash scripts/remote/run_genops_grpo9.sh
}
swap_v0() {   # 推理机上的 policy_v0 换成 <起点>（经 LoRA 同步服务；先拷到本地盘，加载时会就地去掉视觉塔参数）
  rm -rf $A/v0_$2 && cp -r $1 $A/v0_$2 || return 1
  python - "${POLICY_URL:-$RM_URL}" "$A/v0_$2" <<'P'
import sys
from train.gen_grpo import Server
s = Server(sys.argv[1].split(",")[0]); s.unload_lora("policy_v0"); s.load_lora("policy_v0", sys.argv[2]); print("SWAPPED policy_v0", sys.argv[2])
P
}
stage s3c9 s2c9b 1.5 "" || { log "ST34 FAILED: s3c9"; exit 1; }
swap_v0 $G-s3c9/step100/policy s3c9 >> $LOG 2>&1 || { log "ST34 FAILED: swap"; exit 1; }
stage s4c9 s3c9 3.0 "--ops_clear 0.5 --ops_div 0.8" || { log "ST34 FAILED: s4c9"; exit 1; }
log "ST34_DONE"
