#!/bin/bash
# G_free 之上做 DPO（技术报告 Q4：先学真实分布，再往「难 + 真」推）。和闭环验证并行，复用闭环验证 gfree 组的采样。
# 1. 等 closed/gfree.valid.done → C2 打难度、D1（修好 logprob 解析后）打真实性 → 帕累托偏好对（统一 prompt）
# 2. DPO：起点 G_free，学习率 5e-5（GAN 两轮用 5e-6，对 LoRA 偏低、几乎没学到东西）
# 3. 评测：G_free+DPO 在 realgen 评测的同一批回答上采样 4 条 → 35B 判有效 → D1 / D2 真实性 → C2 难度；
#    G_free 的旧评测用修好的 D1、以及 D2 重新打真实性，两者同口径对比。D1 同时当了 DPO 的奖励，所以 D2 是更干净的口径。
# 用法：bash run_freedpo.sh（工作目录 = 共享代码目录，CODE 传入）。日志 /root/autodl-tmp/freedpo.log，结束写 FREEDPO_DONE。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
LOG=/root/autodl-tmp/freedpo.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; B9=$M/Qwen3.5-9B
C=outputs/rt/closed/gfree; O=outputs/rt/freedpo; E1=outputs/rt/realgen/eval_gfree; E2=$O/eval_gfreedpo
MM='{"image":0,"video":0}'
G="python -m ragtruth.gen_dpo"
mkdir -p $O
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
serve() {
  stop; fix_tok "$1"
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len ${3:-8192} --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; exit 1
}
done_mark() { [ -f $O/$1.done ]; }
mark() { touch $O/$1.done; log "$1 done"; }
step() { done_mark $1 || { "${@:2}" >> $LOG 2>&1 && mark $1; } || { log "$1 FAILED"; exit 1; }; }

log "waiting for outputs/rt/closed/gfree.valid.done"
until [ -f outputs/rt/closed/gfree.valid.done ]; do sleep 60; done

serve $CK/critic-gan-r2 critic
step train.hard $G hard --dir $C --model critic --tag c2
serve $CK/discrim-9b discrim
step train.d1 $G realism --dir $C --model discrim --tag d1fix
stop
step pairs $G pairs --dir $C --mode hard_real --canonical --tag c2 --rtag d1fix --margin 0.3
if ! done_mark dpo; then
  sed -e "s|^base_model: .*|base_model: $CK/gen-sft-free|" -e "s|^data: .*|data: $C/pairs_hard_real_canon.jsonl|" \
    -e "s|^output_dir: .*|output_dir: $CK/gen-free-dpo|" -e "s|^  learning_rate: .*|  learning_rate: 5.0e-5|" \
    configs/rt/dpo_gen_9b.yaml > configs/rt/exp/dpo_gen_free.yaml
  log "DPO from G_free"
  python -u -m train.dpo --config configs/rt/exp/dpo_gen_free.yaml > /root/autodl-tmp/dpo-gen-free.log 2>&1 \
    || { log "DPO FAILED"; exit 1; }
  mark dpo
fi

serve $CK/gen-free-dpo gen 16384
step eval.sample $G sample --dir $E2 --free --resp_from outputs/rt/realgen/eval_gsft/samples.jsonl --k 4 --variants 1
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
step eval.valid $G valid --dir $E2 --model judge
serve $CK/discrim-9b discrim
step eval.d1.new $G realism --dir $E2 --model discrim --tag d1fix
step eval.d1.old $G realism --dir $E1 --model discrim --tag d1fix
serve $CK/discrim-9b-r2 discrim
step eval.d2.new $G realism --dir $E2 --model discrim --tag d2
step eval.d2.old $G realism --dir $E1 --model discrim --tag d2
serve $CK/critic-gan-r2 critic
step eval.c2 $G hard --dir $E2 --model critic --tag c2
stop

python - >> $LOG 2>&1 <<'EOF'
import json
from pathlib import Path
def kept(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []
out = {}
for name, d in (("G_free", "outputs/rt/realgen/eval_gfree"), ("G_free+DPO", "outputs/rt/freedpo/eval_gfreedpo")):
    cands = [c for r in kept(f"{d}/valid.jsonl") for c in r["cands"]]
    v = sum(c["valid"] for c in cands)
    row = {"candidates": len(cands), "valid_rate": round(v / max(1, len(cands)), 3)}
    for tag in ("d1fix", "d2"):
        ps = [r["p_natural"] for r in kept(f"{d}/realism_{tag}.jsonl")]
        row[f"{tag}_mean_p_natural"] = round(sum(ps) / max(1, len(ps)), 3)
        row[f"{tag}_frac_natural"] = round(sum(p > 0.5 for p in ps) / max(1, len(ps)), 3)
    hs = [r["detect_rate"] for r in kept(f"{d}/hard_c2.jsonl")]
    row["hard_among_valid"] = round(sum(h < 0.5 for h in hs) / max(1, len(hs)), 3)
    row["mean_detect"] = round(sum(hs) / max(1, len(hs)), 3)
    out[name] = row
Path("outputs/rt/freedpo/summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, ensure_ascii=False))
EOF
log "FREEDPO_DONE"
