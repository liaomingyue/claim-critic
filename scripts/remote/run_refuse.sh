#!/bin/bash
# 拒答标签修正 + 证据段探针（2026-11-04，用户：核心是确认 2B 为什么识别不出明显错误；可以）：
#   GPU0：建 rm1__mcgold_ref（rm1__mcgold + 训练集 10 句能回答的拒答 A → B + 513 条拒答样本，按拒答 RM 的 Claude 标注）→ 训练、校准、RAGTruth++ 报告
#         → 新 critic 打拒答评测（test_refuse）和 200 条 2B GRPO 验收回答（grpo300 / grpo300b）
#   GPU1：基线 2B critic（rm1__mcgold）打证据段探针（probe_ev：完整原文 / 只给证据段）和 test_refuse；再换 9B 金标 critic 打同样两份
# 机器跑完不关（后面接着看结果）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/refuse.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
R=rm1__mcgold_ref; O=outputs/rt/rm/$R; PR=outputs/rt/minicpm/grpo300all/probe_ev.jsonl; PO=outputs/rt/minicpm/grpo300all
mkdir -p $A/s0 $A/s1
python -u -m ragtruth.critic_refuse >> $LOG 2>&1 || { log "build FAILED"; exit 1; }
log "built $R"
RM=$R GPU=0 PORT=8100 bash $S/run_mcrm.sh & T0=$!
( A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8001 --base $MC --log $A/vllm-rfa.log rm1=$FS/adapters/rm1__mcgold-mc/adapter >> $LOG 2>&1 || exit 1
  python -u -m ragtruth.rm3_v2 score --inp $PR --url http://127.0.0.1:8001/v1 --model rm1 --opts ABC --temp $(temp rm1__mcgold) --out $PO/probe_2b.jsonl >> $LOG 2>&1 || exit 1
  python -u -m ragtruth.rm3_v2 score --inp $O/test_refuse.jsonl --url http://127.0.0.1:8001/v1 --model rm1 --opts ABC --temp $(temp rm1__mcgold) --out $O/refuse_2b.jsonl >> $LOG 2>&1 || exit 1
  pkill -f '[v]llm serve.*8001'; sleep 15
  A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8001 --base $FS/models/Qwen3.5-9B --log $A/vllm-rfb.log rm1=$FS/adapters/rm1__rm1_gold_9b-9b/adapter >> $LOG 2>&1 || exit 1
  python -u -m ragtruth.rm3_v2 score --inp $PR --url http://127.0.0.1:8001/v1 --model rm1 --opts ABC --temp $(temp rm1__rm1_gold_9b) --out $PO/probe_9b.jsonl >> $LOG 2>&1 || exit 1
  python -u -m ragtruth.rm3_v2 score --inp $O/test_refuse.jsonl --url http://127.0.0.1:8001/v1 --model rm1 --opts ABC --temp $(temp rm1__rm1_gold_9b) --out $O/refuse_9b.jsonl >> $LOG 2>&1 || exit 1
  pkill -f '[v]llm serve.*8001' ) & T1=$!
wait $T1 && log "probe + refuse scored (2b, 9b)" || log "gpu1 scoring FAILED"
wait $T0
log "$R report $(python -c "import json;r=json.load(open('$O/eval/report.json'));t=r['test_calibrated'];print(json.dumps({'T':r['temperature'],**{k:t.get(k) for k in ('n','auc','recall@0.5','precision@0.5','ece')}}))" 2>&1)"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-rfc.log critic=$FS/adapters/$R-mc/adapter >> $LOG 2>&1 || { log "serve new FAILED"; exit 1; }
python -u -m ragtruth.rm3_v2 score --inp $O/test_refuse.jsonl --url http://127.0.0.1:8000/v1 --model critic --opts ABC --temp $(temp $R) --out $O/refuse_new.jsonl >> $LOG 2>&1 || log "score refuse new FAILED"
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b; do
  python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --temp $(temp $R) --out $d/critic_$R.json >> $LOG 2>&1 || log "score $d FAILED"
done
pkill -f '[v]llm serve'
log "REFUSE_DONE"
