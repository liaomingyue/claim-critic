#!/bin/bash
# 训练中途评一版（训练到 150 步时）。在第三台单卡上跑，不占训练机 / 推理机：
#   ① 9B 金标 critic 补打 grpo_h2 已有的 dev 快评样本（quick_samples_m_s*.jsonl，和上一版的 shadow9b 同口径、按原文配对）
#   ② 新旧两版第 STEP 步检查点在 200 个验收原文-种子上重新生成（同 run_grpo_h2.sh 的设置），9B 逐句打分
# 结果：outputs/rt/minicpm/grpo_h2/run/shadow9b*.jsonl、outputs/rt/minicpm/cmp_h2/s$STEP/。机器跑完不关。
set -u
cd /root/autodl-fs/code/critic_model
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/eval_mid.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
unset VLLM_API_KEY LORA_SYNC_URL
STEP=${STEP:-150}; O=outputs/rt/minicpm/grpo_h2; C=outputs/rt/minicpm/cmp_h2/s$STEP; mkdir -p $C
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
NEW=$(ad $O/run/ckpt/step$STEP); OLD=$(ad $FS/adapters/mc-grpo/step$STEP)
CR=rm1__rm1_gold_9b; T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$CR/eval/report.json'))['temperature'])")
# 先用 9B 补打已有的 dev 快评样本，不干等检查点（9B 可以先启动）
if [ ! -f $NEW/adapter_model.safetensors ]; then
  pkill -f '[v]llm serve'; sleep 5
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-mid9b0.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  python -u -m ragtruth.shadow_rescore --run $O/run --dev $O/dev_ref.jsonl --url http://127.0.0.1:8000/v1 --temp $T >> $LOG 2>&1 || log "early shadow FAILED"
  log "shadow early: $(tail -1 $O/run/shadow9b.jsonl 2>/dev/null | cut -c1-300)"
fi
# 每次重新判断路径：检查点存好之前 ad() 拿不到 policy/ 子目录（踩过：路径在开头定死，一直等不到）
until NEW=$(ad $O/run/ckpt/step$STEP); [ -f $NEW/adapter_model.safetensors ]; do sleep 30; done; sleep 20
[ -f $OLD/adapter_model.safetensors ] || { log "missing old step$STEP ($OLD)"; exit 1; }
log "eval step $STEP: new $NEW, old $OLD"
pkill -f '[v]llm serve'; sleep 5
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-mid2b.log old=$OLD new=$NEW >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
for m in old new; do
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $C/${m}_a.jsonl >> $LOG 2>&1 || log "gen $m a FAILED"
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $C/${m}_b.jsonl >> $LOG 2>&1 || log "gen $m b FAILED"
done
pkill -f '[v]llm serve'; sleep 10
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --base $FS/models/Qwen3.5-9B --log $A/vllm-mid9b.log critic=$FS/adapters/$CR-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
for f in old_a old_b new_a new_b; do
  python -u -m ragtruth.critic_score --answers $C/$f.jsonl --url http://127.0.0.1:8000/v1 --temp $T --out $C/${f}_9b.json >> $LOG 2>&1 || log "score $f FAILED"
done
python -u -m ragtruth.shadow_rescore --run $O/run --dev $O/dev_ref.jsonl --url http://127.0.0.1:8000/v1 --temp $T >> $LOG 2>&1 || log "shadow rescore FAILED"
log "shadow: $(tail -1 $O/run/shadow9b.jsonl 2>/dev/null | cut -c1-300)"
pkill -f '[v]llm serve'
log "EVAL_MID_DONE"
