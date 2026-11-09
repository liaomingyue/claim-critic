#!/bin/bash
# 2026-11-08 用户：sev 训好后在 critic 上实际验证。基线 rm1__core + 9B 错误改写模型样本（按规则挑错句，配同位置原句）→ 重训 critic → 评测。
#   组（ARMS，按顺序训，单卡）：
#     s2  新 sev__gen  P(A–D) ≥ THR2（s2 / s1 / rnd 候选池都去掉有 Claude 结论的 9B 样本，防 sev__gen 记住答案）
#     s1  旧 sev__core 按 P(A–D) 从高到低取，数量对齐 s2
#     rnd 不过滤随机取，数量对齐 s2
#     cl  只取 Claude 审计判为真错的 9B 样本、类型用 Claude 的（上限参照）
#   每组：critic_sevaug 建数据 → run_mcrm（校准 + RAGTruth++）→ 给三套验收回答打分（critic_cap3 本地算 Claude 金标 AUC）
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/critic_sev.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x; unset VLLM_API_KEY
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
THR2=${THR2:-0.9}; THR1=${THR1:-0.9}; ARMS=${ARMS:-"s2 s1 rnd cl"}
N=0
for arm in $ARMS; do
  name=rm1__core_sev$arm
  if [ ! -f outputs/rt/rm/$name/meta.json ]; then
    case $arm in
      s2)  python -u -m ragtruth.critic_sevaug --filter sev2 --thr $THR2 --exclude_labeled --name $name ;;
      s1)  python -u -m ragtruth.critic_sevaug --filter sev1 --rank --n_match $N --exclude_labeled --name $name ;;
      rnd) python -u -m ragtruth.critic_sevaug --filter none --n_match $N --exclude_labeled --name $name ;;
      cl)  python -u -m ragtruth.critic_sevaug --filter claude --claude_type --name $name ;;
    esac >> $LOG 2>&1 || { log "$arm build FAILED"; continue; }
  fi
  [ $arm = s2 ] && N=$(python -c "import json;print(json.load(open('outputs/rt/rm/$name/meta.json'))['added_errors'])")
  log "BUILT $arm $(python -c "import json;m=json.load(open('outputs/rt/rm/$name/meta.json'));print({k:m.get(k) for k in ('filter','thr','passed','added_errors','added_with_claude','added_claude_real')})")"
  if [ ! -f outputs/rt/rm/$name/eval/report.json ]; then
    free_gpu; RM=$name GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
    [ -f outputs/rt/rm/$name/eval/report.json ] || { log "$arm train FAILED"; continue; }
  fi
  T=$(python -c "import json;print(json.load(open('outputs/rt/rm/$name/eval/report.json'))['temperature'])")
  log "TRAINED $arm T=$T $(python -c "import json;r=json.load(open('outputs/rt/rm/$name/eval/report.json'));print({k:{m:r[k].get(m) for m in ('auc','recall@0.5','precision@0.5')} for k in ('calib_calibrated','test_calibrated')})")"
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-c$arm.log c=$FS/adapters/$name-mc/adapter >> $LOG 2>&1 || { log "$arm serve FAILED"; continue; }
  PIDS=""
  for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
    [ -f $d/critic_$name.json ] && continue
    python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$name.json >> $LOG 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait $p; done
  free_gpu
  log "SCORED $arm"
done
log "CRITIC_SEV_DONE"
