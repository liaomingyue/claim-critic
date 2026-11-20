#!/bin/bash
# 两台单卡跑 9B 错误合成 GRPO 的推理机（训练机 = run_train_node.sh + JOB=run_genops_grpo9.sh，配对 runs/pair/gog9）。
# 等 9B 把关模型训完 → 建 B 清零的起点 LoRA → 写要挂的 LoRA 清单 → run_infer_node.sh 起 vLLM（9B 基座 + 把关模型 + critic + 策略起点）和 LoRA 同步服务，等训练机写 train_done。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; R=outputs/rt/rm; LOG=$A/genops_infer9.log; P=$FS/runs/pair/${PAIR:-gog9}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
mkdir -p $P; rm -f $P/infer_ready $P/infer_failed $P/train_done $P/train_failed $P/domains.json; hostname > $P/infer_host
( source $S/models9b.sh; _copy_9b ) > /dev/null 2>&1 &
ad() { echo $FS/adapters/${1}_9b-9b/adapter; }
for i in $(seq 1 480); do
  miss=""; for g in sev__v4 rm2__v4 opm__v1 rm2__d2 rm1__core_r2_bfcl; do [ -f $(ad $g)/adapter_model.safetensors ] && [ -f $R/${g}_9b/eval/report.json ] || miss="$miss $g"; done
  [ -z "$miss" ] && break; [ $((i % 10)) = 1 ] && log "waiting for gates:$miss"; sleep 60
done
[ -z "$miss" ] || { log "GENOPS_INFER9 FAILED: gates missing:$miss"; touch $P/infer_failed; exit 1; }
Z=$FS/adapters/gen-ops-zero-9b/adapter
[ -f $Z/adapter_model.safetensors ] || python scripts/zero_lora.py $FS/adapters/rm1__core_9b-9b/adapter $Z >> $LOG 2>&1 || { log "GENOPS_INFER9 FAILED: zero lora"; touch $P/infer_failed; exit 1; }
# 第一段奖励里没有判别器项，不挂（推理机上的 LoRA 越少越好）；DISC=1 时再挂
printf '%s\n' "sev4=$(ad sev__v4)" "nat4=$(ad rm2__v4)" "opm=$(ad opm__v1)" "rm1=$(ad rm1__core_r2_bfcl)" $([ "${DISC:-0}" = 1 ] && echo "disc=$(ad rm2__d2)") "policy_v0=${INIT:-$Z}" > $P/loras
wait
MAX_LORAS=${MAX_LORAS:-12} PAIR=${PAIR:-gog9} INFER_MODEL=qwen3.5-9b bash $S/run_infer_node.sh || { log "GENOPS_INFER9 FAILED: infer node"; exit 1; }
log "GENOPS_INFER9_DONE"
