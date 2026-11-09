#!/bin/bash
# 循环二（RAG 生成，2026-11-06，用户：一台单卡顺序跑）：从 RAG SFT 出发，R3 奖励 GRPO。只用核心集派生的数据 / 模型；9B 金标 critic 只做训完后的评测。
#   ARM=<组名> CRITIC=<critic> [INIT=<起点 LoRA>] [STEPS=200] bash run_loop2.sh
#   ① 原文池 D2（outputs/rt/core/grpo_train.jsonl，750 个，去掉核心集留出原文）+ 原 dev
#   ② 起点模型重采参考回答（同 1.4：ans_data ref 温度 1.0 × 4 定长度范围、ref2 温度 0.7 × 4 取长度居中的一条当满意度参考）
#   ③ run_r3full.sh：R3（去底噪计数 K = max(2, 0.5 × 参考句数)、句子层 β 0.6 / γ 0.3、满意度只罚 0.8、拒答 0.4 × P）、lr 5e-6、n 16 × 16、TripWire；
#      第 50 / 100 / 200 步 + 起点在三套验收原文上生成、9B 金标 critic 打分 → outputs/rt/core/loop2_<组名>/eval/
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; ARM=${ARM:?}; LOG=$A/loop2_$ARM.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY LORA_SYNC_URL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b || { log "model_env FAILED"; exit 1; }
MC=$MODEL_LOCAL; INIT=${INIT:-$FS/adapters/rag-sft-core/adapter}; P=$C/loop2_prompts_$(basename $(dirname $INIT))
mkdir -p $P
if [ ! -s $P/train_ref.jsonl ]; then
  python - $C/grpo_train.jsonl outputs/rt/minicpm/grpo/dev.jsonl $P <<'EOF'
import json, sys, shutil
src, dev, out = sys.argv[1:]
keep = ("id", "source_id", "task_type", "source")
with open(f"{out}/train_all.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for l in open(src, encoding="utf-8"):
        r = json.loads(l); f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
shutil.copy(dev, f"{out}/dev.jsonl")
EOF
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ref.log policy_v0=$INIT >> $LOG 2>&1 || { log "serve start model FAILED"; exit 1; }
  python -u -m ragtruth.ans_data ref --dir $P --model policy_v0 --tokenizer $MC --sys_qa 1 --keep_refusal 1 >> $LOG 2>&1 || { log "ref FAILED"; exit 1; }
  python -u -m ragtruth.ans_data ref2 --dir $P --model policy_v0 --tokenizer $MC --sys_qa 1 >> $LOG 2>&1 || { log "ref2 FAILED"; exit 1; }
  free_gpu
  grep -a "REF_DONE\|REF2" $LOG | tail -3 | cut -c1-300
fi
log "R3 GRPO arm $ARM: init $INIT critic ${CRITIC:?} steps ${STEPS:-200}"
SRC=$P INIT=$INIT CRITIC=$CRITIC SAT=${SAT:-sat_pair__core} REFUSE=${REFUSE:-refuse__core} STEPS=${STEPS:-200} EVAL_STEPS="${EVAL_STEPS:-50 100 200}" EVAL_V0=${EVAL_V0:-1} \
  TAG=_core_$ARM O=${O_LOCAL:-$C/loop2_$ARM} bash $S/run_r3full.sh >> $LOG 2>&1
# 2026-11-09 用户：训练只读写本地盘（O_LOCAL），结束后整份同步到共享盘
[ -n "${O_LOCAL:-}" ] && mkdir -p $C/loop2_$ARM && rsync -a $O_LOCAL/ $C/loop2_$ARM/ && log "synced $O_LOCAL -> $C/loop2_$ARM"
grep -a "R3FULL_DONE\|FAILED" $A/r3full_core_$ARM.log | tail -3
log "LOOP2_DONE $ARM"
