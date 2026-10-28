#!/bin/bash
# 生成器 DPO 第一阶段（主实例）：等判别器训完 → 原始 9B 生成器采样（DPO 训练种子 + 留出评测种子）
# → 35B 判有效 → critic 打难度（目标 critic 9B v2；留出评测另用 critic 序列第 2 轮模型）→ 判别器打真实性
# → 组两种偏好对（只看难度 / 难度 + 真实性）→ 报告。日志 /root/autodl-tmp/dpo.log，结束写 STAGE1_DONE。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/dpo.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x
M=/root/autodl-tmp/models; CK=/root/autodl-tmp/ckpt; O=outputs/rt/dpo
MM='{"image":0,"video":0}'
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
serve() {  # serve <路径> <名字> <最大长度>
  stop
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len ${3:-8192} --gpu-memory-utilization 0.9 --port 8000 \
    --limit-mm-per-prompt "$MM" > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; exit 1
}
G="python -m ragtruth.gen_dpo"
until tail -1 /root/autodl-tmp/discrim.log 2>/dev/null | grep -q DISCRIM_DONE; do sleep 60; done

log "sample with base 9B generator"
serve $M/Qwen3.5-9B gen 16384
$G sample --dir $O/base --n_seeds ${N_TRAIN:-1500} --k 8 --hard_shots 2 >> $LOG 2>&1
$G sample --dir $O/eval_base_hs --n_seeds 500 --k 4 --seed 20 --hard_shots 2 --exclude_samples $O/base/samples.jsonl >> $LOG 2>&1
$G sample --dir $O/eval_base_noshot --n_seeds 500 --k 4 --seed 20 --hard_shots 0 --exclude_samples $O/base/samples.jsonl >> $LOG 2>&1

log "validity with 35B"
serve $M/Qwen3.5-35B-A3B-GPTQ-Int4 judge 16384
for d in base eval_base_hs eval_base_noshot; do $G valid --dir $O/$d --model judge >> $LOG 2>&1; done

log "hardness with target critic (9B v2)"
cp $M/Qwen3.5-9B/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} $CK/sft-9b-v2-notips/ 2>/dev/null
serve $CK/sft-9b-v2-notips critic
for d in base eval_base_hs eval_base_noshot; do $G hard --dir $O/$d --model critic --tag target >> $LOG 2>&1; done
log "hardness with held-out critic (iter critic r2)"
serve $CK/sft-9b-iter-critic-r2 critic
for d in eval_base_hs eval_base_noshot; do $G hard --dir $O/$d --model critic --tag heldout >> $LOG 2>&1; done

log "realism with discriminator"
serve $CK/discrim-9b discrim
for d in base eval_base_hs eval_base_noshot; do $G realism --dir $O/$d --model discrim --tag discrim >> $LOG 2>&1; done
stop

$G pairs --dir $O/base --mode hard >> $LOG 2>&1
$G pairs --dir $O/base --mode hard_real >> $LOG 2>&1
for d in base eval_base_hs eval_base_noshot; do $G report --dir $O/$d --tag target >> $LOG 2>&1; done
for d in eval_base_hs eval_base_noshot; do $G report --dir $O/$d --tag heldout >> $LOG 2>&1; done
log "STAGE1_DONE"
