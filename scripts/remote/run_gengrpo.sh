#!/bin/bash
# 2026-11-09 用户：生成器 GRPO，核心目标是 sev 奖励、增加合格样本量；自然度和 critic 作为门槛过滤。单卡 5090（用户：2B 不用双卡）：vLLM 和训练同卡，vLLM 占 40%。
#   策略 = 2B 生成器 v2（gen-v2-mc）；提示 = 训练池 657 回答 × 9 机制；奖励 train/gen_reward_v2.py（--reward v2sev）
#   ga：sev + 自然度门槛；gb：sev + 自然度门槛 + critic r2 区间 [0.1, 0.7) 门槛。各 STEPS 步。
#   评测：留出原文 genpool_eval（70 回答 × 9 机制 × 4）上 v0 / ga / gb 采样 → 9B 基座写描述 → sev v3c0（带描述）/ RM2 v3b / critic r2 打分 → 产出率。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gengrpo.log; O=outputs/rt/gen_v2_grpo; C=outputs/rt/core
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL && log "2B ready"
( source $S/model_env.sh qwen3.5-9b && touch $A/9b_copied ) >> $LOG 2>&1 &
CP9=$!
mkdir -p $A/s0 $A/s1
INIT=$FS/adapters/gen-v2-mc/adapter
SEV=${SEV:-sev__v3c0}
[ -s $O/prompts.jsonl ] || python -u -m ragtruth.gen_v2_grpo prompts >> $LOG 2>&1 || { log "prompts FAILED"; exit 1; }
BASE="--causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --reward v2sev --temps sev=$SEV,rm2v3=rm2__v3b,rm1=rm1__core_r2
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512
  --steps ${STEPS:-100} --lr 5e-6 --n 8 --prompts_per_step 16 --save_every 50 --state_every 50"
declare -A ARGS=([ga]="--nat_gate 0.5" [gb]="--nat_gate 0.5 --crit_gate 1 --crit_lo 0.1 --crit_hi 0.7")
for e in ${EXPS:-ga gb}; do
  R=$O/$e
  [ -d $R/ckpt/step${STEPS:-100} ] && { log "skip $e (done)"; continue; }
  free_gpus
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-gg.log --priority \
    sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter policy_v0=$INIT >> $LOG 2>&1 || { log "$e vllm FAILED"; continue; }
  log "train $e: ${ARGS[$e]}"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE ${ARGS[$e]} --version_prefix ${e}_ --out $R >> $A/gengrpo_$e.log 2>&1 && log "$e done" || log "$e FAILED"
  tail -3 $A/gengrpo_$e.log | cut -c1-400 | sed "s/^/[$e] /" | tee -a $LOG
  cp $A/gengrpo_$e.log $R/ 2>/dev/null
done

# 评测
free_gpus
wait $CP9; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
CK=${STEPS:-100}; GA=$O/ga/ckpt/step$CK; [ -d $GA/policy ] && GA=$GA/policy; GB=$O/gb/ckpt/step$CK; [ -d $GB/policy ] && GB=$GB/policy
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ge2.log v0=$INIT ga=$GA gb=$GB >> $LOG 2>&1 || { log "eval gen vllm FAILED"; exit 1; }
for e in v0 ga gb; do
  [ -s $C/samples_grpo_${e}_mc.jsonl ] || python -u -m ragtruth.gen_v2 sample --url http://127.0.0.1:8000/v1 --model $e --tokenizer $MC --pool genpool_eval.jsonl --n 4     --out samples_grpo_${e}_mc.jsonl >> $LOG 2>&1 || { log "sample $e FAILED"; continue; }
  log "SAMPLED $e $(grep -a GENV2_SAMPLE $LOG | tail -1 | cut -c1-300)"
  python -u -m ragtruth.gen_v2_grpo items --samples samples_grpo_${e}_mc.jsonl --name $e >> $LOG 2>&1
done
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.85 --base $B9 --log $A/vllm-ge9.log gen=$FS/adapters/gen-v2-9b/adapter >> $LOG 2>&1 || { log "eval 9b vllm FAILED"; exit 1; }
for e in v0 ga gb; do
  python -u -m ragtruth.gen_v2 describe --url http://127.0.0.1:8001/v1 --model base --samples samples_grpo_${e}_mc.jsonl --no_mech --out grpo_desc_$e.jsonl >> $LOG 2>&1 || log "desc $e FAILED"
done
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ge3.log   v3=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter >> $LOG 2>&1 || { log "eval score vllm FAILED"; exit 1; }
for e in v0 ga gb; do
  python -u -m ragtruth.score_samples --items ../gen_v2_grpo/$e/items.jsonl --desc grpo_desc_$e.jsonl     --temps v3=$(t $SEV) rm2v3=$(t rm2__v3b) rm1=$(t rm1__core_r2) --out ../gen_v2_grpo/$e/scores.jsonl >> $LOG 2>&1 || log "score $e FAILED"
done
python -u -m ragtruth.gen_v2_grpo report >> $LOG 2>&1
grep -a -A40 GEN_V2_GRPO_REPORT $LOG | tail -40 >> $LOG
free_gpus
log "GENGRPO_DONE"
