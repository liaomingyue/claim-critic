#!/bin/bash
# 2026-11-09 用户：核心是造 critic 盲区样本 → critic 难度当主奖励（0.3–0.5 最高、两头 0），sev 有错 ≥ 0.9 + 自然度 ≥ 0.5 当过滤。
#   ge：从 SFT（gen-v2-mc）起；ge2：从 ga 第 100 步起。每步 8 提示 × 32 采样（用户：n 16 → 32；门槛固定 0.9）。评测沿用 run_gengrpo.sh（v0 / ga 已有的跳过）。
# （原 run_gengrpo.sh 注释：生成器 GRPO，sev 奖励；自然度和 critic 作为门槛过滤。）单卡 5090（用户：2B 不用双卡）：vLLM 和训练同卡，vLLM 占 40%。
#   策略 = 2B 生成器 v2（gen-v2-mc）；提示 = 训练池 657 回答 × 9 机制；奖励 train/gen_reward_v2.py（--reward v2sev）
#   ga：sev + 自然度门槛；gb：sev + 自然度门槛 + critic r2 区间 [0.1, 0.7) 门槛。各 STEPS 步。
#   评测：留出原文 genpool_eval（70 回答 × 9 机制 × 4）上 v0 / ga / gb 采样 → 9B 基座写描述 → sev v3c0（带描述）/ RM2 v3b / critic r2 打分 → 产出率。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gengrpo_hard.log; O=outputs/rt/gen_v2_grpo; C=outputs/rt/core
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
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128
  --steps ${STEPS:-100} --lr 5e-6 --n 32 --prompts_per_step 8 --save_every 50 --state_every 50"
GA_CK=$O/ga/ckpt/step100; [ -d $GA_CK/policy ] && GA_CK=$GA_CK/policy
declare -A ARGS=([ge]="--v2_shape hard --err_gate 0.9 --nat_gate 0.5" [ge2]="--v2_shape hard --err_gate 0.9 --nat_gate 0.5 --init $GA_CK")
for e in ${EXPS:-ge ge2}; do
  R=$O/$e
  [ -d $R/ckpt/step${STEPS:-100} ] && { log "skip $e (done)"; continue; }
  free_gpus
  A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-gg.log --priority \
    sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter policy_v0=$([ $e = ge2 ] && echo $GA_CK || echo $INIT) >> $LOG 2>&1 || { log "$e vllm FAILED"; continue; }
  # 训练前试采（用户：担心大量组没有正样本）：64 个提示 × 32，同款奖励；整组无正样本 > MAX_ZERO（默认 0.3）就不训这组
  python -u -m ragtruth.gen_v2_grpo probe --model policy_v0 --tokenizer $MC >> $LOG 2>&1 && mv $O/probe_policy_v0.jsonl $O/probe_$e.jsonl
  PR=$(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1); log "PROBE $e $PR"
  Z=$(echo "$PR" | python -c "import sys,json;print(json.loads(sys.stdin.read().split(' ',1)[1])['groups_zero_pos'])" 2>/dev/null || echo 1)
  # 用户：启动之前先看和线上一致的奖励情况 → 试采完等本地确认（touch $O/GO_$e 继续；SKIP_$e 跳过），最多等 3 小时
  log "WAIT_GO $e"
  for i in $(seq 1 180); do [ -f $O/GO_$e ] || [ -f $O/SKIP_$e ] && break; sleep 60; done
  [ -f $O/GO_$e ] || { log "$e not approved, skip"; continue; }
  log "train $e: ${ARGS[$e]}"
  [ $e = ge2 ] && P0=$GA_CK || P0=$INIT
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo $BASE ${ARGS[$e]} --version_prefix ${e}_ --out $R >> $A/gengrpo_$e.log 2>&1 && log "$e done" || log "$e FAILED"
  tail -3 $A/gengrpo_$e.log | cut -c1-400 | sed "s/^/[$e] /" | tee -a $LOG
  cp $A/gengrpo_$e.log $R/ 2>/dev/null
done

# 评测
free_gpus
wait $CP9; source $S/model_env.sh qwen3.5-9b && B9=$MODEL_LOCAL
CK=${STEPS:-100}; GE=$O/ge/ckpt/step$CK; [ -d $GE/policy ] && GE=$GE/policy; GE2=$O/ge2/ckpt/step$CK; [ -d $GE2/policy ] && GE2=$GE2/policy
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ge2.log ge=$GE ge2=$GE2 >> $LOG 2>&1 || { log "eval gen vllm FAILED"; exit 1; }
for e in ge ge2; do
  [ -s $C/samples_grpo_${e}_mc.jsonl ] || python -u -m ragtruth.gen_v2 sample --url http://127.0.0.1:8000/v1 --model $e --tokenizer $MC --pool genpool_eval.jsonl --n 4     --out samples_grpo_${e}_mc.jsonl >> $LOG 2>&1 || { log "sample $e FAILED"; continue; }
  log "SAMPLED $e $(grep -a GENV2_SAMPLE $LOG | tail -1 | cut -c1-300)"
  python -u -m ragtruth.gen_v2_grpo items --samples samples_grpo_${e}_mc.jsonl --name $e >> $LOG 2>&1
done
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem 0.85 --base $B9 --log $A/vllm-ge9.log gen=$FS/adapters/gen-v2-9b/adapter >> $LOG 2>&1 || { log "eval 9b vllm FAILED"; exit 1; }
for e in ge ge2; do
  python -u -m ragtruth.gen_v2 describe --url http://127.0.0.1:8001/v1 --model base --samples samples_grpo_${e}_mc.jsonl --no_mech --out grpo_desc_$e.jsonl >> $LOG 2>&1 || log "desc $e FAILED"
done
free_gpus
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-ge3.log   v3=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter >> $LOG 2>&1 || { log "eval score vllm FAILED"; exit 1; }
for e in ge ge2; do
  python -u -m ragtruth.score_samples --items ../gen_v2_grpo/$e/items.jsonl --desc grpo_desc_$e.jsonl     --temps v3=$(t $SEV) rm2v3=$(t rm2__v3b) rm1=$(t rm1__core_r2) --out ../gen_v2_grpo/$e/scores.jsonl >> $LOG 2>&1 || log "score $e FAILED"
done
python -u -m ragtruth.gen_v2_grpo report --names v0 ga gb ge ge2 >> $LOG 2>&1
grep -a -A40 GEN_V2_GRPO_REPORT $LOG | tail -40 >> $LOG
free_gpus
log "GENGRPO_HARD_DONE"
