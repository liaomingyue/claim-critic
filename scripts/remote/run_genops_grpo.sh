#!/bin/bash
# 按操作清单造错的生成器 GRPO（奖励 train/gen_reward_ops.py，三个阶段逐步加难）。单卡 5090：vLLM 和训练同卡。
#   策略起点 = 基座 2B + 卡片提示词（B 清零的 LoRA，等于基座）；提示 = 训练池回答 × 20 种操作（outputs/rt/gen_ops_grpo/prompts.jsonl）
#   把关模型 = sev__v4 / rm2__v4 / opm__v1（ragtruth/rm_ops.py），阶段二、三另加 critic rm1__core_r2_bfcl
#   环境变量：E=运行名  STAGE=1|2|3  LR  STEPS  INIT=起点（默认 B 清零的 LoRA；阶段二 / 三接上一阶段的 ckpt）  N  PPS
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo
E=${E:?}; STAGE=${STAGE:-1}; LR=${LR:-1e-5}; STEPS=${STEPS:-100}; N=${N:-16}; PPS=${PPS:-16}; LOG=$A/genops_grpo_$E.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
Z=$FS/adapters/gen-ops-zero-mc/adapter
[ -f $Z/adapter_model.safetensors ] || python scripts/zero_lora.py $FS/adapters/gen-ops-mc/adapter $Z >> $LOG 2>&1 || { log "zero lora FAILED"; exit 1; }
INIT=${INIT:-$Z}
[ -s $O/prompts.jsonl ] || python -u -m ragtruth.gen_ops prompts --pool genpool_train.jsonl >> $LOG 2>&1 || { log "prompts FAILED"; exit 1; }
R=$O/$E
[ -d $R/ckpt/step$STEPS ] && { log "skip $E (done)"; exit 0; }
free_gpu
A=$A bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.4 --base $MC --log $A/vllm-gog.log --priority \
  sev4=$FS/adapters/sev__v4-mc/adapter nat4=$FS/adapters/rm2__v4-mc/adapter opm=$FS/adapters/opm__v1-mc/adapter rm1=$FS/adapters/rm1__core_r2_bfcl-mc/adapter policy_v0=$INIT >> $LOG 2>&1 || { log "$E vllm FAILED"; exit 1; }
log "train $E: stage $STAGE lr $LR steps $STEPS n $N prompts/step $PPS init $INIT"
CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT --prompts $O/prompts.jsonl --reward ops --ops_stage $STAGE \
  --temps sev4=sev__v4,nat4=rm2__v4,opm=opm__v1,rm1=rm1__core_r2_bfcl \
  --share 1 --share_impl train.prefix_share_llama --share_mb 8 --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128 \
  --steps $STEPS --lr $LR --n $N --prompts_per_step $PPS --save_every 50 --state_every 50 --version_prefix ${E}_ --out $R ${EXTRA:-} >> $A/gengrpo_$E.log 2>&1 && log "$E done" || log "$E FAILED"
tail -3 $A/gengrpo_$E.log | cut -c1-400 | sed "s/^/[$E] /" | tee -a $LOG
cp $A/gengrpo_$E.log $R/ 2>/dev/null
for d in $R/ckpt/step*; do [ -d $d ] && mkdir -p $FS/adapters/gen-ops-grpo-$E/$(basename $d) && rsync -a $d/ $FS/adapters/gen-ops-grpo-$E/$(basename $d)/; done
free_gpu
log "GENOPS_GRPO_DONE $E"
