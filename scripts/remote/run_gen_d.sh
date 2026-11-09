#!/bin/bash
# 2026-11-09 用户：第 ③ 步改成更大的提示池，不局限核心集。
#   ① 2B RAG 模型（RAG GRPO 最好版 cont s100；rag-sft-core 版的池子留在 rag_pool_sftcore）在 RAGTruth 训练原文（QA / Summary，去掉评测 / 留出原文）上采回答：温度 1.0、每个原文 8 条（可用原文只有 690 个）、种子固定
#   ② critic r2 逐句打分，P(有幻觉) < 0.1 的算干净句（用户 2026-11-09），只允许在干净句上造错；回答 × 9 种错误类型 = 提示
#   ③ g1（GRPO 50 步）在这些提示上各采 1 条、温度 0（确定性），筛选同 B：过门槛、critic 0.1–0.5、要求类型 ≥ 0.5，并入前两轮难样本
#   ④ SFT → gen-v2h3-mc → 在同一批 64 个试采提示 × 32 上试采
# 接管 run_gen_c.sh：等它把 g1 试采完（PROBE g1s50），停掉它（原定的第 ③ 步不跑），从这里接着做。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/gen_d.log; O=outputs/rt/gen_v2_grpo
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpus() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
source $S/bg_sync.sh
R=$A/runs; V2H3=$A/ckpt/gen-v2h3-mc/adapter; SEV=sev__v3c0; RAG=${RAG:-$FS/adapters/mc-grpo-r3_core_cont/step100/policy}   # 用户：用效果最好的 RAG GRPO（cont 第 100 步：多段、句子级 R3 奖励）采回答
RMS="sev=$FS/adapters/$SEV-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
ck() { local c=$R/$1/ckpt/step50; [ -d $c/policy ] && c=$c/policy; echo $c; }
probe() {
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-dp.log $RMS $1=$2 >> $LOG 2>&1 || { log "probe $1 vllm FAILED"; return 1; }
  python -u -m ragtruth.gen_v2_grpo probe --model $1 --tokenizer $MC >> $LOG 2>&1
  log "PROBE $1 $(grep -a GEN_V2_GRPO_PROBE $LOG | tail -1 | cut -c1-2500)"
}
until grep -aqE 'PROBE g1s50|g1 FAILED|GEN_C_DONE' $A/gen_c.log 2>/dev/null; do sleep 20; done
pkill -f run_gen_c.sh; pkill -f 'gen_v2_grpo harvest'; sleep 5
grep -aq 'g1 FAILED' $A/gen_c.log && { log "g1 FAILED, stop"; exit 1; }
log "took over from run_gen_c after g1 probe"
rsync -a --exclude lora $R/g1/ $O/g1/ >> $A/bg_sync.log 2>&1 &   # g1 最后再同步一次（run_gen_c 的镜像循环被一起停了）
if [ ! -s $O/rag_pool/prompts.jsonl ]; then
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-dr.log $RMS rag=$RAG >> $LOG 2>&1 || { log "rag vllm FAILED"; exit 1; }
  [ -s $O/rag_pool/answers.jsonl ] || python -u -m ragtruth.gen_pool_rag answers --model rag --tokenizer $MC --k 8 --temperature 1.0 >> $LOG 2>&1 || { log "answers FAILED"; exit 1; }
  log "$(grep -a RAG_POOL_ANSWERS $LOG | tail -1 | cut -c1-800)"
  python -u -m ragtruth.gen_pool_rag clean --tokenizer $MC --thr 0.1 >> $LOG 2>&1 || { log "clean FAILED"; exit 1; }
  log "$(grep -a RAG_POOL_CLEAN $LOG | tail -1 | cut -c1-800)"
fi
if [ ! -s $O/sft_hard_v3.jsonl ]; then
  free_gpus
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-dh.log $RMS policy_v0=$(ck g1) >> $LOG 2>&1 || { log "harvest vllm FAILED"; exit 1; }
  python -u -m ragtruth.gen_v2_grpo harvest --model policy_v0 --tokenizer $MC --prompts rag_pool/prompts.jsonl --n 1 --temperature 0 --per_prompt 1 \
    --lo 0.1 --hi 0.5 --min_req 0.5 --tag _v3 --prev_hard harvest_hard_v2.jsonl --concurrency 128 >> $LOG 2>&1 || { log "harvest FAILED"; exit 1; }
  log "HARVEST $(grep -aE 'GEN_V2_GRPO_HARVEST|PREV_HARD' $LOG | tail -2 | tr '\n' ' ' | cut -c1-1500)"
fi
if [ ! -f $A/ckpt/gen-v2h3-mc/SAVED ]; then
  free_gpus
  CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/sft_gen_v2h3_mc.yaml --mode coldstart > $A/gen-v2h3-mc.log 2>&1 || { log "sft v2h3 FAILED"; exit 1; }
  log "trained gen-v2h3-mc $(grep -ao "'train_loss': '[0-9.]*'" $A/gen-v2h3-mc.log | tail -1)"
fi
bg_sync $V2H3 $FS/adapters/gen-v2h3-mc/
[ -s $O/probe_v2h3.jsonl ] || probe v2h3 $V2H3
free_gpus
log "GEN_D_DONE"
