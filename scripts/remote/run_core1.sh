#!/bin/bash
# 核心集第一阶段（在一台 AutoDL 单卡 5090 上把 README 第一到五步实际跑起来；先 2B 全流程，9B 只做自检）。
# 不调 Claude（Claude 标注已在本地跑完、随代码包上传）。每一步做完写标记，重跑跳过已完成的。
#   ① 核心集 build / derive（D1–D4、D7–D9、sev 核心集部分）
#   ② 2B 自检（分词器 / 模板 / prompt 共享单测 + vLLM 挂 LoRA 取选项 logprob）
#   ③ 训练：错误改写模型 v0（gen-core-mc）、RAG SFT（rag-sft-core）、critic（rm1__core）、满意度（sat_pair__core）、拒答（refuse__core）
#   ④ 改写模型 v0 在核心集训练池上采样（每个 prompt 4 条 → D5 RM2 负样本 + sev 待标池；只用核心集回答）；训 RM2（rm2__core）
#   ⑤ 9B 自检；RAG SFT 在三套验收原文上生成 + 9B 金标 critic 打分（run_poleval.sh）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; LOG=$A/core1.log; mkdir -p $C/done
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
unset VLLM_API_KEY LORA_SYNC_URL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
step() { [ -f $C/done/$1 ]; }
mark() { touch $C/done/$1; log "step $1 done"; }
source $S/model_env.sh minicpm5-2b || { log "model_env FAILED"; exit 1; }
MC=$MODEL_LOCAL
source $S/models9b.sh; start_9b_bg

# ① 数据
if ! step data; then
  python -u -m ragtruth.core build >> $LOG 2>&1 && python -u -m ragtruth.core derive --tokenizer $MC >> $LOG 2>&1 || { log "core data FAILED"; exit 1; }
  grep -a "CORE_BUILD\|CORE_DERIVE" $LOG | tail -2 | cut -c1-1500; mark data
fi

# ② 2B 自检
if ! step check2b; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.5 --base $MC --log $A/vllm-check2b.log chk=$MODEL_CHECK_ADAPTER >> $LOG 2>&1 || { log "serve 2b FAILED"; exit 1; }
  python -u -m scripts.model_check --model minicpm5-2b --url http://127.0.0.1:8000/v1 >> $LOG 2>&1 && mark check2b || log "check 2b FAILED (see outputs/rt/model_check)"
  free_gpu
fi

# ③ 训练
sft() {   # sft <名字> <模板配置> <数据>
  local name=$1 cfg=$2 data=$3 CK=$A/ckpt/$1
  [ -f $FS/adapters/$name/adapter/adapter_model.safetensors ] && return 0
  mkdir -p configs/rt/exp
  sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $data|" -e "s|output_dir: .*|output_dir: $CK|" $cfg > configs/rt/exp/$name.yaml
  log "train $name ($(wc -l < $data) rows)"
  python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "$name FAILED"; return 1; }
  mkdir -p $FS/adapters/$name && rsync -a $CK/adapter $FS/adapters/$name/ && cp $CK/run_manifest.json $CK/sft_meta.json $FS/adapters/$name/ 2>/dev/null
  log "$name done: $(grep -ao "'train_loss': '[0-9.]*'" $A/$name.log | tail -n 1)"
}
free_gpu
sft gen-core-mc configs/rt/sft_gen_seed5_mc.yaml $C/seed_sft_train.jsonl || exit 1
sft rag-sft-core configs/rt/sft_policy_mc.yaml $C/rag_sft_train.jsonl || exit 1
for rm in rm1__core sat_pair__core refuse__core; do
  [ -f outputs/rt/rm/$rm/eval/report.json ] && continue
  RM=$rm GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/$rm/eval/report.json ] || { log "$rm FAILED"; exit 1; }
done

# ④ 改写模型 v0 采样 → RM2
if ! step sample; then
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-gen.log gen=$FS/adapters/gen-core-mc/adapter >> $LOG 2>&1 || { log "serve gen FAILED"; exit 1; }
  python -u -m ragtruth.core sample --model gen --tokenizer $MC --pool genpool_train.jsonl --out samples_core.jsonl --n 4 >> $LOG 2>&1 || { log "sample core FAILED"; exit 1; }
  python -u -m ragtruth.core sample --model gen --tokenizer $MC --pool genpool_eval.jsonl --out samples_eval.jsonl --n 4 >> $LOG 2>&1 || { log "sample eval FAILED"; exit 1; }
  free_gpu
  python -u -m ragtruth.core rm2 --inp samples_core.jsonl >> $LOG 2>&1 && python -u -m ragtruth.core sevpool --inp samples_core.jsonl --n 800 >> $LOG 2>&1 || { log "rm2 / sevpool FAILED"; exit 1; }
  grep -a "CORE_SAMPLE\|CORE_RM2\|CORE_SEVPOOL" $LOG | tail -5 | cut -c1-800; mark sample
fi
if [ ! -f outputs/rt/rm/rm2__core/eval/report.json ]; then
  RM=rm2__core GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/rm2__core/eval/report.json ] || { log "rm2__core FAILED"; exit 1; }
fi

# ⑤ 9B 自检 + RAG SFT 评测
if ! step check9b; then
  ensure_9b || { log "9b copy FAILED"; exit 1; }
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $A/models/Qwen3.5-9B --log $A/vllm-check9b.log chk=/root/autodl-fs/adapters/rm1__rm1_gold_9b-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
  python -u -m scripts.model_check --model qwen3.5-9b --url http://127.0.0.1:8000/v1 >> $LOG 2>&1 && mark check9b || log "check 9b FAILED (see outputs/rt/model_check)"
  free_gpu
fi
NAME=rag-sft-core ADAPTER=$FS/adapters/rag-sft-core/adapter OUT=$C/eval_rag_sft bash $S/run_poleval.sh >> $LOG 2>&1 || log "poleval FAILED"
grep -ah "MODEL_CHECK\|report\|auc" $LOG | tail -12 | cut -c1-300
log "CORE1_DONE"
