#!/bin/bash
# S2 拒绝采样 SFT（2026-11-05，用户：先把 S2 跑出来；和 GRPO 同一批 831 个 prompt）。
#   ① v0 每个 prompt 采 16 条，用和 R3 GRPO 完全相同的奖励打分（critic h3ctl、去底噪计数、满意度 0.8、拒答 0.4、硬规则）
#   ② 每个 prompt 取奖励最高的一条 → SFT（LoRA 同回答 GRPO 的策略：r 32 / alpha 64，2 轮，lr 5e-5）
#   ③ 独立评测：三套验收原文生成 + 9B 打分（run_poleval.sh）→ outputs/rt/minicpm/s2/eval/
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/minicpm/s2; LOG=$A/s2.log; mkdir -p $O
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!   # 保活进程 PID：下面的 wait 不能等它（2026-11-05 worker1 生成完卡死在 wait）
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg   # 2026-11-05 用户：先拷马上要用的 2B，9B 后台拷，把时间岔开
MC=$A/models/MiniCPM5-2B; CRITIC=${CRITIC:-rm1__mcgold_h3ctl}
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
if [ ! -s $O/sft_train.jsonl ]; then
  free_gpu; mkdir -p $A/s0
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-s2.log \
    critic=$FS/adapters/$CRITIC-mc/adapter sat=$FS/adapters/sat_pair__mc-mc/adapter refuse=$FS/adapters/refuse-mc/adapter policy_v0=$FS/adapters/mc-init >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
  python -u -m ragtruth.policy_sft s2_sample --out $O --url http://127.0.0.1:8000/v1 --tokenizer $MC --critic $CRITIC --n 16 --concurrency 16 >> $LOG 2>&1 || { log "s2_sample FAILED"; exit 1; }
  free_gpu
  python -u -m ragtruth.policy_sft s2_build --out $O --tokenizer $MC >> $LOG 2>&1 || { log "s2_build FAILED"; exit 1; }
fi
log "s2 data: $(cat $O/meta.json | tr -d '\n')"
CK=$A/ckpt/policy-s2
if [ ! -f $FS/adapters/policy-s2/adapter/adapter_model.safetensors ]; then
  sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $O/sft_train.jsonl|" -e "s|output_dir: .*|output_dir: $CK|" configs/rt/sft_policy_mc.yaml > configs/rt/exp/policy-s2.yaml
  python -u -m train.sft --config configs/rt/exp/policy-s2.yaml --mode coldstart > $A/policy-s2.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "s2 sft FAILED"; exit 1; }
  mkdir -p $FS/adapters/policy-s2 && rsync -a $CK/adapter $FS/adapters/policy-s2/ && log "s2 sft done: $(grep -ao "'train_loss': '[0-9.]*'" $A/policy-s2.log | tail -n 1)"
fi
NAME=s2 ADAPTER=$FS/adapters/policy-s2/adapter OUT=$O/eval bash $S/run_poleval.sh
log "S2_DONE"
