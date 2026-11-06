#!/bin/bash
# S1 Claude 修正 SFT（2026-11-05，用户：Claude 只改错句，正确部分原样拼回；和 S2 / GRPO 同一批 831 个 prompt）。
#   ① Claude 修正结果（本地已跑完：fix_raw.jsonl + fix_single_raw.jsonl，上传到共享盘）→ s1_build 把改写按句子区间拼回参考回答
#   ② SFT（同 S2：LoRA r 32 / alpha 64，2 轮，lr 5e-5）
#   ③ 独立评测：三套验收原文生成 + 9B 打分（run_poleval.sh）→ outputs/rt/minicpm/s1/eval/
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/minicpm/s1; LOG=$A/s1.log; mkdir -p $O
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg
MC=$A/models/MiniCPM5-2B
[ -s $O/fix_raw.jsonl ] || { log "no fix_raw.jsonl"; exit 1; }
[ -s $O/sft_train.jsonl ] || python -u -m ragtruth.policy_sft s1_build --out $O --tokenizer $MC >> $LOG 2>&1 || { log "s1_build FAILED"; exit 1; }
log "s1 data: $(cat $O/meta.json | tr -d '\n')"
CK=$A/ckpt/policy-s1
if [ ! -f $FS/adapters/policy-s1/adapter/adapter_model.safetensors ]; then
  sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $O/sft_train.jsonl|" -e "s|output_dir: .*|output_dir: $CK|" configs/rt/sft_policy_mc.yaml > configs/rt/exp/policy-s1.yaml
  python -u -m train.sft --config configs/rt/exp/policy-s1.yaml --mode coldstart > $A/policy-s1.log 2>&1
  [ -f $CK/adapter/adapter_model.safetensors ] || { log "s1 sft FAILED"; exit 1; }
  mkdir -p $FS/adapters/policy-s1 && rsync -a $CK/adapter $FS/adapters/policy-s1/ && log "s1 sft done: $(grep -ao "'train_loss': '[0-9.]*'" $A/policy-s1.log | tail -n 1)"
fi
NAME=s1 ADAPTER=$FS/adapters/policy-s1/adapter OUT=$O/eval bash $S/run_poleval.sh
log "S1_DONE"
