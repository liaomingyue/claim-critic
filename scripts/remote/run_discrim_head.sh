#!/bin/bash
# 分类头版本的真实性判别器（加开的机器上跑）：训练 → 留出集评测 → 把适配器和分类头同步回主实例。
# 日志 /root/autodl-tmp/discrim_head.log。用法：bash run_discrim_head.sh <主实例 ssh 主机> <端口>
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/discrim_head.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/root/autodl-tmp/ckpt/discrim-head-9b
log "train classification-head discriminator"
python -u -m train.discrim_head train --out $CK > /root/autodl-tmp/discrim-head-train.log 2>&1 || { log "train FAILED"; exit 1; }
log "score heldout"
python -u -m train.discrim_head score --ckpt $CK --rows outputs/rt/discrim/heldout.jsonl \
  --out outputs/rt/discrim/heldout_head_scores.jsonl --eval >> $LOG 2>&1 || { log "score FAILED"; exit 1; }
if [ -n "${1:-}" ]; then   # 适配器 + 分类头约 0.2 GB，同步回主实例
  E="ssh -p $2 -o StrictHostKeyChecking=no"
  rsync -a -e "$E" $CK root@$1:/root/autodl-tmp/ckpt/ && \
  rsync -a -e "$E" outputs/rt/discrim/heldout_head_scores* root@$1:/root/autodl-tmp/critic_model/outputs/rt/discrim/ && \
  log "synced to main"
fi
log "DISCRIM_HEAD_DONE"
