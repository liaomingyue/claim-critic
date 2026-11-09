#!/bin/bash
# 2026-11-09 用户：第二轮闭环（看能不能迭代式上升）。RAG 这一步：从 cont 第 100 步，用第一轮 critic（rm1__core_r2_bfcl）当奖励接着训 100 步（R3），
#   第 50 / 100 步 + 起点在三套验收上生成、9B 金标打分 → outputs/rt/core/loop2_cont4；adapter → adapters/mc-grpo-r3_core_cont4。训练只读写本地盘，结束后同步。
cd ${CODE:-/root/autodl-fs/code/critic_model}
START=/root/autodl-fs/adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r2_bfcl ARM=cont4 O_LOCAL=/root/autodl-tmp/runs/loop2_cont4 CODE=$PWD \
  bash scripts/remote/run_loop2cont.sh
echo "$(date '+%F %T') ROUND2_RAG_DONE" >> /root/autodl-tmp/round2_rag.log
