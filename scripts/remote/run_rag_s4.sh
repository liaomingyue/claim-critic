#!/bin/bash
# 2026-11-09 用户：第二轮 critic（rm1__core_rd2）完成后继续训 GRPO。
#   RAG-S4-rd2：从 RAG-S3-bf（cont4 第 100 步）出发，奖励模型换成 rm1__core_rd2，其余同前（R3、lr 5e-6、16 × 16、100 步、起点重采参考回答、
#   第 50 / 100 步 + 起点在三套验收上生成、9B 打分）→ outputs/rt/core/loop2_cont6；adapter → adapters/mc-grpo-r3_core_cont6。
cd ${CODE:-/root/autodl-fs/code/critic_model}
pkill -f '[v]llm serve'; sleep 5
START=/root/autodl-fs/adapters/mc-grpo-r3_core_cont4/step100 CRITIC=rm1__core_rd2 ARM=cont6 O_LOCAL=/root/autodl-tmp/runs/loop2_cont6 CODE=$PWD \
  bash scripts/remote/run_loop2cont.sh
[ -f outputs/rt/core/loop2_cont6/eval/s100_c_9b.json ] || exit 1
