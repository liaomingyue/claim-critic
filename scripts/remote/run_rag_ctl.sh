#!/bin/bash
# 对照实验——「接着训更好」到底是因为多训了步数，还是因为换了新 critic？
#   RAG-S3-r1：从 RAG-S2（cont 第 100 步）出发，奖励模型保持 S2 原来那个（rm1__core_r1）不换，其余和 RAG-S3-bf 完全相同（R3、lr 5e-6、16 × 16、100 步、
#   起点重采参考回答、第 50 / 100 步 + 起点在三套验收上生成、9B 打分）→ outputs/rt/core/loop2_cont5；adapter → adapters/mc-grpo-r3_core_cont5。
cd ${CODE:-/root/autodl-fs/code/critic_model}
pkill -f '[v]llm serve'; sleep 5
START=/root/autodl-fs/adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r1 ARM=cont5 O_LOCAL=/root/autodl-tmp/runs/loop2_cont5 CODE=$PWD \
  bash scripts/remote/run_loop2cont.sh
[ -f outputs/rt/core/loop2_cont5/eval/s100_c_9b.json ] || exit 1
