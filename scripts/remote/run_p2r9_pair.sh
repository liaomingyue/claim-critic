#!/bin/bash
# 多机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：2B 策略 × 9B 打分模型的 RAG GRPO。
# 2B 策略的采样服务和训练都在训练机这张卡上，critic / 满意度 / 拒答（9B）在推理机上打分；对照 RAG-S1（2B 策略 × 2B 打分）。
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
bash scripts/remote/q9b_task.sh grpo p2r9
