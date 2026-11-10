#!/bin/bash
# 双机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：9B 策略 × 9B 打分模型的 RAG GRPO
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
bash scripts/remote/q9b_task.sh grpo p9r9
