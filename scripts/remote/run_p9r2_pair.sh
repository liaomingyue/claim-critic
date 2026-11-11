#!/bin/bash
# 多机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：9B 策略 × 2B 打分模型的 RAG GRPO。
# 三台：训练机 + 9B 采样机（POLICY_URL，收每步 LoRA）+ 2B 打分机（RM_URL），见 run_train_node.sh 的 RM_PAIR。
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
bash scripts/remote/q9b_task.sh grpo p9r2
