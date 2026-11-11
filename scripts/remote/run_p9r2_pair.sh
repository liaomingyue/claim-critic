#!/bin/bash
# 多机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：9B 策略 × 2B 打分模型的 RAG GRPO。
# 推理机同卡挂 9B 策略和 2B critic / 满意度 / 拒答，6006 转发层按 model 名分流（scripts/vllm_router.py），训练机这边和 9B × 9B 一样。
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
bash scripts/remote/q9b_task.sh grpo p9r2
