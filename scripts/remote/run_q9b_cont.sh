#!/bin/bash
# 双机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：全 9B 链路第 4 步，9B RAG 策略从第 200 步接着训、奖励换新 critic
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
bash scripts/remote/q9b_task.sh grpo_c
