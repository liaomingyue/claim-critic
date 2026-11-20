#!/bin/bash
# 9B 错误合成 GRPO 阶段二（两台单卡布局的训练机 JOB，run_train_node.sh 调用），设置同 2B 的 st2c：
#   从 9B 第一段（s1c9b）第 100 步出发；过门槛后 9B critic 给分越低分越高，难度权重固定 1.5（2B 的 st2 自动调权重时被压到 0.2–0.4，等于关掉了难度项）；
#   加判别器项（rm2__d2_9b）；每步 8 个任务 × 32 条；任务池同 2B 阶段二（prompts_free_c.jsonl：普查里 8 条至少一条达标的任务）。
#   基准线 = 第一段最后 10 步的平均分和三项平均概率（真错、自然、操作一致），任何一项掉到基准线下就给那一项加补偿权重。
#   推理机要挂判别器、起点策略 = 第一段终点：DISC=1 INIT=<第一段 step100> bash run_genops_infer9.sh
cd ${CODE:-/root/autodl-fs/code/critic_model}
FS=/root/autodl-fs; O=outputs/rt/gen_ops_grpo
E=${E:-s2c9} STAGE=2 N=32 PPS=8 STEPS=${STEPS:-100} LR=2e-5 PROMPTS=$O/prompts_free_c.jsonl INIT=${INIT:-$FS/adapters/gen-ops-grpo-s1c9b/step100/policy} \
  EXTRA="--ops_disc 1 --ops_ref ${REF:-0.394} --ops_refs ${REFS:-0.759,0.699,0.645} --ops_w_hard ${W:-1.5} --ops_w_fixed 1" \
  bash scripts/remote/run_genops_grpo9.sh
