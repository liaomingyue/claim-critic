#!/bin/bash
# 多机模式训练机的 JOB（run_train_node.sh 调用，RM_URL 等已设好）：从 RAG-S2 出发，奖励里的 critic / 满意度 / 拒答换成 9B 版，训 100 步。
# 两台单卡：2B 策略的采样服务和训练都在训练机这张卡上，9B 打分在另一台（配对 runs/pair/s2r9，bootstrap 角色 s2r9rm / s2r9tr）。
# 对照：同一起点、同样 100 步、只换奖励 critic 的 RAG-S3-r1 / r2 / r3 / bf（2B critic）。其余设定同 run_loop2cont.sh
# （R3、lr 5e-6、16 × 16、沿用 RAG-S2 起点重采的参考回答 loop2_prompts_step100；起点 + 第 50 / 100 步在三套验收上生成、9B 金标打分）。
# 结果 → outputs/rt/core/loop2_cont7；adapter → adapters/mc-grpo-r3_core_cont7。
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}
INIT=$FS/adapters/mc-grpo-r3_core_cont/step100/policy
[ -f $INIT/adapter_model.safetensors ] || { echo "missing start adapter $INIT"; exit 1; }
[ -s outputs/rt/core/loop2_prompts_step100/train_ref.jsonl ] || { echo "missing reference answers (loop2_prompts_step100)"; exit 1; }
POLICY_MODEL=minicpm5-2b RM_MODEL=qwen3.5-9b ARM=cont7 CRITIC=rm1__core_9b SAT=sat_pair__core_9b REFUSE=refuse__core_9b INIT=$INIT \
  STEPS=100 EVAL_STEPS="50 100" EVAL_V0=1 SHARE_MB=32 O_LOCAL=$A/runs/loop2_cont7 CODE=$PWD bash scripts/remote/run_loop2.sh
[ -f outputs/rt/core/loop2_cont7/eval/s100_c_9b.json ] || exit 1
