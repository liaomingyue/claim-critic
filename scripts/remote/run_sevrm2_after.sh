#!/bin/bash
# 2026-11-09 用户：两台机器都并行跑。sev / RM2 放主机器（f1274a9a78）：等第二轮 RAG 的训练和 9B 评测跑完（ROUND2_RAG_DONE），GPU 空出来就训；
#   第二台同时在用新 RAG 造样本，互不占用。
cd ${CODE:-/root/autodl-fs/code/critic_model}
until grep -aq 'ROUND2_RAG_DONE' /root/autodl-tmp/round2_rag.log 2>/dev/null; do sleep 30; done
bash scripts/remote/run_sevrm2_r1.sh
