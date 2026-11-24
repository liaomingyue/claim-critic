#!/bin/bash
# 客服 Agent GRPO 的额外推理机：和主配对（MAIN_PAIR，默认 cs1）共用同一把密钥、同一个策略起点，自己的配对目录是 PAIR（默认 cs1b）。
#   训练机（run_train_node.sh）读主配对目录里的 extra_pairs，把采样分摊到所有推理机、每步把 LoRA 推给每一台。
FS=/root/autodl-fs; R=${CODE:-$FS/code/critic_model}; MAIN=$FS/runs/pair/${MAIN_PAIR:-cs1}; P=$FS/runs/pair/${PAIR:-cs1b}
mkdir -p $P
cp $MAIN/key $P/key && chmod 600 $P/key
echo "policy_v0=$FS/adapters/cs-zero-9b" > $P/loras
rm -f $P/infer_ready $P/infer_failed $P/train_done
PAIR=${PAIR:-cs1b} INFER_MODEL=qwen3.5-9b MAX_LEN=32768 MAX_LORAS=6 CODE=$R bash $R/scripts/remote/run_infer_node.sh
