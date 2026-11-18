#!/bin/bash
# 按操作合成的盲区样本补过的 critic 当奖励，接着训 RAG（一台机器先后两组）。
#   起点都是 RAG-S3-bf（cont4 第 100 步，上一个显著提升的版本，它的奖励模型就是这次的基线 critic），其余同 RAG-S4-rd2：
#   R3、lr 5e-6、16 × 16、100 步、起点重采参考回答、起点 + 第 50 / 100 步在三套验收上生成、9B 打分。
#   每组写成 名字=critic[:满意度权重]，不写权重就是 0.8。
#   cont7：critic = rm1__core_of1cl（核实留下的 318 条）；cont8：critic = rm1__core_of1all（盲区全量 1,420 条、不核实）
cd ${CODE:-/root/autodl-fs/code/critic_model}
for x in ${ARMS:-cont7=rm1__core_of1cl cont8=rm1__core_of1all}; do
  ARM=${x%%=*}; CR=${x#*=}; SW=0.8; case $CR in *:*) SW=${CR#*:}; CR=${CR%%:*} ;; esac
  [ -f outputs/rt/core/loop2_$ARM/eval/s100_c_9b.json ] && continue
  pkill -f '[v]llm serve'; sleep 5
  START=/root/autodl-fs/adapters/mc-grpo-r3_core_cont4/step100 SAT_W=$SW CRITIC=$CR ARM=$ARM O_LOCAL=/root/autodl-tmp/runs/loop2_$ARM CODE=$PWD bash scripts/remote/run_loop2cont.sh
  [ -f outputs/rt/core/loop2_$ARM/eval/s100_c_9b.json ] || { echo "$(date '+%F %T') RAGS5 FAILED $ARM" | tee -a /root/autodl-tmp/loop2cont.log; exit 1; }
  echo "$(date '+%F %T') RAGS5_ARM_DONE $ARM" | tee -a /root/autodl-tmp/loop2cont.log
done
echo "$(date '+%F %T') RAGS5_DONE" | tee -a /root/autodl-tmp/loop2cont.log
