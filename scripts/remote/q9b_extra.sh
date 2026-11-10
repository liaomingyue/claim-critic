#!/bin/bash
# 9B 对照的垫底任务（sev / RM2 的 9B 版）：prep 只建了 RAG 这组三个打分模型的 9B 数据目录，这里补建后再调 q9b_task.sh rm。
# 单独成一个脚本，是为了不去改正在运行的 q9b_task.sh（bash 边执行边读脚本，中途改文件会读错位置）。
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
n=${1:?name}; d=outputs/rt/rm/${n}_9b
if [ ! -f $d/READY ]; then
  mkdir -p $d
  for f in train.jsonl calib.jsonl test.jsonl meta.json; do [ -f outputs/rt/rm/$n/$f ] && cp outputs/rt/rm/$n/$f $d/; done
  touch $d/READY
fi
bash scripts/remote/q9b_task.sh rm $n
