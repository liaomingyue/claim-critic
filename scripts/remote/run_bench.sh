#!/bin/bash
# 训练框架基准（用户提出，2026-10-28）：同一份 critic 训练数据、同样 LoRA 设置，各跑 30 步，比较速度、显存、loss。
#   trl    ：共享环境（同 train.sft）
#   unsloth：独立 venv（Unsloth 自带 torch / transformers 版本要求，不能动 vLLM 用的共享环境）
# 用法：bash run_bench.sh（工作目录 = 共享代码目录）。日志 /root/autodl-tmp/bench.log，结束写 BENCH_DONE。
cd ${CODE:-/root/autodl-fs/code/critic_model}
LOG=/root/autodl-tmp/bench.log; O=outputs/rt/bench; V=/root/autodl-tmp/uvenv
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
DATA=outputs/rt/claims/closed_g0.jsonl
mkdir -p $O
pkill -f '[v]llm serve'; sleep 5
log "bench trl"
python -u -m train.bench_train --framework trl --data $DATA --steps 30 --out $O/trl.json > /root/autodl-tmp/bench-trl.log 2>&1 \
  && log "trl: $(cat $O/trl.json | tr -d '\n ')" || log "trl bench FAILED"
log "install unsloth venv"
python -m venv $V && timeout 1500 $V/bin/pip install -q unsloth openai pyyaml rapidfuzz > /root/autodl-tmp/bench-pip.log 2>&1 \
  && log "unsloth $($V/bin/python -c 'import unsloth; print(unsloth.__version__)' 2>/dev/null | tail -1)" || log "unsloth install FAILED"
log "bench unsloth"
$V/bin/python -u -m train.bench_train --framework unsloth --data $DATA --steps 30 --out $O/unsloth.json \
  > /root/autodl-tmp/bench-unsloth.log 2>&1 && log "unsloth: $(cat $O/unsloth.json | tr -d '\n ')" || log "unsloth bench FAILED"
log "BENCH_DONE"
