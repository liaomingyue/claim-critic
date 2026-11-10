#!/bin/bash
# 训练提速对比（定位 GPU 利用率低的原因并优化）：同一对机器、同样设置各跑 STEPS 步，
#   thread：采样 / 打分在训练进程的线程里（原来的做法）；proc：放子进程（--proc_producer 1，训练进程独占 GIL）。
# 双机模式，推理机已起（run_infer_node.sh）。不写 train_done（推理机留着）。比较第 3 步以后的 t_train / t_step 中位数。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/speedtest.log; S=$(dirname $(readlink -f $0))
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
P=$FS/runs/pair/${PAIR:-default}
export VLLM_API_KEY=$(cat $P/key) PYTHONHASHSEED=0
export OPENAI_API_KEY=$VLLM_API_KEY
D6=$(python -c "import json;print(json.load(open('$P/domains.json'))['svc6006'])"); D8=$(python -c "import json;print(json.load(open('$P/domains.json'))['svc6008'])")
export RM_URL=https://$D6/v1 LORA_SYNC_URL=https://$D8 LORA_SYNC_MODE=fs LORA_SYNC_FS_DIR=$FS/xfer/speed
mkdir -p $LORA_SYNC_FS_DIR
for mode in ${MODES:-thread proc}; do
  O=outputs/rt/minicpm/speed_$mode; rm -rf $O; mkdir -p $O
  for f in train_all.jsonl dev.jsonl train.jsonl train.meta.json train_ref.jsonl dev_ref.jsonl; do cp outputs/rt/minicpm/grpo_h2/$f $O/; done
  X="--eval_k 0"; [ $mode = proc ] && X="$X --proc_producer 1"; [ $mode = pipe ] && X="$X --proc_producer 1 --pipeline 1"
  log "speedtest $mode start"
  O=$O CRITIC=rm1__mcgold_h2 LR=5e-6 STEPS=${STEPS:-12} ADOUT=speed-$mode TRAIN_OPTS="--share_ckpt 1 --share_mb 32" EXTRA_ARGS="$X" bash $S/run_mcgrpo.sh
  python -c "
import json, statistics as st
r=[json.loads(l) for l in open('$O/run/log.jsonl') if 't_step' in l][2:]
print('SPEED', '$mode', len(r), {k: round(st.median(x[k] for x in r), 1) for k in ('t_step','t_train','t_wait','t_gen','t_rm','t_push')})
" | tee -a $LOG
  tail -60 $O/run/gpu.csv 2>/dev/null | python -c "import sys,statistics as st; u=[float(l.split(',')[2].strip().rstrip(' %')) for l in sys.stdin if l.count(',')>=3]; print('GPU', '$mode', 'util mean', round(st.mean(u),1) if u else None)" | tee -a $LOG
done
rm -rf $LORA_SYNC_FS_DIR
log "SPEEDTEST_DONE"
