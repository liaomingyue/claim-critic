#!/bin/bash
# 双机 GRPO 的训练机：等推理机就绪（run_infer_node.sh 写 runs/pair/$PAIR/infer_ready），
# 读密钥和推理机的对外域名（domains.json 由本地 scripts/pair_setup.py 写：{"svc6006": "...", "svc6008": "..."}），
# 设好 RM_URL / LORA_SYNC_URL / VLLM_API_KEY / PUSH_DIR 后跑 JOB（默认 run_grpo_h2.sh，训练卡 GPU0），结束写 train_done 让推理机收尾。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/train_node.log; S=$(dirname $(readlink -f $0))
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
PAIR=${PAIR:?}; P=$FS/runs/pair/$PAIR; JOB=${JOB:-run_grpo_h2.sh}
log "waiting for infer node ($P)"
until [ -f $P/infer_ready ] && [ -f $P/domains.json ]; do [ -f $P/infer_failed ] && { log "infer node FAILED"; exit 1; }; sleep 20; done
export VLLM_API_KEY=$(cat $P/key)
export OPENAI_API_KEY=$VLLM_API_KEY
D6=$(python -c "import json;print(json.load(open('$P/domains.json'))['svc6006'])"); D8=$(python -c "import json;print(json.load(open('$P/domains.json'))['svc6008'])")
# LoRA 直接上传给推理机（LORA_SYNC_MODE 默认 upload），存本机内存盘；要走共享盘就设 LORA_SYNC_MODE=fs PUSH_DIR=$FS/xfer/$PAIR
export RM_URL=https://$D6/v1 LORA_SYNC_URL=https://$D8 SINGLE_GPU=1
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" $RM_URL/models)
[ "$code" = 200 ] || { log "remote vllm not reachable ($code)"; exit 1; }
log "remote inference reachable; running $JOB"
bash $S/$JOB; rc=$?
log "$JOB exit $rc"
# 只有成功才让推理机收尾；失败时推理机留着，改好直接重跑训练机（踩过：训练机 OOM 退出，推理机跟着关掉，重启又要等 5 分钟加载）
if [ $rc = 0 ]; then date '+%F %T' > $P/train_done; else date '+%F %T' > $P/train_failed; fi
log "TRAIN_NODE_DONE"
exit $rc
