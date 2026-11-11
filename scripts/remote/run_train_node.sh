#!/bin/bash
# 双机 GRPO 的训练机：等推理机就绪（run_infer_node.sh 写 runs/pair/$PAIR/infer_ready），
# 读密钥和推理机的对外域名（domains.json 由本地 scripts/pair_setup.py 写：{"svc6006": "...", "svc6008": "..."}），
# 设好 RM_URL / LORA_SYNC_URL / VLLM_API_KEY / PUSH_DIR 后跑 JOB（默认 run_grpo_h2.sh，训练卡 GPU0），结束写 train_done 让推理机收尾。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/train_node.log; S=$(dirname $(readlink -f $0))
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
PAIR=${PAIR:?}; P=$FS/runs/pair/$PAIR; JOB=${JOB:-run_grpo_h2.sh}
# 一开机就在后台把底座拷到本地盘（9B 训完评测总要用；2B 只有 2B 策略的训练机要，设 PRECOPY_2B=1），和等推理机的时间重叠（踩过：训练机空等推理机十几分钟，之后才开始拷 9B）。
# 先拷到临时目录再改名，别的脚本只看正式目录的 config.json；9B 用 models9b.sh 的同一把锁，后面 ensure_9b 不会重复拷
( source $S/models9b.sh
  [ "${PRECOPY_2B:-0}" = 1 ] && [ ! -f $A/models/MiniCPM5-2B/config.json ] && { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B/ $A/models/.MiniCPM5-2B.partial/ && mv $A/models/.MiniCPM5-2B.partial $A/models/MiniCPM5-2B; }
  [ "${PRECOPY_9B:-1}" = 1 ] && _copy_9b ) > /dev/null 2>&1 &
log "waiting for infer node ($P)"
until [ -f $P/infer_ready ] && [ -f $P/domains.json ]; do [ -f $P/infer_failed ] && { log "infer node FAILED"; exit 1; }; sleep 20; done
export VLLM_API_KEY=$(cat $P/key)
export OPENAI_API_KEY=$VLLM_API_KEY
# 多台推理机：$P/extra_pairs 每行一个配对名（runs/pair/<名>，同一把密钥、同一组 LoRA，由 bootstrap 角色 q9binf2 这类建好），
# 采样和打分按 prompt 分摊到所有推理机，每步的 LoRA 推给每一台（RM_URL / LORA_SYNC_URL 逗号分隔、顺序对应）
PS=$P; [ -f $P/extra_pairs ] && for x in $(cat $P/extra_pairs); do PS="$PS $FS/runs/pair/$x"; done
RMS=; SYNCS=
for p in $PS; do
  until [ -f $p/infer_ready ] && [ -f $p/domains.json ]; do [ -f $p/infer_failed ] && { log "infer node FAILED ($p)"; exit 1; }; sleep 20; done
  cmp -s $P/key $p/key || { log "key mismatch: $p"; exit 1; }
  D6=$(python -c "import json;print(json.load(open('$p/domains.json'))['svc6006'])"); D8=$(python -c "import json;print(json.load(open('$p/domains.json'))['svc6008'])")
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" https://$D6/v1/models)
  [ "$code" = 200 ] || { log "remote vllm not reachable ($p: $code)"; exit 1; }
  RMS="${RMS:+$RMS,}https://$D6/v1"; SYNCS="${SYNCS:+$SYNCS,}https://$D8"
done
# 单独的打分机（RM_PAIR=<配对名>，策略和打分模型底座不同、一张卡放不下两个服务时用，如 9B 策略 × 2B 打分）：
# 上面的推理机只做策略采样（POLICY_URL），打分请求全发给打分机（RM_URL）；打分机不收 LoRA
POLS=
if [ -n "${RM_PAIR:-}" ]; then
  Q=$FS/runs/pair/$RM_PAIR
  until [ -f $Q/infer_ready ] && [ -f $Q/domains.json ]; do [ -f $Q/infer_failed ] && { log "rm node FAILED ($Q)"; exit 1; }; sleep 20; done
  cmp -s $P/key $Q/key || { log "key mismatch: $Q"; exit 1; }
  D6=$(python -c "import json;print(json.load(open('$Q/domains.json'))['svc6006'])")
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" https://$D6/v1/models)
  [ "$code" = 200 ] || { log "remote rm not reachable ($Q: $code)"; exit 1; }
  POLS=$RMS; RMS=https://$D6/v1; PS="$PS $Q"
  log "rm node: $Q"
fi
# LoRA 直接上传给推理机（LORA_SYNC_MODE 默认 upload），存本机内存盘；要走共享盘就设 LORA_SYNC_MODE=fs PUSH_DIR=$FS/xfer/$PAIR
export RM_URL=$RMS LORA_SYNC_URL=$SYNCS SINGLE_GPU=1
[ -n "$POLS" ] && export POLICY_URL=$POLS
log "inference nodes: $(echo $PS | wc -w)"
log "remote inference reachable; running $JOB"
bash $S/$JOB; rc=$?
log "$JOB exit $rc"
# 只有成功才让推理机收尾；失败时推理机留着，改好直接重跑训练机（踩过：训练机 OOM 退出，推理机跟着关掉，重启又要等 5 分钟加载）
for p in $PS; do if [ $rc = 0 ]; then date '+%F %T' > $p/train_done; else date '+%F %T' > $p/train_failed; fi; done
log "TRAIN_NODE_DONE"
exit $rc
