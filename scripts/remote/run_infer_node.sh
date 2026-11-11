#!/bin/bash
# 双机 GRPO 的推理机（抢不到双卡时用两台单卡，一台训练一台推理）。
#   1. 生成本次配对的 API 密钥，写共享盘 runs/pair/$PAIR/key（只有自己的实例能读共享盘）
#   2. 6006 端口起 vLLM（MiniCPM 底座 + critic / sat / refuse / policy_v0 LoRA，同 run_mcgrpo.sh 的单机配置），要求密钥
#   3. 6008 端口起 LoRA 同步服务（scripts/lora_sync.py）：训练机每步把 LoRA 存共享盘，这里拷到内存盘再热加载
#   4. 写 infer_ready，等训练机写 train_done 再收尾。
# 两台机器经 AutoDL 自定义服务的对外域名通信（容器内网跨宿主机不通，实测域名带宽 78 MB/s、每次请求约 30 ms）。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/infer_node.log; S=$(dirname $(readlink -f $0)); MC=$FS/models/MiniCPM5-2B
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
PAIR=${PAIR:?}; P=$FS/runs/pair/$PAIR; mkdir -p $P
# 默认对应训练机的默认 JOB（run_grpo_h2.sh：h2 critic 当奖励）；换实验时两边要一致
CRITIC=${CRITIC:-rm1__mcgold_h2}; SATRM=${SATRM:-sat_pair__mc}; INIT=$FS/adapters/mc-init
[ -f $P/key ] || { python -c "import secrets;print(secrets.token_urlsafe(32))" > $P/key; chmod 600 $P/key; }
export VLLM_API_KEY=$(cat $P/key) PYTHONHASHSEED=0
LORAS="critic=$FS/adapters/$CRITIC-mc/adapter sat=$FS/adapters/$SATRM-mc/adapter refuse=$FS/adapters/refuse-mc/adapter policy_v0=$INIT"
# 换实验（如生成器 GRPO）时在共享盘 runs/pair/<配对>/loras 写「名字=路径」每行一个（路径可用 $FS），优先用它
[ -f $P/loras ] && LORAS=$(sed "s|\$FS|$FS|g" $P/loras | paste -sd' ')
pkill -f '[v]llm serve'
# 9B 对照：INFER_MODEL=qwen3.5-9b 时用 9B 底座（先拷到本地盘，从共享盘直接加载太慢）
if [ "${INFER_MODEL:-}" = qwen3.5-9b ]; then source $S/models9b.sh; ensure_9b || { log "9b copy FAILED"; touch $P/infer_failed; exit 1; }; MC=$M9; fi
if [ -f $P/rm_loras ]; then
  # 9B 策略 × 2B 打分：同卡两个服务（2B 打分 :8001 + 9B 策略 :8000），6006 放转发层按 model 名分流（scripts/vllm_router.py）。
  # 打分模型在 runs/pair/<配对>/rm_loras（底座 MiniCPM5-2B），loras 里只留策略。9B 权重约 18 GB，0.70 留几 GB 给 KV（只挂策略，LoRA 槽 4 个够用）；2B 0.22。
  RMLORAS=$(sed "s|\$FS|$FS|g" $P/rm_loras | paste -sd' ')
  [ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8001 --mem ${RM_MEM:-0.22} --base $A/models/MiniCPM5-2B --log $A/vllm-infer-rm.log --priority $RMLORAS >> $LOG 2>&1 || { log "vllm rm FAILED"; touch $P/infer_failed; exit 1; }
  A=$A/s1 MAX_LORAS=4 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${INFER_MEM:-0.70} --base $MC --log $A/vllm-infer.log --priority $LORAS >> $LOG 2>&1 || { log "vllm policy FAILED"; touch $P/infer_failed; exit 1; }
  setsid nohup python -u scripts/vllm_router.py --port 6006 --policy http://127.0.0.1:8000 --rm http://127.0.0.1:8001 > $A/vllm_router.log 2>&1 < /dev/null &
  for i in $(seq 1 20); do curl -s -m 2 http://127.0.0.1:6006/health | grep -q true && break; sleep 1; done
  curl -s -m 2 http://127.0.0.1:6006/health | grep -q true || { log "router FAILED"; touch $P/infer_failed; exit 1; }
  SYNC_VLLM=http://127.0.0.1:8000/v1; LORAS="$LORAS | rm: $RMLORAS"
else
  A=$A bash $S/serve_loras.sh --gpu 0 --port 6006 --mem ${INFER_MEM:-0.9} --base $MC --log $A/vllm-infer.log --priority $LORAS >> $LOG 2>&1 || { log "vllm FAILED"; touch $P/infer_failed; exit 1; }
  SYNC_VLLM=http://127.0.0.1:6006/v1
fi
setsid nohup python -u scripts/lora_sync.py --port 6008 --vllm $SYNC_VLLM > $A/lora_sync.log 2>&1 < /dev/null &
for i in $(seq 1 20); do curl -s -m 2 http://127.0.0.1:6008/health | grep -q ok && break; sleep 1; done
curl -s -m 2 http://127.0.0.1:6008/health | grep -q ok || { log "lora_sync FAILED"; touch $P/infer_failed; exit 1; }
log "infer node up: vllm 6006 ($LORAS), lora_sync 6008"
date '+%F %T' > $P/infer_ready
until [ -f $P/train_done ]; do sleep 30; done
log "train_done seen, shutting down"
pkill -f '[v]llm serve'; pkill -f '[l]ora_sync.py'; pkill -f '[v]llm_router.py'
log "INFER_NODE_DONE"
