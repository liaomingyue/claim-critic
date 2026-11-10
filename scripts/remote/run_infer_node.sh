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
A=$A bash $S/serve_loras.sh --gpu 0 --port 6006 --base $MC --log $A/vllm-infer.log --priority $LORAS >> $LOG 2>&1 || { log "vllm FAILED"; touch $P/infer_failed; exit 1; }
setsid nohup python -u scripts/lora_sync.py --port 6008 --vllm http://127.0.0.1:6006/v1 > $A/lora_sync.log 2>&1 < /dev/null &
for i in $(seq 1 20); do curl -s -m 2 http://127.0.0.1:6008/health | grep -q ok && break; sleep 1; done
curl -s -m 2 http://127.0.0.1:6008/health | grep -q ok || { log "lora_sync FAILED"; touch $P/infer_failed; exit 1; }
log "infer node up: vllm 6006 ($LORAS), lora_sync 6008"
date '+%F %T' > $P/infer_ready
until [ -f $P/train_done ]; do sleep 30; done
log "train_done seen, shutting down"
pkill -f '[v]llm serve'; pkill -f '[l]ora_sync.py'
log "INFER_NODE_DONE"
