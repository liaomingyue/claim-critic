#!/bin/bash
# 新开的 5090（同地区）从主实例拉环境、模型和数据，然后跑分给它的那部分实验。
# 用法：bash bootstrap_worker.sh <主实例 ssh 主机> <端口> <角色>
#   角色 diag  ：格式诊断（e4、9B 只判定），然后 9B 合成实验的 direct 组（不需要识破率）
#   角色 syn9b ：9B 判识破率，然后 9B 合成实验的 critic / random 组
#   角色 iterrandom：critic 循环迭代实验的随机序列（run_iter.sh random）
#   角色 discrimhead：分类头版本的真实性判别器（run_discrim_head.sh）
#   角色 gancritic：GAN 式两轮循环的 critic 训练与评测（run_gan_critic.sh）
#   角色 realgen：真实幻觉句 SFT 生成器与对比评测（run_realgen.sh），只拉需要的模型
#   角色 closedloop：闭环验证 G0 vs G_free 合成数据训 critic（run_closedloop.sh）
# 免密：本地脚本（scripts.autodl_extra）事先把本机公钥加进主实例的 authorized_keys。
set -u
SRC=$1 PORT=$2 ROLE=$3
LOG=/root/autodl-tmp/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
# --timeout：连接卡住（2026-10-27 一台机器拉到 4.8G 后停住不动、也不报错）时 60 秒没数据就退出；
# --partial + 重试：断点续传，最多 10 次
pull() {  # pull <源路径> <目标路径> [rsync 其他参数]
  for i in $(seq 1 10); do
    rsync -a --delete --partial --timeout=60 -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15" \
      "${@:3}" "root@$SRC:$1" "$2" && return 0
    log "pull $1 attempt $i failed, retry"; sleep 10
  done
  return 1
}

log "pull start ($ROLE)"
mkdir -p /root/autodl-tmp/models /root/autodl-tmp/ckpt
# 整个 conda 环境照搬（同一基础镜像，路径相同），不重装依赖；--delete 顺带去掉镜像里旧 torch 的残留文件
pull /root/miniconda3/ /root/miniconda3/ || { log "pull env FAILED"; exit 1; }
# 只拉本角色用得到的模型（用户要求，2026-10-28：新机器只传必要的东西）
case $ROLE in
  realgen) NEED="models/Qwen3.5-9B models/Qwen3.5-35B-A3B-GPTQ-Int4 ckpt/gen-dpo-r2 ckpt/critic-gan-r2 ckpt/discrim-9b" ;;
  closedloop) NEED="models/Qwen3.5-9B models/Qwen3.5-35B-A3B-GPTQ-Int4 ckpt/gen-sft-free" ;;
  *) NEED="models/Qwen3-0.6B models/Qwen3.5-9B ckpt/sft-9b-v2-notips" ;;
esac
for d in $NEED; do
  pull /root/autodl-tmp/$d/ /root/autodl-tmp/$d/ || { log "pull $d FAILED"; exit 1; }
done
pull /root/autodl-tmp/critic_model/ /root/autodl-tmp/critic_model/ --exclude 'outputs/rt/eval/*/predictions.jsonl' \
  || { log "pull code FAILED"; exit 1; }
# 主实例上的实验日志和结果不属于这台机器，清掉免得跳过逻辑误判
rm -rf /root/autodl-tmp/critic_model/outputs/rt/eval/sft-0.6b-syn-* /root/autodl-tmp/critic_model/outputs/rt/eval/sft-0.6b-e4 \
       /root/autodl-tmp/critic_model/outputs/rt/eval/sft-9b-verdict /root/autodl-tmp/critic_model/outputs/rt/eval/sft-9b-syn-*        /root/autodl-tmp/critic_model/outputs/rt/eval/sft-9b-iter-* /root/autodl-tmp/critic_model/outputs/rt/iter
log "pull done: $(du -sh /root/autodl-tmp | cut -f1)"
python -c "import vllm, trl, peft, fla, rapidfuzz" || { log "env check FAILED"; exit 1; }

S=/root/autodl-tmp/critic_model/scripts/remote
case $ROLE in
  diag)  NOWAIT=1 bash $S/run_diag_exp.sh; NOWAIT=1 ARMS=direct bash $S/run_synth9b_exp.sh ;;
  syn9b) NOWAIT=1 ARMS="critic random" bash $S/run_synth9b_exp.sh ;;
  iterrandom) POOL_SRC=$SRC:$PORT NOWAIT=1 bash $S/run_iter.sh random ${ROUNDS:-3} ;;   # 迭代实验的随机序列，合成池从主实例拉
  discrimhead) bash $S/run_discrim_head.sh $SRC $PORT ;;   # 分类头版本的真实性判别器，结果同步回主实例
  gancritic) bash $S/run_gan_critic.sh $SRC $PORT ;;   # GAN 式循环的 critic 部分，与主实例的生成器 DPO 并行
  closedloop) bash $S/run_closedloop.sh   # 闭环验证：G0 vs G_free 合成数据各训一个 critic；结果和 critic 适配器推回主实例
    R=/root/autodl-tmp/critic_model; E="ssh -p $PORT -o StrictHostKeyChecking=no"; CK=/root/autodl-tmp/ckpt
    for n in g0 gfree; do $E root@$SRC "mkdir -p $CK/critic-closed-$n"; done
    rsync -a -e "$E" $R/outputs/rt/closed root@$SRC:$R/outputs/rt/ &&     rsync -a -e "$E" $R/outputs/rt/eval/sft-9b-critic-closed-g0 $R/outputs/rt/eval/sft-9b-critic-closed-gfree root@$SRC:$R/outputs/rt/eval/ &&     rsync -a -e "$E" $R/outputs/rt/claims/closed_g0.jsonl $R/outputs/rt/claims/closed_gfree.jsonl root@$SRC:$R/outputs/rt/claims/ &&     for n in g0 gfree; do rsync -a -e "$E" $CK/critic-closed-$n/adapter $CK/critic-closed-$n/run_manifest.json $CK/critic-closed-$n/sft_meta.json root@$SRC:$CK/critic-closed-$n/ || exit 1; done &&     rsync -a -e "$E" $R/outputs/runs/registry.jsonl root@$SRC:$R/outputs/runs/registry_closedloop.jsonl && log "pushed to main"       || log "push FAILED" ;;
  realgen) bash $S/run_realgen.sh   # 真实幻觉句 SFT 生成器 + 对比评测（主实例无卡开机只当数据源）；结果和新模型推回主实例
    R=/root/autodl-tmp/critic_model; E="ssh -p $PORT -o StrictHostKeyChecking=no"
    rsync -a -e "$E" $R/outputs/rt/realgen root@$SRC:$R/outputs/rt/ &&     rsync -a -e "$E" /root/autodl-tmp/ckpt/gen-sft-real root@$SRC:/root/autodl-tmp/ckpt/ &&     rsync -a -e "$E" $R/outputs/runs/registry.jsonl root@$SRC:$R/outputs/runs/registry_realgen.jsonl && log "pushed to main"       || log "push FAILED" ;;
esac
log "WORKER_DONE"
