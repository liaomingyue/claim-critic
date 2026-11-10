#!/bin/bash
# 接着第一轮 GRPO 训，看能否再提升。起点 = 只种子集组第 200 步（adapters/mc-grpo-r3_core_coreonly/step200），
# 奖励 critic 换成循环一重训的 rm1__core_r1（在第一轮剩下的 9B 显然错上召回 25% → 37%），其余同循环二：R3 奖励、lr 5e-6、n 16 × 16。
# 100 步；起点重采参考回答（长度范围、满意度参考、K），KL 相对起点。评测：起点 + 第 50 / 100 步在三套验收集上生成，9B 金标 critic 打分。
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; LOG=$A/loop2cont.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch ${TMP_ROOT:-/root/autodl-tmp}/KEEPALIVE; sleep 60; done ) &
source scripts/remote/model_env.sh minicpm5-2b && log "2B ready"
source scripts/remote/models9b.sh; start_9b_bg
D=${START:-$FS/adapters/mc-grpo-r3_core_coreonly/step200}   # 第二次接着训：START=adapters/mc-grpo-r3_core_cont/step100 CRITIC=rm1__core_r2 ARM=cont2
INIT=$D; [ -f $D/policy/adapter_model.safetensors ] && INIT=$D/policy
[ -f $INIT/adapter_model.safetensors ] || { log "missing start adapter $INIT"; exit 1; }
log "continue from $INIT"
ARM=${ARM:-cont} CRITIC=${CRITIC:-rm1__core_r1} INIT=$INIT STEPS=100 EVAL_STEPS="50 100" EVAL_V0=1 CODE=$PWD bash $(dirname $(readlink -f $0))/run_loop2.sh
log "LOOP2CONT_DONE"
