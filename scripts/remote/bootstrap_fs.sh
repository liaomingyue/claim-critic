#!/bin/bash
# 新开的 GPU 机器用「自定义镜像 + 文件存储」起步（取代 bootstrap_worker.sh 从主实例经 SSH 拉的做法）：
#   环境：自定义镜像自带；镜像没有时从文件存储的环境包解压（单个大文件，比 rsync 几十万个小文件快得多）
#   代码和数据：直接在文件存储的 code/critic_model 上跑（共用，免拷贝；结果写进去所有机器立刻可见）。
#              本地上传的最新代码包先解压到这里。bash 是边跑边读脚本的，所以角色脚本先拷到本地再执行，
#              防止别的机器同时改代码时打断正在跑的流程；Python 模块启动时已载入，不受影响
#   模型：基座拷到本地盘（实测比直接从文件存储加载快，见下）；只存了适配器、要完整权重的在本机合并，写本地盘
#   结果：outputs 直接写在共享目录里；新训练的适配器写 adapters/；日志存到 runs/<运行名>/
# 不依赖主实例。用法：bash bootstrap_fs.sh <角色> <运行名>。日志 /root/autodl-tmp/worker.log，结束写 WORKER_DONE。
set -u
ROLE=$1 RUN=$2
FS=/root/autodl-fs; A=/root/autodl-tmp; R=$FS/code/critic_model
LOG=$A/worker.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
mkdir -p $A/models $A/ckpt $FS/runs/$RUN
touch $A/.run_start
log "bootstrap_fs start ($ROLE, $RUN)"
[ -d $FS/models ] || { log "file storage not mounted"; exit 1; }
# 认领角色：多台机器（含主动排队的机器）抢同一批任务时，先在共享目录 mkdir 占位（原子操作），占不到就退出
mkdir -p $FS/runs/claims
if ! mkdir $FS/runs/claims/$ROLE 2>/dev/null; then
  log "role $ROLE already claimed by $(cat $FS/runs/claims/$ROLE/by 2>/dev/null)"; log "WORKER_DONE"; exit 0
fi
echo "$(hostname) $RUN $(date '+%F %T')" > $FS/runs/claims/$ROLE/by

# 环境
if ! python -c "import vllm, trl, peft, fla, rapidfuzz" 2>/dev/null; then
  [ -f $FS/env/miniconda3.tar ] || { log "env missing and no env tarball"; exit 1; }
  log "env not in image: unpacking $FS/env/miniconda3.tar"
  rm -rf /root/miniconda3 && tar -xf $FS/env/miniconda3.tar -C /root || { log "env unpack FAILED"; exit 1; }
  python -c "import vllm, trl, peft, fla, rapidfuzz" || { log "env check FAILED"; exit 1; }
fi

# 代码：本地最新代码包（scripts.autodl_extra 上传，不含 outputs）解压到共享目录
if [ -f $A/code_update.tar ]; then
  # 坑：直接解压到共享目录 + sed -i 会把所有脚本换成新文件（内容没变也换），别的机器上正在执行的 bash
  # 读到一半 "error reading input file"（worker6 nat1500 训完 200 步后评测没跑）。改成：先解压到本地、本地去 \r，
  # 再 rsync -c 按内容比对，只替换真正变了的文件（rsync 写临时文件再改名，正在跑的旧文件保持可读）。
  ST=$A/code_stage; rm -rf $ST; mkdir -p $ST
  tar -xf $A/code_update.tar -C $ST || { log "code update FAILED"; exit 1; }
  find $ST/scripts -name '*.sh' -exec sed -i 's/\r$//' {} +
  rsync -rc $ST/ $R/ || { log "code sync FAILED"; exit 1; }
  rm -rf $ST
  # 用完即删：这台机器以后再跑别的角色时，不能把开机时的旧代码再解压一遍、覆盖共享目录里更新的代码（踩过）
  rm -f $A/code_update.tar
fi
mkdir -p $A/scripts && cp $R/scripts/remote/*.sh $A/scripts/   # 本次运行用的脚本快照

# 本角色需要的模型：MODELS = 基座（拷到本地），MERGE = 只存了适配器、要完整权重的（拷适配器后在本机合并到 9B 上）
case $ROLE in
  gentarget3) MODELS=""; MERGE="" ;;
  grpoh2) MODELS=""; MERGE="" ;;
  sevrm) MODELS=""; MERGE="" ;;
  mcgrpo) MODELS=""; MERGE="" ;;
  rewardexp9) MODELS=""; MERGE="" ;;
  bigeval) MODELS=""; MERGE="" ;;
  harvest) MODELS=""; MERGE="" ;;
  clean) MODELS=""; MERGE="" ;;
  refuse) MODELS=""; MERGE="" ;;
  pairinfer|pairtrain) MODELS=""; MERGE="" ;;
  evalmid) MODELS=""; MERGE="" ;;
  core1|loop2co|loop2cont|r2|r3|gen9b|sevfilter|sevgen|sevvar|v3c|gengrpo|scopeno|gengrpoh|genrank|blindfill2|round2fill|round2rag|sevrm2|repro_bf|repro_r1|eval9b|q9bd|q9bs|q9bs2|q9binf2|p9r2inf|p9r2rm|p9r2tr|p9r2i48|p9r2trb|p9r2one) MODELS=""; MERGE="" ;;
  worker1|worker2|worker3|worker4|worker5|worker6|worker7|worker8|worker9) MODELS=""; MERGE="" ;;   # 共享队列 $FS/runs/q2（run_worker.sh），模型脚本自己拷
  smoke)      MODELS="Qwen3.5-9B"; MERGE="" ;;
  *) log "unknown role $ROLE"; exit 1 ;;
esac
# 基座拷到本地盘：顺序拷贝约 100 MB/s（9B 约 3 分钟）；直接从文件存储加载时 transformers 按需随机读，
# 实测只有约 25 MB/s（9B 光加载就要十几分钟）。已是本地目录或软链接的跳过
for m in $MODELS; do
  [ -e $A/models/$m ] && continue
  t0=$(date +%s); rsync -a $FS/models/$m $A/models/ || { log "copy $m FAILED"; exit 1; }
  log "model $m copied in $(( $(date +%s) - t0 ))s"
done
for c in $MERGE; do
  [ -f $A/ckpt/$c/model.safetensors ] && continue   # 本机已有完整权重（排队跑的机器）
  mkdir -p $A/ckpt/$c && rsync -a $FS/adapters/$c/ $A/ckpt/$c/ || { log "copy adapter $c FAILED"; exit 1; }
  (cd $R && python -u -m train.merge_lora --ckpt $A/ckpt/$c --base $A/models/Qwen3.5-9B > $A/merge-$c.log 2>&1) \
    || { log "merge $c FAILED"; exit 1; }
  log "merged $c"
done
log "ready: $(du -sh $A | cut -f1) local"

# 跑本角色（脚本用本地快照，工作目录是共享的代码目录）
S=$A/scripts
cd $R
case $ROLE in
  gentarget3) CODE=$R bash $S/run_gentarget3.sh; rc=$? ;;   # 种子按上下文重新改对 + 指定类型 + RM2 补 2B 真错
  grpoh2) CODE=$R bash $S/run_grpo_h2.sh; rc=$? ;;   # 边界样本 critic（h2）当奖励重跑 2B 回答 GRPO，9B 对比上一版
  sevrm) CODE=$R bash $S/run_sev.sh; rc=$? ;;   # 严重程度 RM（Claude 标的 6 选 1）+ 学习曲线
  mcgrpo) CODE=$R bash $S/run_mcgrpo.sh; rc=$? ;;   # 2B 回答 GRPO（CRITIC / O / ADOUT / STEPS / LR 可用环境变量换）
  rewardexp9) CODE=$R bash $S/run_rewardexp9.sh; rc=$? ;;   # 错误生成器 GRPO（奖励版本见脚本，a7 为当前选定生成器）
  bigeval) CODE=$R bash $S/run_bigeval.sh; rc=$? ;;   # 生成器检查点放大评测
  harvest) CODE=$R bash $S/run_harvest.sh; rc=$? ;;   # 选定生成器造数据 + 边界版 critic 重训
  clean) CODE=$R bash $S/run_clean.sh; rc=$? ;;   # 清洗后的边界样本 critic（h2 / h2n）
  refuse) CODE=$R bash $S/run_refuse.sh; rc=$? ;;   # 拒答标签修正 critic（ref）+ 证据段探针
  # 双机 GRPO：配对名从 runs/pair/CURRENT 读（本地 scripts.pair_setup 写），训练机跑 JOB（默认 run_grpo_h2.sh）
  pairinfer) CODE=$R PAIR=$(cat $FS/runs/pair/CURRENT 2>/dev/null || echo default) bash $S/run_infer_node.sh; rc=$? ;;
  pairtrain) CODE=$R PAIR=$(cat $FS/runs/pair/CURRENT 2>/dev/null || echo default) bash $S/run_train_node.sh; rc=$? ;;
  evalmid) CODE=$R STEP=${STEP:-150} bash $S/run_eval_mid.sh; rc=$? ;;
  core1) CODE=$R bash $S/run_core1.sh; rc=$? ;;   # 核心集第一阶段（README 第一到五步，2B 全流程 + 9B 自检）
  loop2co) CODE=$R bash $S/run_loop2co.sh; rc=$? ;;   # 循环二「只核心集」组（第二台并行）
  loop2cont) CODE=$R bash $S/run_loop2cont.sh; rc=$? ;;
  r3) CODE=$R bash $S/run_r3.sh; rc=$? ;;   # 长句 / 列表句增强 → rm1__core_r3，再用它接着训 100 步
  gen9b) CODE=$R bash $S/run_gen9b.sh; rc=$? ;;   # 同一份种子集训 9B 错误改写模型 + 在 2B 抽检过的提示上采样
  sevfilter) CODE=$R bash $S/run_sevfilter.sh; rc=$? ;;   # sev__core 当过滤器的精确率（对 Claude 审计结论）
  sevgen) CODE=$R bash $S/run_sevgen.sh; rc=$? ;;   # sev 第二版（生成器真实输出 + Claude 统一审计）训练 + 新旧逐条打分
  sevvar) CODE=$R bash $S/run_sevvar.sh; rc=$? ;;   # sev 变体（改动标记 / 不给解释）
  v3c) CODE=$R bash $S/run_v3c.sh; rc=$? ;;   # sev 补范围偏移 + 矛盾放宽（v3c / v3c0）
  gengrpo) CODE=$R bash $S/run_gengrpo.sh; rc=$? ;;   # 生成器 v2 GRPO（sev 奖励，自然度 / critic 门槛），单卡
  scopeno) CODE=$R bash $S/run_scope_noorig.sh; rc=$? ;;   # 范围偏移不配原句
  blindfill2) CODE=$R bash $S/run_blind_fill2.sh; rc=$? ;;   # 补 critic 盲区第二台（g1 采样 + nc 组 critic）
  round2fill) CODE=$R bash $S/run_round2_fill.sh; rc=$? ;;   # 第二轮闭环：新 RAG 回答 → 造难样本 → 等 Claude → rm1__core_rd2
  round2rag) CODE=$R bash $S/run_round2_rag.sh; rc=$? ;;
  eval9b) CODE=$R bash $S/run_eval9b_agree.sh; rc=$? ;;
  q9bd|q9bs|q9bs2) mkdir -p $FS/runs/q9b; [ -f $FS/runs/q9b/tasks ] || cp $R/scripts/remote/q9b.tasks $FS/runs/q9b/tasks; CODE=$R bash $S/queue_worker.sh $FS/runs/q9b; rc=$? ;;   # 9B 对照实验：q9bd 双卡、q9bs 单卡，同一个队列   # 9B 评测器和 Claude 金标的一致性
  q9binf2)   # 9B GRPO 的第二台推理机：和第一台（runs/pair/q9b）用同一把密钥、同一组 LoRA，训练机把采样和打分分摊到两台
    P1=$FS/runs/pair/q9b; P2=$FS/runs/pair/q9b2; mkdir -p $P2; rm -f $P2/infer_ready $P2/infer_failed $P2/train_done $P2/domains.json
    cp $P1/key $P2/key && chmod 600 $P2/key && cp $P1/loras $P2/loras && hostname > $P2/infer_host
    PAIR=q9b2 INFER_MODEL=qwen3.5-9b CODE=$R bash $S/run_infer_node.sh; rc=$? ;;
  p9r2inf) CODE=$R bash $S/q9b_task.sh pair_infer_p9r2; rc=$? ;;   # 9B 策略 × 2B 打分的采样机（只挂 9B 策略）
  p9r2one) CODE=$R bash $S/q9b_task.sh one_p9r2; rc=$? ;;   # 单卡版（大显存卡上训练 + 采样 + 打分）
  p9r2trb) CODE=$R bash $S/q9b_task.sh pair_train_p9r2b; rc=$? ;;   # 两张卡版的训练机（对 48 GB 推理卡，配对 p9r2b；有续训状态就接着训）
  p9r2i48) CODE=$R bash $S/q9b_task.sh pair_infer_p9r2b; rc=$? ;;   # 两张卡版：48 GB 卡同时挂 9B 策略和 2B 打分
  p9r2rm) CODE=$R bash $S/q9b_task.sh pair_rm_p9r2; rc=$? ;;   # 9B 策略 × 2B 打分的打分机（2B critic / 满意度 / 拒答）
  p9r2tr) CODE=$R bash $S/q9b_task.sh pair_train_p9r2; rc=$? ;;   # 9B 策略 × 2B 打分的训练机：训练 + 评测 + 2×2 配对比较
  repro_bf|repro_r1) bash $S/run_repro_test.sh rp_${ROLE#repro_}; rc=$? ;;   # 从头复现实测（scripts/repro），见 run_repro_test.sh
  sevrm2) CODE=$R bash $S/run_sevrm2_r1.sh; rc=$? ;;   # sev / RM2 加第一轮盲区 Claude 标注（sev__v3d / rm2__v3d）
  gengrpoh) CODE=$R bash $S/run_gengrpo_hard.sh; rc=$? ;;   # 生成器 GRPO，critic 难度主奖励（ge / ge2）
  r2) CODE=$R bash $S/run_r2.sh; rc=$? ;;   # Claude 核验过的盲区 / 边界真错 → rm1__core_r2   # 接着第一轮 GRPO（只种子集组第 200 步）用 rm1__core_r1 再训 100 步
  worker4|worker5) CODE=$R bash $S/run_worker2.sh; rc=$? ;;
  worker6|worker7) CODE=$R bash $S/run_worker3.sh; rc=$? ;;   # SFT 上限 / 被钻空子 / nat1500 200 步（q4 队列）
  worker8|worker9) CODE=$R bash $S/run_worker4.sh; rc=$? ;;   # 纯人工 vs 纯合成 critic（q5 队列）
  worker1|worker2|worker3) CODE=$R bash $S/run_worker.sh; rc=$? ;;   # 抢新卡跑队列（h3ctl 评测 / ref / z1500 / S2）   # 训练中途评一版（9B 补打 dev 快评 + 新旧两版同步数验收对比）
  smoke) python -c "import torch; print(torch.cuda.get_device_name(0))" >> $LOG 2>&1; rc=$? ;;
esac
log "role $ROLE exit $rc"

# 存档：本次新产生的结果清单、日志；新训练的适配器写进 adapters/
cd $R && find outputs -type f -newer $A/.run_start > $FS/runs/$RUN/new_outputs.txt
cp $A/*.log $FS/runs/$RUN/ 2>/dev/null
for d in $A/ckpt/*/; do d=${d%/}; n=$(basename $d)
  # 文件存储里没有、或本机的比存档新就存（不按 .run_start：排队重启过的机器会漏存之前训好的，踩过）
  f=$d/adapter/adapter_model.safetensors; [ -f $f ] || continue
  [ -f $FS/adapters/$n/adapter/adapter_model.safetensors ] && [ ! $f -nt $FS/adapters/$n/adapter/adapter_model.safetensors ] && continue
  mkdir -p $FS/adapters/$n && rsync -a $d/adapter $FS/adapters/$n/ \
    && cp $d/run_manifest.json $d/sft_meta.json $FS/adapters/$n/ 2>/dev/null && log "adapter $n saved to file storage"
done
log "$(wc -l < $FS/runs/$RUN/new_outputs.txt) new output files (list in $FS/runs/$RUN/new_outputs.txt)"
[ $rc = 0 ] && touch $FS/runs/$RUN/DONE
log "WORKER_DONE"
