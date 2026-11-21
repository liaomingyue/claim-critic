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
  core1|loop2co|loop2cont|r2|r3|gen9b|sevfilter|sevgen|sevvar|v3c|gengrpo|scopeno|gengrpoh|genrank|blindfill2|round2fill|round2rag|sevrm2|repro_bf|repro_r1|satw|satdyn|cont8dyn|cont9dyn|ownfill|attrl|attrt|spkl|c9base|c9spk|c9blind|g9sev|g9nat|g9opm|g9disc|gog9|gog9i|gog9t|gog9i2|gog9t2|gog9i3|gog9t3|gog9i4|gog9t4|of9|q9ci|q9ct|q9ci2|q9ct2|g5sev|g5nat|g5opm|gr2i|gr2t|corag|eval9b|q9bd|q9bs|q9bs2|q9binf2|p9r2inf|p9r2rm|p9r2tr|p9r2i48|p9r2trb|p9r2one|suitans|synthsuit|mech|s2r9rm|s2r9tr|genops|genopsvar|genopsprompt|rmops|rmops2|gogrpoc|gost4|rmopc|gost6|gost7|gost8|gost9|opsfill|rags5|opsfillb|of2) MODELS=""; MERGE="" ;;
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
  suitans)   # 合适度模型对照：输入再加一条正确回答（数据 outputs/rt/suit/*_ans.jsonl 随代码包 --with 上传），训完评测，结果写回共享盘
    ( cd $R && export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && mkdir -p $A/models       && { [ -f $A/models/MiniCPM5-2B/config.json ] || rsync -a $FS/models/MiniCPM5-2B $A/models/; }       && python -u -m train.suit_train --base $A/models/MiniCPM5-2B --out $A/suit_ans --with_answer       && mkdir -p $FS/adapters/suit_ans-mc outputs/rt/suit/suit_ans && rsync -a $A/suit_ans/adapter $FS/adapters/suit_ans-mc/       && cp $A/suit_ans/*.json* outputs/rt/suit/suit_ans/ && echo SUIT_ANS_SAVED ); rc=$? ;;
  synthsuit)   # 合成适合度模型（train.synth_suit_train）：数据 outputs/rt/synth_suit 随代码包 --with 上传，训完在留出集上评测，结果写回共享盘
    ( cd $R && export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && mkdir -p $A/models       && { [ -f $A/models/MiniCPM5-2B/config.json ] || rsync -a $FS/models/MiniCPM5-2B $A/models/; }       && python -u -m train.synth_suit_train --base $A/models/MiniCPM5-2B --out $A/synth_suit       && mkdir -p $FS/adapters/synth_suit-mc outputs/rt/synth_suit/run && rsync -a $A/synth_suit/adapter $FS/adapters/synth_suit-mc/       && cp $A/synth_suit/*.json* outputs/rt/synth_suit/run/ && echo SYNTH_SUIT_SAVED ); rc=$? ;;
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
  mech) CODE=$R bash $S/run_mech.sh; rc=$? ;;
  s2r9rm)   # RAG-S2 + 9B 奖励的打分机：9B 底座 + 9B critic / 满意度 / 拒答（配对 runs/pair/s2r9；domains.json 由本地 autodl_extra 建实例时写，这里不删）
    P=$FS/runs/pair/s2r9; mkdir -p $P; rm -f $P/infer_ready $P/infer_failed $P/train_done $P/train_failed $P/extra_pairs $P/rm_loras
    hostname > $P/infer_host
    printf '%s\n' "critic=$FS/adapters/rm1__core_9b-9b/adapter" "sat=$FS/adapters/sat_pair__core_9b-9b/adapter" "refuse=$FS/adapters/refuse__core_9b-9b/adapter" > $P/loras
    rc=0; for f in $(sed "s|.*=||" $P/loras); do [ -f $f/adapter_model.safetensors ] || { log "missing adapter $f"; touch $P/infer_failed; rc=1; }; done
    [ $rc = 0 ] && { PAIR=s2r9 INFER_MODEL=qwen3.5-9b CODE=$R bash $S/run_infer_node.sh; rc=$?; } ;;
  s2r9tr) PRECOPY_2B=1 PAIR=s2r9 JOB=run_s2r9_pair.sh CODE=$R bash $S/run_train_node.sh; rc=$? ;;   # 训练机：2B 策略采样 + 训练，训完本机评测   # critic 当奖励的机制分析（各起点采样 + 历次 critic 打分 + 9B 金标）
  genops) CODE=$R bash $S/run_genops.sh; rc=$? ;;   # 按错误操作清单下指令的错误合成模型：训练 + 留出回答上采样
  genopsvar) CODE=$R bash $S/run_genops_var.sh; rc=$? ;;   # 同上，提示词（附例子 / 指令位置）和训练设置对照
  genopsprompt) CODE=$R bash $S/run_genops_prompt.sh; rc=$? ;;   # 只靠提示词的错误合成：2B / 9B 基座 × 三种提示词写法
  rmops2) ORDER="rm2__v4 opm__v1 sev__v4" CODE=$R bash $S/run_rmops.sh; rc=$? ;;   # 第二台单卡机器：先训自然度
  rmops) ORDER="sev__v4 opm__v1 rm2__v4" CODE=$R bash $S/run_rmops.sh; rc=$? ;;   # 给错误合成把关的三个 2B 模型（sev__v4 / rm2__v4 / opm__v1），两张卡并行
  gogrpoc) E=s1c STAGE=1 STYLE=free LR=2e-5 STEPS=100 N=16 PPS=16 CODE=$R bash $S/run_genops_grpo.sh; rc=$? ;;   # 生成器 GRPO 阶段一，奖励里带 critic 的「不明显」系数
  rmopc) CODE=$R bash $S/run_rmopc.sh; rc=$?   # 事后分类模型（错在哪一层、哪一种操作），训完给各版生成器的留出样本分类。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gost4) CODE=$R bash $S/run_genops_st4.sh; rc=$?   # 生成器 GRPO 阶段三再训一轮：明显错系数 + 组内多样性。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gost6) CODE=$R bash $S/run_genops_st6.sh; rc=$?   # 生成器 GRPO 多样性再训一轮：双向的占比系数。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gost7) CODE=$R bash $S/run_genops_st7.sh; rc=$?   # 生成器 GRPO 多样性再训一轮：目标换成真实错误的分布。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gost8) CODE=$R bash $S/run_genops_st8.sh; rc=$?   # 生成器 GRPO 多样性再训一轮：近义词并组、操作系数。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gost9) E=st9 XTRA="--ops_div_group 0.5 --ops_div_all 0.3" CODE=$R bash $S/run_genops_st8.sh; rc=$?   # 生成器 GRPO：在 st8 的设置上加两项多样性（操作一致的样本之间按手段节点算组内多样性；没有真实目标的操作用累计分布）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  opsfill) CODE=$R bash $S/run_ops_fill.sh; rc=$?   # 闭环后半段：合成错误的盲区样本补进 critic 重训（中间等本地 Claude 核实）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  rags5) CODE=$R bash $S/run_rag_s5.sh; rc=$?   # 补过盲区的两个新 critic 当奖励，从 RAG-S3-bf 接着训 RAG（先后两组）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  opsfillb) CODE=$R bash $S/run_ops_fill_b.sh; rc=$?   # 针对性补采：挂接层和直接矛盾对应的 10 种操作，每个任务采 6 条并打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  satw) ARMS="satw08=rm1__core_r2_bfcl:0.8 satw02=rm1__core_r2_bfcl:0.2" CODE=$R bash $S/run_rag_s5.sh; rc=$?   # 只改满意度扣分的权重（0.8 对 0.2），critic 用基线，从 RAG-S3-bf 接着训。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  satdyn) ARMS="satdyn=rm1__core_r2_bfcl:0.8" RW_EXTRA="--sat_s0 0.1 --adv_std_floor 0.15" CODE=$R bash $S/run_rag_s5.sh; rc=$?   # 和 satw08 只差两处：满意度权重按组里 critic 的区分度缩放、组标准差加下限（critic 没信号的组少更新）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  cont8dyn) ARMS="cont8dyn=rm1__core_of1all:0.8" RW_EXTRA="--sat_s0 0.1 --adv_std_floor 0.15" CODE=$R bash $S/run_rag_s5.sh; rc=$?   # 和 cont8 只差奖励的两处（满意度权重按组缩放、组标准差下限），critic 同为盲区样本补过的那版。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  cont9dyn) ARMS="cont9dyn=rm1__core_of1cl:0.8 cont8dyn=rm1__core_of1all:0.8" RW_EXTRA="--sat_s0 0.1 --adv_std_floor 0.15" CODE=$R bash $S/run_rag_s5.sh; rc=$?   # 先核实过的那版 critic，再全量未核实的那版；都只比各自的上一组多奖励的两处改动。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  ownfill) CODE=$R bash $S/run_own_fill.sh; rc=$?   # 用 RAG 模型自己写的、核实过的句子补 critic（只用验收原文的一半），训两个 critic 并给已标注的回答打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  attrl) ARM=l CODE=$R bash $S/run_attr_fill.sh; rc=$?   # 专项补张冠李戴：按规则构造的错句 + 原句，只给标签。训一个 critic 并给已标注的回答打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  attrt) ARM=t CODE=$R bash $S/run_attr_fill.sh; rc=$?   # 同一批句子，输出改成先写核对过程再给结论。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  spkl) ARM=l RD=sp1 CODE=$R bash $S/run_attr_fill.sh; rc=$?   # 专项补说话人 / 出处错位（ragtruth/spk_fill.py 按规则构造），只给标签。训一个 critic 并给已标注的回答打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  c9base) SRC=rm1__core_r2_bfcl CODE=$R bash $S/run_crit9b.sh; rc=$?   # 基线 critic 训练集换 9B 底座训一遍并给已标注的回答打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  c9spk) SRC=rm1__core_sp1l CODE=$R bash $S/run_crit9b.sh; rc=$?   # 同上，训练集多了说话人错位的补料
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  c9blind) rc=0; for x in rm1__core rm1__core_r2_bfcl rm1__core_sp1l; do   # 找 9B critic 在 9B RAG 回答上的盲区：三个已训好的 9B critic 给 9B 策略的验收回答打分（只打分）。跑完留 20 分钟取结果，然后关机
       NOTRAIN=1 SETS="q9b_p9r9:s200 q9b_p9r2:s200" SRC=$x CODE=$R bash $S/run_crit9b.sh || rc=1; done
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  g9sev|g9nat|g9opm|g9disc)   # 给错误合成把关的模型换成 9B 底座各训一遍（数据同 2B 版）：真错和严重度 / 自然度 / 操作一致 / 判别器。一台一个，只训练。跑完留 20 分钟，然后关机
     case $ROLE in g9sev) x=sev__v4 ;; g9nat) x=rm2__v4 ;; g9opm) x=opm__v1 ;; g9disc) x=rm2__d2 ;; esac
     SETS=none SRC=$x CODE=$R bash $S/run_crit9b.sh; rc=$?
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9) CODE=$R bash $S/run_genops_grpo9.sh; rc=$?   # 错误合成模型换成 9B 的 GRPO 阶段一（双卡：卡 0 训练，卡 1 采样和把关），前后都在留出回答上采样打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9i) CODE=$R bash $S/run_genops_infer9.sh; rc=$?   # 9B 错误合成 GRPO，两台单卡布局的推理机（采样 + 把关打分）。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9t) CODE=$R PAIR=gog9 JOB=run_genops_grpo9.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机：训练前后的留出采样打分在本机做，训练时采样和打分走推理机。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9i2) CODE=$R bash $S/run_genops_infer9.sh; rc=$?   # 同 gog9i，重跑用的新角色名（参考 logprob 改到训练机算、不挂判别器）
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9t2) CODE=$R E=s1c9b PAIR=gog9 JOB=run_genops_grpo9.sh bash $S/run_train_node.sh; rc=$?   # 同 gog9t，重跑（运行名 s1c9b）
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9i3) DISC=1 INIT=$FS/adapters/gen-ops-grpo-s1c9b/step100/policy CODE=$R bash $S/run_genops_infer9.sh; rc=$?   # 9B 错误合成 GRPO 阶段二的推理机（挂判别器，起点 = 第一段终点）。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9t3) E=s2c9b CODE=$R PAIR=gog9 JOB=run_genops_st2_9.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机（run_genops_st2_9.sh）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9i4) DISC=1 INIT=$FS/adapters/gen-ops-grpo-s2c9b/step100/policy CODE=$R bash $S/run_genops_infer9.sh; rc=$?   # 9B 错误合成 GRPO 阶段三、四的推理机（起点 = 第二段终点）。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gog9t4) CODE=$R PAIR=gog9 JOB=run_genops_st34_9.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机（run_genops_st34_9.sh：阶段三 → 多样性那一段）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  of9) GEN_E=s4c9 CODE=$R bash $S/run_ops_fill9.sh; rc=$?   # 全 9B 链路第 3 步：9B 错误合成模型造盲区样本 → 补进 critic，9B 底座重训 → 给已标注的回答打分。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  q9ci) CODE=$R bash $S/q9b_task.sh pair_infer_c; rc=$?   # 全 9B 链路第 4 步的推理机（9B 底座 + 新 critic / 满意度 / 拒答 + RAG 策略第 200 步）。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  q9ct) CODE=$R PAIR=q9c JOB=run_q9b_cont.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机：从第 200 步接着训 100 步（奖励换新 critic）+ 第 50 / 100 步验收评测。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  q9ci2) PAIR_C=q9c2 CRIT_C=rm1__core_r2_bfcl_9b CODE=$R bash $S/q9b_task.sh pair_infer_c; rc=$?   # 第 4 步的对照组推理机：奖励 critic 用补样本前的基线 critic，其余同 q9ci。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  q9ct2) CRIT_C=rm1__core_r2_bfcl_9b CRIT_TAG=r2b STEPS_C=50 EVAL_C=50 CODE=$R PAIR=q9c2 JOB=run_q9b_cont.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机：从第 200 步接着训 50 步 + 第 50 步验收评测，和新 critic 那组的第 50 步比。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  g5sev) WHICH=sev CODE=$R bash $S/run_gate9v5.sh; rc=$?   # 第二轮：9B 把关模型（严重程度）加 Claude 标签重训。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  g5nat) WHICH=nat CODE=$R bash $S/run_gate9v5.sh; rc=$?   # 第二轮：9B 把关模型（自然度）加 Claude 标签重训。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  g5opm) WHICH=opm CODE=$R bash $S/run_gate9v5.sh; rc=$?   # 第二轮：9B 把关模型（类别一致）加 Claude 标签重训。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gr2i) PAIR=gog9r2 DISC=1 G_SEV=sev__v5 G_NAT=rm2__v5 G_OPM=opm__v2 G_CR=rm1__core_of9all INIT=$FS/adapters/gen-ops-grpo-s4c9/step100/policy CODE=$R bash $S/run_genops_infer9.sh; rc=$?   # 第二轮错误合成 GRPO 的推理机（新把关模型 + 新 critic，起点 = 第一轮终点）。训练机写 train_done 后退出，留 20 分钟，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  gr2t) CODE=$R PAIR=gog9r2 JOB=run_genops_r2_9.sh bash $S/run_train_node.sh; rc=$?   # 同上，训练机（run_genops_r2_9.sh：重打分定基准线 → 阶段三 → 阶段四）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  of2) CODE=$R bash $S/run_of2.sh; rc=$?   # 第二轮补 critic：效果差的类别多采样补量 → 重训两个 critic → 各自接着训 RAG。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
  corag) CODE=$R bash $S/run_critic_on_rag.sh; rc=$?   # 定位：补过样本的 critic 训出来的 RAG 模型自己写错的句子，critic 认不认得出（只打分）。跑完留 20 分钟取结果，然后关机
     nohup bash -c 'sleep 1200; /usr/bin/shutdown' > /dev/null 2>&1 & ;;
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
