#!/bin/bash
# 正式回答 GRPO（把 R3 作为新的奖励设定，正式跑完整的回答 GRPO）。单卡：策略 vLLM（0.35 显存）+ RM LoRA + 训练都在 GPU0。
# 进奖励：rm1__mcgold_h3ctl（幻觉：去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3）、sat_pair__mc（满意度只罚 0.8 ×，最多 0.4）、
#         refuse（拒答 0.4 × P(可拒答)）、硬规则。只监控：9B 金标 critic（训完对检查点独立评测，不进奖励）。
# mc-init 起，lr 5e-6，n 16 × 16 prompt，STEPS（默认 300）步，每 50 步存检查点 + 续训状态，TripWire 打开。
# 训完：第 100 / 200 / 300 步在 200 个验收原文-种子上生成（同 run_after：grpo300 T1.0、grpo300b T0.7，top_p 0.95），9B 金标 critic 打分 → $O/eval/
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
A=${TMP_ROOT:-/root/autodl-tmp}; FS=${FS_ROOT:-/root/autodl-fs}; LOG=$A/r3full${TAG:-}.log; S=scripts/remote
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
# 看门狗保活（拷 / 加载 9B 时 GPU 空闲超过 10 分钟会被平台关机）：脚本活着就每分钟 touch，脚本结束循环自己退出
( while kill -0 $$ 2>/dev/null; do touch ${TMP_ROOT:-/root/autodl-tmp}/KEEPALIVE; sleep 60; done ) & KA=$!   # 保活进程 PID：下面的 wait 不能等它（踩过：worker1 生成完卡死在 wait）
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 双机模式（两台单卡：run_train_node.sh 设好 RM_URL / LORA_SYNC_URL / VLLM_API_KEY）：打分和采样都走推理机，本机只训练；否则清掉这些变量
[ -n "${RM_URL:-}" ] || unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
# 2B 底座只在本机要用时才拷：策略是 2B，或打分模型是 2B 且不在推理机上（多机 9B 策略 × 2B 打分时训练机用不到 2B）
if [ ${POLICY_MODEL:-minicpm5-2b} = minicpm5-2b ] || { [ ${RM_MODEL:-minicpm5-2b} = minicpm5-2b ] && [ -z "${RM_URL:-}" ]; }; then
  [ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
fi
source scripts/remote/models9b.sh; start_9b_bg   # 先拷马上要用的 2B，9B 后台拷，把时间岔开
# 9B 对照（2×2：策略大小 × 打分模型大小）：POLICY_MODEL / RM_MODEL = minicpm5-2b（默认）| qwen3.5-9b。
#   打分模型（critic / 满意度 / 拒答）一起换大小；9B 版的名字带 _9b 后缀（数据目录 outputs/rt/rm/<名字>，adapter <名字>-9b）。
#   GPU 布局：都是 2B 时和原来一样全在 GPU0；否则训练在 GPU0，9B 的服务放 GPU1，2B 策略的采样服务留在 GPU0。
PM=${POLICY_MODEL:-minicpm5-2b}; RMM=${RM_MODEL:-minicpm5-2b}
B9=$A/models/Qwen3.5-9B
{ [ $PM = qwen3.5-9b ] || [ $RMM = qwen3.5-9b ]; } && { ensure_9b || { log "9b copy FAILED"; exit 1; }; }
# CAUSAL：MiniCPM5-2B 是纯文本 CausalLM；Qwen3.5-9B 的 LoRA 按视觉语言结构保存（参数名带 language_model），要用 load_policy 加载，否则一个参数都对不上
if [ $PM = qwen3.5-9b ]; then PB=$B9; PSFX=9b; SHARE_IMPL=train.prefix_share; CAUSAL=0; else PB=$A/models/MiniCPM5-2B; PSFX=mc; SHARE_IMPL=train.prefix_share_llama; CAUSAL=1; fi
if [ $RMM = qwen3.5-9b ]; then RB=$B9; RS=9b; else RB=$A/models/MiniCPM5-2B; RS=mc; fi
MC=$PB; SRC=${SRC:-outputs/rt/minicpm/grpo}; INIT=${INIT:-$FS/adapters/mc-init}; CRITIC=${CRITIC:-rm1__mcgold_h3ctl}; STEPS=${STEPS:-300}
SAT=${SAT:-sat_pair__mc}; REFUSE=${REFUSE:-refuse}   # 循环二：起点 / 原文池 / 满意度 / 拒答模型可换（核心集版：INIT=rag-sft-core SAT=sat_pair__core REFUSE=refuse__core）
TAG=${TAG:-}; O=${O:-outputs/rt/minicpm/r3full$TAG}; E=$O/eval; mkdir -p $E   # TAG：换 critic 的对照组（两个显著不同的 RM 各用 R3 训 RAG）
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
RW="--reward_shape count --count_ratio 0.5 --count_min 2 --sat_w 0.8 --sent_beta 0.6 --sent_gamma 0.3"
free_gpu; mkdir -p $A/s0 $A/s1
RMS="critic=$FS/adapters/$CRITIC-$RS/adapter sat=$FS/adapters/$SAT-$RS/adapter refuse=$FS/adapters/$REFUSE-$RS/adapter"
POLURL=http://127.0.0.1:8000/v1; ASYNC=""
if [ -n "${RM_URL:-}" ] && [ $PM = minicpm5-2b ]; then   # 多机，2B 策略 × 9B 打分：打分在推理机上，2B 策略的采样服务留在训练卡（同单卡 2B 的布局），LoRA 本机热加载
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${MC_MEM:-0.35} --base $PB --log $A/vllm-r3full.log --priority policy_v0=$INIT >> $LOG 2>&1 || { log "serve policy FAILED"; exit 1; }
  RMURL=$RM_URL; POLURL=http://127.0.0.1:8000/v1; ASYNC=""
  log "remote scoring $RM_URL, local 2B policy"
elif [ -n "${RM_URL:-}" ]; then   # 双机：推理机上已经挂好策略 + 打分模型（run_infer_node.sh），LoRA 每步后台上传给它热加载
  RMURL=$RM_URL; POLURL=${POLICY_URL:-$RM_URL}; ASYNC="--async_push 1"   # POLICY_URL：策略采样在另一组推理机上（单独的打分机，run_train_node.sh RM_PAIR）
  log "remote inference $RM_URL"
elif [ $PM = minicpm5-2b ] && [ $RMM = minicpm5-2b ]; then   # 原布局：一个服务，GPU0
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${MC_MEM:-0.35} --base $MC --log $A/vllm-r3full.log --priority $RMS policy_v0=$INIT >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
  RMURL=http://127.0.0.1:8000/v1
elif [ $PM = $RMM ]; then   # 9B × 9B：同一基座，一个服务放 GPU1（策略 + 打分模型），训练独占 GPU0
  A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8000 --mem 0.85 --base $PB --log $A/vllm-r3full.log --priority $RMS policy_v0=$INIT >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
  RMURL=http://127.0.0.1:8000/v1
elif [ $PM = minicpm5-2b ]; then   # 2B 策略 × 9B 打分：策略服务和训练同在 GPU0（同原布局），9B 打分服务放 GPU1
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${MC_MEM:-0.35} --base $PB --log $A/vllm-r3full.log --priority policy_v0=$INIT >> $LOG 2>&1 || { log "serve policy FAILED"; exit 1; }
  A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8001 --mem 0.85 --base $RB --log $A/vllm-r3rm.log --priority $RMS >> $LOG 2>&1 || { log "serve rm FAILED"; exit 1; }
  RMURL=http://127.0.0.1:8001/v1
else   # 9B 策略 × 2B 打分：两个服务都在 GPU1（9B 策略 0.55 + 2B 打分 0.3），训练独占 GPU0
  A=$A/s1 bash $S/serve_loras.sh --gpu 1 --port 8000 --mem 0.55 --base $PB --log $A/vllm-r3full.log --priority policy_v0=$INIT >> $LOG 2>&1 || { log "serve policy FAILED"; exit 1; }
  A=$A/s0 bash $S/serve_loras.sh --gpu 1 --port 8001 --mem 0.3 --base $RB --log $A/vllm-r3rm.log --priority $RMS >> $LOG 2>&1 || { log "serve rm FAILED"; exit 1; }
  RMURL=http://127.0.0.1:8001/v1
fi
LAST=$(ls -d $O/run/state/step* 2>/dev/null | sort -V | tail -n 1)
log "start r3full: critic $CRITIC, $RW, steps $STEPS ${LAST:+(resume $LAST)}"
CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo --causal $CAUSAL --sys_qa 1 --base $MC --init $INIT --vllm_url $RMURL --policy_url $POLURL $ASYNC \
  --n 16 --share 1 --share_impl $SHARE_IMPL --share_ckpt 1 --share_mb ${SHARE_MB:-32} --prompts $SRC/train_ref.jsonl --dev $SRC/dev_ref.jsonl \
  --critic_report outputs/rt/rm/$CRITIC/eval/report.json --shadow= \
  --sat_model sat --sat_kind pair --sat_report outputs/rt/rm/$SAT/eval/report.json --refuse_model refuse --refuse_report outputs/rt/rm/$REFUSE/eval/report.json --refuse_w 0.4 \
  $RW --proc_producer 1 --pipeline 1 --lr 5e-6 --steps $STEPS --save_every 50 --state_every 50 --keep_states 2 --eval_k 2 --trip 1 \
  --version_prefix r3f${TAG}_ --push_dir /dev/shm/r3full$TAG --out $O/run ${LAST:+--resume $LAST} >> $A/r3full${TAG}_train.log 2>&1 && log "train done" || log "train FAILED or stopped (see r3full_train.log)"
grep -ah "QUICK_EVAL\|TRIP" $A/r3full${TAG}_train.log | cut -c1-500 | tee -a $LOG
mkdir -p $FS/adapters/$PSFX-grpo-r3$TAG && rsync -a $O/run/ckpt/ $FS/adapters/$PSFX-grpo-r3$TAG/ && log "ckpts saved to adapters/$PSFX-grpo-r3$TAG: $(ls $O/run/ckpt | tr '\n' ' ')"
free_gpu
# 评测在本机做（训练卡此时空了）：双机模式下切回本机服务，不带密钥
if [ -n "${RM_URL:-}" ]; then unset RM_URL LORA_SYNC_URL VLLM_API_KEY PUSH_DIR; export OPENAI_API_KEY=x; fi

# 独立评测
POL=""; [ "${EVAL_V0:-1}" = 1 ] && POL="v0=$INIT"; for s in ${EVAL_STEPS:-100 200 300}; do [ -d $O/run/ckpt/step$s ] && POL+=" s$s=$(ad $O/run/ckpt/step$s)"; done
POL=${POL# }; [ -n "$POL" ] || { log "no ckpts to eval"; exit 1; }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-r3gen.log $POL >> $LOG 2>&1 || { log "serve policies FAILED"; exit 1; }
for kv in $POL; do m=${kv%%=*}
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_a.jsonl >> $LOG 2>&1 &
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_b.jsonl >> $LOG 2>&1 &
  # 第三套（扩大评测量，两个 RM 的差异可能比 R3 对 R0 小）：132 篇新原文 × 3 种子，温度 0.7
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300c/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_c.jsonl >> $LOG 2>&1 &
done
for p in $(jobs -p); do [ "$p" = "$KA" ] || wait $p; done; free_gpu
C9=rm1__rm1_gold_9b; T=$(temp $C9)
ensure_9b || { log "9b copy FAILED"; exit 1; }   # 后台拷（start_9b_bg）还没完就等它
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $A/models/Qwen3.5-9B --log $A/vllm-r39b.log critic=$FS/adapters/$C9-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
jobs="["; for kv in $POL; do m=${kv%%=*}; for s in ${SETS:-a b c}; do jobs+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${m}_$s.jsonl\",\"out\":\"$E/${m}_${s}_9b.json\"},"; done; done
echo "${jobs%,}]" > $E/jobs9b.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $E/jobs9b.json --concurrency 256 >> $LOG 2>&1 || log "9b score FAILED"
free_gpu
log "R3FULL_DONE"
