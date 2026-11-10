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
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg   # 先拷马上要用的 2B，9B 后台拷，把时间岔开
MC=$A/models/MiniCPM5-2B; SRC=${SRC:-outputs/rt/minicpm/grpo}; INIT=${INIT:-$FS/adapters/mc-init}; CRITIC=${CRITIC:-rm1__mcgold_h3ctl}; STEPS=${STEPS:-300}
SAT=${SAT:-sat_pair__mc}; REFUSE=${REFUSE:-refuse}   # 循环二：起点 / 原文池 / 满意度 / 拒答模型可换（核心集版：INIT=rag-sft-core SAT=sat_pair__core REFUSE=refuse__core）
TAG=${TAG:-}; O=${O:-outputs/rt/minicpm/r3full$TAG}; E=$O/eval; mkdir -p $E   # TAG：换 critic 的对照组（两个显著不同的 RM 各用 R3 训 RAG）
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
RW="--reward_shape count --count_ratio 0.5 --count_min 2 --sat_w 0.8 --sent_beta 0.6 --sent_gamma 0.3"
free_gpu; mkdir -p $A/s0
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${MC_MEM:-0.35} --base $MC --log $A/vllm-r3full.log --priority \
  critic=$FS/adapters/$CRITIC-mc/adapter sat=$FS/adapters/$SAT-mc/adapter refuse=$FS/adapters/$REFUSE-mc/adapter policy_v0=$INIT >> $LOG 2>&1 || { log "serve FAILED"; exit 1; }
LAST=$(ls -d $O/run/state/step* 2>/dev/null | sort -V | tail -n 1)
log "start r3full: critic $CRITIC, $RW, steps $STEPS ${LAST:+(resume $LAST)}"
CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo --causal 1 --sys_qa 1 --base $MC --init $INIT --vllm_url http://127.0.0.1:8000/v1 --policy_url http://127.0.0.1:8000/v1 \
  --n 16 --share 1 --share_impl train.prefix_share_llama --share_ckpt 1 --share_mb ${SHARE_MB:-32} --prompts $SRC/train_ref.jsonl --dev $SRC/dev_ref.jsonl \
  --critic_report outputs/rt/rm/$CRITIC/eval/report.json --shadow= \
  --sat_model sat --sat_kind pair --sat_report outputs/rt/rm/$SAT/eval/report.json --refuse_model refuse --refuse_report outputs/rt/rm/$REFUSE/eval/report.json --refuse_w 0.4 \
  $RW --proc_producer 1 --pipeline 1 --lr 5e-6 --steps $STEPS --save_every 50 --state_every 50 --keep_states 2 --eval_k 2 --trip 1 \
  --version_prefix r3f${TAG}_ --push_dir /dev/shm/r3full$TAG --out $O/run ${LAST:+--resume $LAST} >> $A/r3full${TAG}_train.log 2>&1 && log "train done" || log "train FAILED or stopped (see r3full_train.log)"
grep -ah "QUICK_EVAL\|TRIP" $A/r3full${TAG}_train.log | cut -c1-500 | tee -a $LOG
mkdir -p $FS/adapters/mc-grpo-r3$TAG && rsync -a $O/run/ckpt/ $FS/adapters/mc-grpo-r3$TAG/ && log "ckpts saved to adapters/mc-grpo-r3$TAG: $(ls $O/run/ckpt | tr '\n' ' ')"
free_gpu

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
