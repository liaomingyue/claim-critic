#!/bin/bash
# MiniCPM5-2B 回答 GRPO（2026-11-02，用户：两张卡都放 vLLM；关闭思考 + QA 系统提示）。
#   GPU0：MiniCPM vLLM（策略采样，热加载 LoRA，占一部分显存）+ HF 训练（2B 放得下；prompt 共享用 Llama 版 train.prefix_share_llama）
#   GPU1：RM 的 vLLM。默认全 2B（用户：整个都是 2B）：MiniCPM 底座挂 critic（rm1__mcgold）、对照 critic（rm1__mc）、满意度（sat_pair__mc）；
#         RMBASE / CRITIC / SHADOW / SATRM 可以换回 9B 版（RMBASE=$A/models/Qwen3.5-9B CRITIC=rm1__rm1_gold_9b SHADOW=rm1 SATRM=sat_pair，adapter 后缀 -9b）
# 奖励同 9B 第二轮：幻觉 + 满意度护栏 + 硬规则。prompt 和 9B 同一批原文，但长度参考、拒答筛选、参考回答都用 MiniCPM 自己的。
# 先试学习率（2B 和 9B 不一定一样），再正式跑 STEPS 步。结束写 MCGRPO_DONE；机器不关。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/mcgrpo.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MC=$FS/models/MiniCPM5-2B; O=${O:-outputs/rt/minicpm/grpo}; INIT=$FS/adapters/mc-init
RMBASE=${RMBASE:-$MC}; CRITIC=${CRITIC:-rm1__mcgold}; SHADOW=${SHADOW:-rm1__mc}; SATRM=${SATRM:-sat_pair__mc}
SUF=$([ "$RMBASE" = "$MC" ] && echo mc || echo 9b)
ad() { case $1 in rm1) echo $FS/adapters/rm1-9b/adapter ;; *) echo $FS/adapters/$1-$SUF/adapter ;; esac; }
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
fail() { log "$1 FAILED"; [ -n "${RM_URL:-}" ] || pkill -f '[v]llm serve'; log "MCGRPO_DONE"; exit 1; }
mkdir -p $O
[ -f $O/train_all.jsonl ] || cp outputs/rt/ansgrpo/train_all.jsonl outputs/rt/ansgrpo/dev.jsonl $O/
[ -f $INIT/adapter_model.safetensors ] || python -u -m train.ans_grpo --make_init --causal 1 --base $MC --out $INIT >> $LOG 2>&1 || fail init
if [ -n "${RM_URL:-}" ]; then
  # 双机模式（2026-11-04，用户：抢不到双卡就两台单卡，一台训练一台推理）：推理机（run_infer_node.sh）已起好同样的服务，
  # 经 AutoDL 对外域名访问（VLLM_API_KEY）。LoRA 每步存本机内存盘，后台线程直接上传给推理机的同步服务（LORA_SYNC_URL）再热加载，
  # 不经共享盘、不挡训练（用户：直接上传 + 后台推送）；LORA_SYNC_MODE=fs 时改走共享盘（PUSH_DIR 要设在 /root/autodl-fs 上）。
  [ -n "${VLLM_API_KEY:-}" ] && [ -n "${LORA_SYNC_URL:-}" ] || fail "remote mode needs VLLM_API_KEY and LORA_SYNC_URL"
  URLS=$RM_URL; PUSH=${PUSH_DIR:-/dev/shm/mc_lora}; ASYNC="--async_push 1"
  log "remote inference $RM_URL, push dir $PUSH"
else
pkill -f '[v]llm serve'
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done
# 两张卡对称（用户：底座都是 MiniCPM 2B，GPU1 也做推理）：各起一个同样的服务，都挂策略 + 3 个 RM LoRA；采样轮流分，打分按原文固定分到一个服务。
# 训练卡上的服务只占 MC_MEM 显存，其余留给训练。两次 serve_loras 用不同的 A，去视觉塔的副本各放一处，互不覆盖。
LORAS="critic=$(ad $CRITIC) sat=$(ad $SATRM) refuse=$(ad refuse) policy_v0=$INIT"
# DUAL=1：两张卡各一份服务（训练卡上的只占 MC_MEM 显存）。默认 DUAL=0（用户：训练才是瓶颈，没必要两个 vLLM）：
# 实测推理每步约 9 秒、训练约 25 秒；训练卡上的 vLLM 占的 11 GB 正好让快速训练设置 OOM，所以训练卡只训练，推理全在 GPU1。
if [ "${DUAL:-0}" = 1 ]; then
  mkdir -p $A/s0
  A=$A/s0 bash scripts/remote/serve_loras.sh --gpu 0 --port 8101 --mem ${MC_MEM:-0.35} --base $RMBASE --log $A/vllm-mc.log --priority $LORAS >> $LOG 2>&1 || fail "gpu0 vllm"
  URLS="http://127.0.0.1:8000/v1,http://127.0.0.1:8101/v1"
else
  URLS="http://127.0.0.1:8000/v1"
fi
A=$A bash scripts/remote/serve_loras.sh --gpu 1 --port 8000 --base $RMBASE --log $A/vllm-mcrm.log --priority $LORAS >> $LOG 2>&1 || fail "gpu1 vllm"
log "rm: base $RMBASE critic $CRITIC sat $SATRM refuse；两张卡各一份（对照 critic 训练中不打，训完用 9B 金标 critic 补打评测样本）"
log "both vllm up"
PUSH=/dev/shm/mc_lora
fi
URL0=${URLS%%,*} python -u - >> $LOG 2>&1 <<'EOF' || fail "zero-init check"
import os, requests
ids = list(range(1000, 1040))
H = {"Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "x")}
def lp(m):
    r = requests.post(os.environ["URL0"] + "/completions", headers=H, json={"model": m, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}, timeout=300).json()
    return [next(iter(x.values()))["logprob"] for x in r["choices"][0]["prompt_logprobs"][1:]]
d = max(abs(x - y) for x, y in zip(lp("base"), lp("policy_v0")))
print("zero_init_max_abs_diff", d)
assert d < 1e-2
EOF
U="--base_url ${URLS%%,*} --model policy_v0 --tokenizer $MC --dir $O --sys_qa 1"
[ -f $O/train.jsonl ] || python -u -m ragtruth.ans_data ref $U --keep_refusal 1 >> $LOG 2>&1 || fail ref   # 有拒答 RM：常拒答的题也留着（用户：要降拒答率）
log "ref done: $(tr -d '\n' < $O/train.meta.json)"
[ -f $O/train_ref.jsonl ] || python -u -m ragtruth.ans_data ref2 $U >> $LOG 2>&1 || fail ref2
# 每组采样数 N（用户：推理有余量，N 提高训练效率更高）：prompt 共享下多出的回答只多算回答部分；组内基线也更稳
COMMON="--causal 1 --sys_qa 1 --base $MC --init $INIT --vllm_url $URLS --policy_url $URLS --n ${N:-16} --share 1 --share_impl train.prefix_share_llama --prompts $O/train_ref.jsonl --dev $O/dev_ref.jsonl \
  --critic_report outputs/rt/rm/$CRITIC/eval/report.json --shadow= \
  --sat_model sat --sat_kind pair --sat_report outputs/rt/rm/$SATRM/eval/report.json --sat_w 0.5 --push_dir $PUSH   --refuse_model refuse --refuse_report outputs/rt/rm/refuse/eval/report.json --refuse_w 0.4 ${ASYNC:-} ${EXTRA_ARGS:-}"
# 训练提速（用户：训练是瓶颈）：2B 显存够，prompt 共享不逐层重算、8 条回答一批算完；第一次试跑 OOM 就退回逐层重算、每批 4 条
FAST="--share_ckpt 0 --share_mb 8"; SAFE="--share_ckpt 1 --share_mb 4"; TRAIN_OPTS=${TRAIN_OPTS:-$FAST}
if [ -z "${LR:-}" ]; then
  LR=5e-6
  for L in 5e-6 2e-5 5e-5; do
    P=$O/pilot/lr$L; rm -rf $P
    if ! CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo $COMMON $TRAIN_OPTS --steps 15 --lr $L --eval_k 0 --trip 0 --save_every 1000 --state_every 1000 --version_prefix p${L}_ --out $P >> $LOG 2>&1; then
      if [ "$TRAIN_OPTS" = "$FAST" ] && grep -aq "OutOfMemory\|CUDA out of memory" $LOG; then
        log "pilot $L OOM with fast settings, falling back to $SAFE"; TRAIN_OPTS=$SAFE; rm -rf $P
        CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo $COMMON $TRAIN_OPTS --steps 15 --lr $L --eval_k 0 --trip 0 --save_every 1000 --state_every 1000 --version_prefix p${L}b_ --out $P >> $LOG 2>&1 || { log "pilot $L FAILED"; break; }
      else
        log "pilot $L FAILED"; break
      fi
    fi
    ok=$(python -c "
import json
r=[json.loads(l) for l in open('$P/log.jsonl')]; r=[x for x in r if 't_step' in x][-5:]
m=lambda k:sum(x[k] for x in r)/len(r)
print(json.dumps({'lr':'$L','kl':m('kl'),'clip':m('clip_frac'),'vllm_hf':m('vllm_hf_absdiff'),'refusal':m('refusal'),'t_step':m('t_step'),'t_gen':m('t_gen'),'t_rm':m('t_rm'),'t_train':m('t_train')}))
import sys; sys.exit(0 if m('kl')<0.01 and m('clip_frac')<0.2 else 1)
") && { LR=$L; log "pilot pass $ok"; } || log "pilot stop $ok"
  done
fi
log "chosen lr $LR, train opts $TRAIN_OPTS"
CUDA_VISIBLE_DEVICES=0 python -u -m train.ans_grpo $COMMON $TRAIN_OPTS --steps ${STEPS:-200} --lr $LR --version_prefix m_ ${RESUME:+--resume $RESUME} --out $O/run >> $LOG 2>&1 && log "train done" || log "train FAILED"
AD=${ADOUT:-mc-grpo}; mkdir -p $FS/adapters/$AD && rsync -a $O/run/ckpt/ $FS/adapters/$AD/ && log "ckpts saved to adapters/$AD"
[ -n "${RM_URL:-}" ] || pkill -f '[v]llm serve'
log "MCGRPO_DONE"
