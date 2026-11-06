#!/bin/bash
# 只做 run_r3full.sh 的独立评测部分（2026-11-05：第一组 h3ctl 用的旧版脚本只评了 a、b，补 v0 和第三套 c）。TAG 同 run_r3full。
# 正式回答 GRPO（2026-11-05，用户：把 R3 作为新的奖励设定，正式跑完整的回答 GRPO）。单卡：策略 vLLM（0.35 显存）+ RM LoRA + 训练都在 GPU0。
# 进奖励：rm1__mcgold_h3ctl（幻觉：去底噪计数 K = max(2, 0.5 × 参考句数) + 句子层 β 0.6 + 前一句 γ 0.3）、sat_pair__mc（满意度只罚 0.8 ×，最多 0.4）、
#         refuse（拒答 0.4 × P(可拒答)）、硬规则。只监控：9B 金标 critic（训完对检查点独立评测，不进奖励）。
# mc-init 起，lr 5e-6，n 16 × 16 prompt，STEPS（默认 300）步，每 50 步存检查点 + 续训状态，TripWire 打开。
# 训完：第 100 / 200 / 300 步在 200 个验收原文-种子上生成（同 run_after：grpo300 T1.0、grpo300b T0.7，top_p 0.95），9B 金标 critic 打分 → $O/eval/
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; LOG=$A/r3full${TAG:-}.log; S=scripts/remote
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
# 看门狗保活（2026-11-05：拷 / 加载 9B 时 GPU 空闲超过 10 分钟被关机）：脚本活着就每分钟 touch，脚本结束循环自己退出
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!   # 保活进程 PID：下面的 wait 不能等它（2026-11-05 worker1 生成完卡死在 wait）
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
[ -f $A/models/MiniCPM5-2B/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg   # 2026-11-05 用户：先拷马上要用的 2B，9B 后台拷，把时间岔开
MC=$A/models/MiniCPM5-2B; SRC=outputs/rt/minicpm/grpo; INIT=$FS/adapters/mc-init; CRITIC=${CRITIC:-rm1__mcgold_h3ctl}; STEPS=${STEPS:-300}
TAG=${TAG:-}; O=outputs/rt/minicpm/r3full$TAG; E=$O/eval; mkdir -p $E   # TAG：换 critic 的对照组（2026-11-05，用户：两个显著不同的 RM 各用 R3 训 RAG）
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
[ "${NOWAIT:-0}" = 1 ] || until grep -q "R3FULL_DONE" $LOG 2>/dev/null; do sleep 60; done   # 等训练脚本（旧版只评 a b）跑完，再补评：已有的生成 / 打分会跳过
free_gpu; mkdir -p $A/s0
# 独立评测
POL="v0=$INIT"; for s in ${EVAL_STEPS:-100 200 300}; do [ -d $O/run/ckpt/step$s ] && POL+=" s$s=$(ad $O/run/ckpt/step$s)"; done
[ -n "$POL" ] || { log "no ckpts to eval"; exit 1; }
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-r3gen.log $POL >> $LOG 2>&1 || { log "serve policies FAILED"; exit 1; }
for kv in $POL; do m=${kv%%=*}
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_a.jsonl >> $LOG 2>&1 &
  python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_b.jsonl >> $LOG 2>&1 &
  # 第三套（2026-11-05：扩大评测量，两个 RM 的差异可能比 R3 对 R0 小）：132 篇新原文 × 3 种子，温度 0.7
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
log "R3EVAL_DONE"
