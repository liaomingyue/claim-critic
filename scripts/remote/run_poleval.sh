#!/bin/bash
# 策略独立评测（2026-11-05，通用）：NAME=<名> ADAPTER=<LoRA 目录> OUT=<输出目录>，在三套验收原文上生成
# （grpo300 T1.0、grpo300b T0.7、grpo300c T0.7，top_p 0.95，同 run_r3full），9B 金标 critic 用 critic_score_multi 打分。已有的跳过。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; MC=$A/models/MiniCPM5-2B; E=${OUT:?}; mkdir -p $E; LOG=$A/poleval.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!   # 保活进程 PID：下面的 wait 不能等它（2026-11-05 worker1 生成完卡死在 wait）
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false
unset VLLM_API_KEY
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
[ -f $MC/config.json ] || { mkdir -p $A/models && rsync -a $FS/models/MiniCPM5-2B $A/models/; }
source scripts/remote/models9b.sh; start_9b_bg   # 2026-11-05 用户：先拷马上要用的 2B，9B 后台拷，把时间岔开
m=${NAME:?}
free_gpu; mkdir -p $A/s0
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $MC --log $A/vllm-poleval.log $m=${ADAPTER:?} >> $LOG 2>&1 || { log "serve $m FAILED"; exit 1; }
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_a.jsonl >> $LOG 2>&1 &
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300b/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_b.jsonl >> $LOG 2>&1 &
python -u -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300c/answers.jsonl --temperature 0.7 --top_p 0.95 --url http://127.0.0.1:8000/v1 --model $m --out $E/${m}_c.jsonl >> $LOG 2>&1 &
for p in $(jobs -p); do [ "$p" = "$KA" ] || wait $p; done; free_gpu
C9=rm1__rm1_gold_9b; T=$(temp $C9)
ensure_9b || { log "9b copy FAILED"; exit 1; }   # 后台拷（start_9b_bg）还没完就等它
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --base $A/models/Qwen3.5-9B --log $A/vllm-poleval9b.log critic=$FS/adapters/$C9-9b/adapter >> $LOG 2>&1 || { log "serve 9b FAILED"; exit 1; }
jobs="["; for s in a b c; do jobs+="{\"model\":\"critic\",\"temp\":$T,\"answers\":\"$E/${m}_$s.jsonl\",\"out\":\"$E/${m}_${s}_9b.json\"},"; done
echo "${jobs%,}]" > $E/jobs9b_$m.json
python -u -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs $E/jobs9b_$m.json --concurrency 256 >> $LOG 2>&1 || log "9b score $m FAILED"
free_gpu
log "POLEVAL_DONE $m"
