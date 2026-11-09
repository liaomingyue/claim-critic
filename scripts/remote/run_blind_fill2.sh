#!/bin/bash
# 2026-11-09 用户：多开卡并行。补 critic 盲区（blind_fill）的第二台机器：
#   ① g1（GRPO 50 步的生成器，共享盘 outputs/rt/gen_v2_grpo/g1/ckpt/step50）在大提示池上温度 0 采样 → samples_g1.jsonl + G1_DONE
#   ② 等主机器 build 出 rm1__core_r2_bfnc 的训练集 → 训 nc 组 critic → 三套验收打分
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; LOG=$A/blind_fill2.log; O=outputs/rt/gen_v2_grpo; B=$O/blind_fill
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
t() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
RMS="sev=$FS/adapters/sev__v3c0-mc/adapter rm2v3=$FS/adapters/rm2__v3b-mc/adapter rm1=$FS/adapters/rm1__core_r2-mc/adapter"
G1=$O/g1/ckpt/step50; [ -d $G1/policy ] && G1=$G1/policy
mkdir -p $B $A/lora && rsync -a $G1/ $A/lora/g1/ && G1=$A/lora/g1   # LoRA 先拷本地盘
if [ ! -f $B/G1_DONE ]; then
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bf2.log $RMS g1=$G1 >> $LOG 2>&1 || { log "vllm FAILED"; exit 1; }
  python -u -m ragtruth.blind_fill sample --model g1 --tag g1 --tokenizer $MC >> $LOG 2>&1 || { log "sample g1 FAILED"; exit 1; }
  touch $B/G1_DONE
  log "$(grep -a '^BLIND_FILL_SAMPLE' $LOG | tail -1)"
  free_gpu
fi
name=rm1__core_r2_bfnc
log "waiting for build ($name)"
for i in $(seq 1 600); do [ -f outputs/rt/rm/$name/meta.json ] && break; sleep 60; done
[ -f outputs/rt/rm/$name/meta.json ] || { log "no build after 10h, stop"; exit 1; }
if [ ! -f outputs/rt/rm/$name/eval/report.json ]; then
  free_gpu; RM=$name GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/$name/eval/report.json ] || { log "nc train FAILED"; exit 1; }
fi
T=$(t $name); log "TRAINED nc T=$T"
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-bfnc.log c=$FS/adapters/$name-mc/adapter >> $LOG 2>&1 || { log "nc serve FAILED"; exit 1; }
PIDS=""
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
  [ -f $d/critic_$name.json ] && continue
  python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model c --temp $T --out $d/critic_$name.json >> $LOG 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
free_gpu
log "BLIND_FILL2_DONE"
