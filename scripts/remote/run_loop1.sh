#!/bin/bash
# 循环一（错误合成，2026-11-06，用户：训练完成后开始循环；一台单卡顺序跑）。全部只用核心集派生的数据和 2B 模型，9B 不进奖励、不筛样本。
#   ① sev__core：等本地 Claude 单独标注（outputs/rt/core/sevpool_labels.jsonl，随代码上传）→ core sev → run_mcrm.sh
#   ② 错误改写模型 GRPO（单卡：vLLM 0.35 显存挂 rm1 / rm2 / sev / policy_v0 + 训练同卡），起点 gen-core-mc，D4 训练池，n 16 × 16 prompt，lr 5e-6：
#        前 50 步 shape sev（门槛 P(显然) ≥ 0.5，保底 0.1）；续 50 步区间模式（保底 0.02，critic 0.3–0.7 加 0.5，难度项 = 离 0.5 的接近度）
#   ③ 采集：第 100 步在 D4 训练池每个 prompt 采 8 条（sev 门槛）→ outputs/rt/core/harvest_r1.jsonl
#   ④ 入库：critic_harvest --base rm1__core，P(显然) ≥ 0.5、sev P(F) ≤ 0.5、RM2 P(自然) ≥ 0.3，critic 分数不限，最多 550 对 → rm1__core_r1，run_mcrm.sh
#   ⑤ 三套验收回答（Claude 金标）上给 rm1__core / rm1__core_r1 打分（本地 critic_cap3 算显然错 AUC / 召回）
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; C=outputs/rt/core; LOG=$A/loop1.log; mkdir -p $C/done
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch /root/autodl-tmp/KEEPALIVE; sleep 60; done ) & KA=$!
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset VLLM_API_KEY LORA_SYNC_URL LORA_SYNC_MODE LORA_SYNC_FS_DIR
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
temp() { python -c "import json;print(json.load(open('outputs/rt/rm/$1/eval/report.json'))['temperature'])"; }
ad() { [ -f $1/policy/adapter_model.safetensors ] && echo $1/policy || echo $1; }
step() { [ -f $C/done/$1 ]; }
mark() { touch $C/done/$1; log "step $1 done"; }
source $S/model_env.sh minicpm5-2b || { log "model_env FAILED"; exit 1; }
MC=$MODEL_LOCAL; G=outputs/rt/core/grpo_gen; INIT=$FS/adapters/gen-core-mc/adapter
TEMPS=rm1=rm1__core,rm2=rm2__core,sev=sev__core

# ① sev__core
log "waiting for $C/sevpool_labels.jsonl"
until [ -s $C/sevpool_labels.jsonl ]; do sleep 60; done
if [ ! -f outputs/rt/rm/sev__core/eval/report.json ]; then
  python -u -m ragtruth.core sev >> $LOG 2>&1 || { log "core sev FAILED"; exit 1; }
  RM=sev__core GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/sev__core/eval/report.json ] || { log "sev__core FAILED"; exit 1; }
fi
for r in rm1__core rm2__core sev__core; do log "$r T=$(temp $r)"; done

# ② GRPO
serve_rms() {   # serve_rms <额外 LoRA ...>
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem ${1:-0.35} --base $MC --log $A/vllm-loop1.log --priority \
    rm1=$FS/adapters/rm1__core-mc/adapter rm2=$FS/adapters/rm2__core-mc/adapter sev=$FS/adapters/sev__core-mc/adapter policy_v0=$INIT "${@:2}" >> $LOG 2>&1
}
grpo() {   # grpo <总步数> <额外参数...>
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --causal 1 --base $MC --init $INIT --prompts $C/genpool_train.jsonl --eval_prompts $C/genpool_eval.jsonl \
    --reward target --div_mode place --err_mode ab --w_div 0.1 --conflict_range 0.25 0.8 --other_max 1.0 --top_max 0.75 --raw_ratio 4 \
    --share 1 --share_impl $MODEL_SHARE_IMPL --share_mb 8 --ref_on_vllm 1 --steps $1 --lr 5e-6 --n 16 --prompts_per_step 16 --eval_k 4 --save_every 50 \
    --vllm_url http://127.0.0.1:8000/v1 --temps $TEMPS --shape sev --gate 0.5 --proc_producer 1 --pipeline 1 --version_prefix core_ --out $G "${@:2}" >> $A/loop1_grpo.log 2>&1
}
if ! step grpo50; then
  serve_rms 0.35 || { log "serve rms FAILED"; exit 1; }
  log "grpo steps 0-50 (shape sev, base_fail 0.1)"
  grpo 50 --base_fail 0.1 && mark grpo50 || { log "grpo50 FAILED (see loop1_grpo.log)"; exit 1; }
fi
if ! step grpo100; then
  serve_rms 0.35 || { log "serve rms FAILED"; exit 1; }
  log "grpo steps 50-100 (band 0.3-0.7, bonus 0.5, base_fail 0.02)"
  grpo 100 --base_fail 0.02 --blind_bonus 0.5 --blind_lo 0.3 --blind_thr 0.7 --resume $G/state/step50 --reset_reward 1 && mark grpo100 || { log "grpo100 FAILED"; exit 1; }
fi
grep -ah QUICK_EVAL $A/loop1_grpo.log | cut -c1-600 | tee -a $LOG
mkdir -p $FS/adapters/gen-core-grpo && rsync -a $G/ckpt/ $FS/adapters/gen-core-grpo/

# ③ 采集
G100=$(ad $G/ckpt/step100)
if [ ! -s $C/harvest_r1.jsonl ]; then
  serve_rms 0.85 gen=$G100 || { log "serve gen FAILED"; exit 1; }
  python -u -m ragtruth.gen_target_data harvest --base $MC --url http://127.0.0.1:8000/v1 --model gen --inp ../core/genpool_train.jsonl --samples 8 \
    --rm1 rm1 --rm1_temp $(temp rm1__core) --rm2 rm2 --rm2_temp $(temp rm2__core) --sev sev --sev_temp $(temp sev__core) --gate 0.5 --out ../core/harvest_r1.jsonl >> $LOG 2>&1 \
    || { log "harvest FAILED"; exit 1; }
  grep -a HARVEST $LOG | tail -1 | cut -c1-600
fi

# ④ 入库 + 重训 critic
if [ ! -f outputs/rt/rm/rm1__core_r1/eval/report.json ]; then
  python -u -m ragtruth.critic_harvest --harvest $C/harvest_r1.jsonl --base rm1__core --name rm1__core_r1 --lo 0.0 --hi 1.01 --gate 0.5 --min_nat 0.3 --max_add 550 >> $LOG 2>&1 \
    || { log "critic_harvest FAILED"; exit 1; }
  free_gpu
  RM=rm1__core_r1 GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
  [ -f outputs/rt/rm/rm1__core_r1/eval/report.json ] || { log "rm1__core_r1 FAILED"; exit 1; }
fi

# ⑤ 验收回答上打分（critic_cap3 的输入：<集>/critic_<名>.json）
serve_rms 0.85 c0=$FS/adapters/rm1__core-mc/adapter c1=$FS/adapters/rm1__core_r1-mc/adapter || { log "serve critics FAILED"; exit 1; }
for d in outputs/rt/minicpm/grpo300 outputs/rt/minicpm/grpo300b outputs/rt/minicpm/grpo300c; do
  for kv in c0=rm1__core c1=rm1__core_r1; do m=${kv%%=*} r=${kv#*=}; f=$d/critic_$r.json
    [ -f $f ] || python -u -m ragtruth.critic_score --answers $d/answers.jsonl --url http://127.0.0.1:8000/v1 --model $m --temp $(temp $r) --out $f >> $LOG 2>&1 || log "score $r $d FAILED"
  done
done
free_gpu
log "LOOP1_DONE"
