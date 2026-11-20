#!/bin/bash
# 按操作清单造错的生成器 GRPO，整套换成 Qwen3.5-9B（对应 2B 的 run_genops_grpo.sh）。双卡 5090：卡 0 训练，卡 1 vLLM（策略采样 + 把关模型 + critic，都是同一个 9B 基座上的 LoRA）。
#   策略起点 = 9B 基座 + 卡片提示词（B 清零的 LoRA，等于基座）；提示 = 训练池回答 × 20 种操作（和 2B 同一份 prompts_free.jsonl）
#   把关模型 = sev__v4_9b / rm2__v4_9b / opm__v1_9b / rm2__d2_9b（数据同 2B 版，run_crit9b.sh 训），critic = rm1__core_r2_bfcl_9b；没训完就等
#   ① 起点（9B 基座）在留出回答上采样，9B 把关模型打分；2B 各版生成器的留出样本也用 9B 把关模型重打一遍（文件后缀 _rm9，不覆盖 2B 打的 _rm4）
#   ② GRPO（默认阶段一，设置同 2B 的 s1c：lr 2e-5、100 步、每步 16 个任务 × 16 条）  ③ 训完的生成器在同一批留出回答上采样打分
#   环境变量：E=运行名（默认 s1c9）  STAGE  LR  STEPS  N  PPS  INIT  PROMPTS  EXTRA  SHARE_MB  RESCORE=要重打分的 2B 样本
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; O=outputs/rt/gen_ops_grpo; C=outputs/rt/core; R=outputs/rt/rm; T=scripts/analysis
E=${E:-s1c9}; STAGE=${STAGE:-1}; LR=${LR:-2e-5}; STEPS=${STEPS:-100}; N=${N:-16}; PPS=${PPS:-16}; LOG=$A/genops9_$E.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8; unset VLLM_API_KEY
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 3; done; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || { log "GENOPS9 FAILED: needs two GPUs"; exit 1; }
source $S/models9b.sh; ensure_9b || { log "GENOPS9 FAILED: 9b model"; exit 1; }; B9=$A/models/Qwen3.5-9B
GATES="sev__v4 rm2__v4 opm__v1 rm2__d2 rm1__core_r2_bfcl"
ad() { echo $FS/adapters/${1}_9b-9b/adapter; }
for i in $(seq 1 480); do   # 把关模型在别的机器上训，最多等 8 小时
  miss=""; for g in $GATES; do [ -f $(ad $g)/adapter_model.safetensors ] && [ -f $R/${g}_9b/eval/report.json ] || miss="$miss $g"; done
  [ -z "$miss" ] && break; [ $((i % 10)) = 1 ] && log "waiting for gates:$miss"; sleep 60
done
[ -z "$miss" ] || { log "GENOPS9 FAILED: gates missing:$miss"; exit 1; }
t() { python -c "import json;print(json.load(open('$R/${1}_9b/eval/report.json'))['temperature'])"; }
TJ="{\"sev4\": $(t sev__v4), \"nat4\": $(t rm2__v4), \"opm\": $(t opm__v1), \"rm1\": $(t rm1__core_r2_bfcl), \"disc\": $(t rm2__d2)}"
LORAS="sev4=$(ad sev__v4) nat4=$(ad rm2__v4) opm=$(ad opm__v1) rm1=$(ad rm1__core_r2_bfcl) disc=$(ad rm2__d2)"
log "gates ready, temperatures $TJ"
Z=$FS/adapters/gen-ops-zero-9b/adapter
[ -f $Z/adapter_model.safetensors ] || python scripts/zero_lora.py $FS/adapters/rm1__core_9b-9b/adapter $Z >> $LOG 2>&1 || { log "GENOPS9 FAILED: zero lora"; exit 1; }
INIT=${INIT:-$Z}; PF=${PROMPTS:-$O/prompts_free.jsonl}
[ -s $PF ] || python -u -m ragtruth.gen_ops prompts --pool genpool_train.jsonl --style free >> $LOG 2>&1 || { log "GENOPS9 FAILED: prompts"; exit 1; }
ev() {   # ev <标签> <生成器 adapter>：留出回答 × 20 种操作各采 4 条，9B 把关模型打分 → samples_ops_ev_<标签>.jsonl / _rm9.jsonl
  [ -s $C/samples_ops_ev_$1_rm9.jsonl ] && return 0
  free_gpu
  A=$A bash $S/serve_loras.sh --gpu 1 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-ev-$1.log $LORAS g=$2 >> $LOG 2>&1 || { log "ev serve $1 FAILED"; return 1; }
  [ -s $C/samples_ops_ev_$1.jsonl ] || python -u -m ragtruth.gen_ops sample --model g --tokenizer $B9 --card --style free --n 4 --out samples_ops_ev_$1.jsonl >> $LOG 2>&1 || { log "ev sample $1 FAILED"; return 1; }
  GENOPS_SUF=rm9 python -u $T/genops_score.py "$TJ" ev_$1 >> $LOG 2>&1 || { log "ev score $1 FAILED"; return 1; }
  log "EV $1 $(grep -a GENOPS_SAMPLE $LOG | tail -1 | cut -c1-300)"
}
ev base9 $Z
for x in ${RESCORE-ev_base_free ev_s1c ev_st4 ev_st8}; do   # 2B 生成器的留出样本换 9B 把关模型重打（vLLM 还开着）
  [ -s $C/samples_ops_$x.jsonl ] || { [ -s $C/samples_ops_$x.jsonl.gz ] && gunzip -k $C/samples_ops_$x.jsonl.gz; }
  [ -s $C/samples_ops_$x.jsonl ] || { log "rescore: missing $x"; continue; }
  [ -s $C/samples_ops_${x}_rm9.jsonl ] || GENOPS_SUF=rm9 python -u $T/genops_score.py "$TJ" $x >> $LOG 2>&1 || log "rescore $x FAILED"
done
log "BASELINES_DONE"
RD=$O/$E
if [ ! -d $RD/ckpt/step$STEPS ]; then
  free_gpu
  A=$A bash $S/serve_loras.sh --gpu 1 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-gog9.log --priority $LORAS policy_v0=$INIT >> $LOG 2>&1 || { log "GENOPS9 FAILED: vllm"; exit 1; }
  log "train $E: stage $STAGE lr $LR steps $STEPS n $N prompts/step $PPS init $INIT"
  CUDA_VISIBLE_DEVICES=0 python -u -m train.gen_grpo --base $B9 --init $INIT --prompts $PF --reward ops --ops_stage $STAGE \
    --temps sev4=sev__v4_9b,nat4=rm2__v4_9b,opm=opm__v1_9b,rm1=rm1__core_r2_bfcl_9b,disc=rm2__d2_9b \
    --share 1 --share_mb ${SHARE_MB:-4} --ref_on_vllm 1 --trip 0 --eval_k 0 --max_tokens 512 --pipeline 1 --rm_concurrency 128 \
    --steps $STEPS --lr $LR --n $N --prompts_per_step $PPS --save_every 50 --state_every 50 --version_prefix ${E}_ --out $RD ${EXTRA:-} >> $A/gengrpo_$E.log 2>&1 && log "$E done" || log "$E FAILED"
  tail -3 $A/gengrpo_$E.log | cut -c1-400 | sed "s/^/[$E] /" | tee -a $LOG
  cp $A/gengrpo_$E.log $RD/ 2>/dev/null
  for d in $RD/ckpt/step*; do [ -d $d ] && mkdir -p $FS/adapters/gen-ops-grpo-$E/$(basename $d) && rsync -a $d/ $FS/adapters/gen-ops-grpo-$E/$(basename $d)/; done
fi
G=$FS/adapters/gen-ops-grpo-$E/step$STEPS/policy
[ -f $G/adapter_model.safetensors ] || { log "GENOPS9 FAILED: no checkpoint $G"; free_gpu; exit 1; }
ev $E $G
free_gpu
mkdir -p $FS/results/genops9_$E && cp $LOG $A/gengrpo_$E.log $FS/results/genops9_$E/ 2>/dev/null
log "GENOPS9_DONE $E"
