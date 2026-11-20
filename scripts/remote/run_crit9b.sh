#!/bin/bash
# 同一份 critic 训练集换成 9B 底座训一遍，再给已标注的 RAG 回答逐句打分：用来分清一批补料没起作用是数据的问题还是 2B 的上限。
#   SRC=rm1__core_r2_bfcl（基线训练集）或 rm1__core_sp1l（基线 + 说话人错位补料），数据目录原样复制成 <SRC>_9b；
#   超参 configs/rt/rm_9b.yaml（run_rmtrain.sh），adapter 存 adapters/<SRC>_9b-9b；打分文件 <版本>_<集>_critic_<SRC>_9b.json。
# 一台机器跑一个 SRC（两台并行）。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; R=outputs/rt/rm
SRC=${SRC:?}; RM=${SRC}_9b; name=$RM-9b; LOG=$A/crit9b_$SRC.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/models9b.sh; B9=$A/models/Qwen3.5-9B
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
for f in train.jsonl calib.jsonl test.jsonl meta.json; do [ -s $R/$SRC/$f ] || { log "CRIT9B FAILED: missing $R/$SRC/$f"; exit 1; }; done
ensure_9b || { log "CRIT9B FAILED: 9b model"; exit 1; }
if [ ! -f $FS/adapters/$name/adapter/adapter_model.safetensors ] || [ ! -f $R/$RM/eval/report.json ]; then
  mkdir -p $R/$RM configs/rt/exp
  for f in train.jsonl calib.jsonl test.jsonl meta.json; do cp $R/$SRC/$f $R/$RM/; done
  touch $R/$RM/READY
  free_gpu; RM=$RM CODE=$PWD bash $S/run_rmtrain.sh >> $LOG 2>&1
  [ -f $R/$RM/eval/report.json ] || { log "CRIT9B FAILED: train $RM"; exit 1; }
  mkdir -p $FS/adapters/$name && rsync -a $A/ckpt/$name/adapter $FS/adapters/$name/ || { log "CRIT9B FAILED: save"; exit 1; }
fi
log "TRAINED $RM"
free_gpu
A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $B9 --log $A/vllm-crit9b.log $RM=$FS/adapters/$name/adapter >> $LOG 2>&1 || { log "CRIT9B FAILED: serve"; exit 1; }
T=$(python -c "import json;print(json.load(open('$R/$RM/eval/report.json'))['temperature'])")
for x in ${SETS:-cont8:v0 cont8:s100 cont9:v0 cont9:s100 cont9dyn:s100 cont8dyn:s100}; do
  arm=${x%%:*}; m=${x##*:}; K=outputs/rt/core/loop2_$arm/eval
  for s in a b c; do
    [ -f $K/${m}_$s.jsonl ] || { log "missing $K/${m}_$s.jsonl"; continue; }
    [ -f $K/${m}_${s}_critic_$RM.json ] || python -u -m ragtruth.critic_score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model $RM --temp $T --out $K/${m}_${s}_critic_$RM.json >> $LOG 2>&1 || { log "CRIT9B FAILED: score $x $s"; exit 1; }
  done
  log "scored $x"
done
pkill -f '[v]llm serve'
mkdir -p $FS/results/crit9b/$RM && cp $LOG $R/$RM/eval/report.json $FS/results/crit9b/$RM/ 2>/dev/null
log "CRIT9B_DONE $SRC"
