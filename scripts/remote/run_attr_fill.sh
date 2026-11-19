#!/bin/bash
# 专项补 critic 的张冠李戴盲区（ragtruth/attr_fill.py）：按规则构造的错句 + 原句，本地拼好传到 outputs/rt/attr_fill/<轮>/add.jsonl。
#   ARM=l：只给标签（单字母，和基线同格式）→ run_mcrm.sh 训练 → run_critic_on_rag.sh 给已标注的回答打分
#   ARM=t：先写核对过程再给结论 → 同样的超参训练 → attr_fill score 打分（生成核对过程，取 Answer: 后那个字母的概率）
# 两组加的句子、顺序都一样，只差输出格式。一台机器跑一组（两台并行），打分文件名 <版本>_<集>_critic_<名>.json。
cd ${CODE:-/root/autodl-fs/code/critic_model}
A=/root/autodl-tmp; FS=/root/autodl-fs; S=scripts/remote; R=outputs/rt/rm
ARM=${ARM:?}; RD=${RD:-at1}; CR=${CR:-rm1__core_r2_bfcl}; NAME=${NAME:-rm1__core_$RD}; B=outputs/rt/attr_fill/$RD; RM=$NAME$ARM; LOG=$A/attr_fill_$ARM.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset VLLM_API_KEY
( while kill -0 $$ 2>/dev/null; do touch $A/KEEPALIVE; sleep 60; done ) &
source $S/model_env.sh minicpm5-2b && MC=$MODEL_LOCAL
free_gpu() { pkill -f '[v]llm serve'; for i in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done; }
for i in $(seq 1 90); do [ -s $B/add.jsonl ] && break; [ $i = 1 ] && log "waiting for $B/add.jsonl"; sleep 20; done   # 本地构造的句子在开机后才传得上来（先传成 .part 再改名）
for f in $R/$CR/train.jsonl $R/$CR/calib.jsonl $R/$CR/meta.json $B/add.jsonl configs/rt/accept_split.json; do [ -s $f ] || { log "ATTRFILL FAILED: missing $f"; exit 1; }; done
python -u -m ragtruth.attr_fill build --round $RD --base $CR --name $NAME --arms $ARM >> $LOG 2>&1 || { log "ATTRFILL FAILED: build"; exit 1; }
log "$(grep -a '^ATTR_FILL_BUILD' $LOG | tail -1)"
SETS="${SETS:-cont8:v0 cont8:s100 cont9:v0 cont9:s100 cont9dyn:s100 cont8dyn:s100}"
if [ "$ARM" = l ]; then
  if [ ! -f $R/$RM/eval/report.json ]; then
    free_gpu; RM=$RM GPU=0 PORT=8100 MC=$MC bash $S/run_mcrm.sh >> $LOG 2>&1; free_gpu
    [ -f $R/$RM/eval/report.json ] || { log "ATTRFILL FAILED: train $RM"; exit 1; }
  fi
  log "TRAINED $RM"
  # 基线 critic 在新标注的两组回答上还没有分数，一起打（已有的文件会跳过）
  CRITICS="$CR $RM" ARMS="cont8 cont9 cont9dyn cont8dyn" CODE=$PWD bash $S/run_critic_on_rag.sh >> $LOG 2>&1 || { log "ATTRFILL FAILED: score"; exit 1; }
else
  CK=$A/ckpt; name=$RM-mc; D=$R/$RM
  if [ ! -f $FS/adapters/$name/adapter/adapter_model.safetensors ]; then
    ML=$(( $(python -c "import json;print(json.load(open('$D/meta.json'))['max_tokens'])") + 256 ))   # 核对过程多出的 token
    mkdir -p configs/rt/exp
    sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $D/train.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" -e "s|max_length: .*|max_length: $ML|" configs/rt/rm_mc.yaml > configs/rt/exp/$name.yaml
    grep -q "base_model: $MC" configs/rt/exp/$name.yaml || { log "ATTRFILL FAILED: config"; exit 1; }
    free_gpu; log "train $name"
    CUDA_VISIBLE_DEVICES=0 python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1 || { log "ATTRFILL FAILED: train $RM"; exit 1; }
    mkdir -p $FS/adapters/$name && rsync -a $CK/$name/adapter $FS/adapters/$name/ || { log "ATTRFILL FAILED: save"; exit 1; }
  fi
  log "TRAINED $RM"
  free_gpu
  A=$A/s0 bash $S/serve_loras.sh --gpu 0 --port 8000 --mem 0.85 --base $MC --log $A/vllm-attr.log $RM=$FS/adapters/$name/adapter >> $LOG 2>&1 || { log "ATTRFILL FAILED: serve"; exit 1; }
  for x in $SETS; do
    arm=${x%%:*}; m=${x##*:}; K=outputs/rt/core/loop2_$arm/eval
    for s in a b c; do
      [ -f $K/${m}_$s.jsonl ] || { log "missing $K/${m}_$s.jsonl"; continue; }
      [ -f $K/${m}_${s}_critic_$RM.json ] || python -u -m ragtruth.attr_fill score --answers $K/${m}_$s.jsonl --url http://127.0.0.1:8000/v1 --model $RM --out $K/${m}_${s}_critic_$RM.json >> $LOG 2>&1 || { log "ATTRFILL FAILED: score $x $s"; exit 1; }
    done
    log "scored $x"
  done
  pkill -f '[v]llm serve'
fi
mkdir -p $FS/results/attr_fill_$RD/$RM && cp $LOG $R/$RM/meta.json $FS/results/attr_fill_$RD/$RM/ 2>/dev/null
log "ATTRFILL_DONE $ARM"
