#!/bin/bash
# MiniCPM5-2B 版的单字母 RM（2026-11-02，用户：整个都用 2B）：RM=rm1__mcgold GPU=0 PORT=8100 bash run_mcrm.sh
# 数据和 9B 版完全相同（目录由 run_mc_all.sh 复制），超参 configs/rt/rm_mc.yaml（同 9B 版，只换底座、批大小改成 4 × 4）。合并后校准、出报告，adapter 存共享盘 adapters/$RM-mc。
set -u
cd ${CODE:-/root/autodl-fs/code/critic_model}
RM=${RM:?}; GPU=${GPU:?}; PORT=${PORT:?}
A=/root/autodl-tmp; LOG=$A/mcrm_$RM.log; MC=/root/autodl-fs/models/MiniCPM5-2B
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
CK=$A/ckpt; name=$RM-mc; D=outputs/rt/rm/$RM
fix_tok() { for f in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja generation_config.json; do cp $MC/$f "$1"/ 2>/dev/null; done; }
fail() { log "$RM $1 FAILED"; touch $D/FAILED; log "MCRM_DONE $RM"; exit 1; }
# 认领：同一个 RM 只训一次。别处已认领就等它出报告（run_mc_all.sh 的 wait 要等到所有 RM 都训完）
mkdir -p /root/autodl-fs/runs/claims
if ! mkdir /root/autodl-fs/runs/claims/mcrm_$RM 2>/dev/null; then
  log "$RM claimed elsewhere, waiting for its report"
  until [ -f $D/eval/report.json ] || [ -f $D/FAILED ]; do sleep 30; done
  log "MCRM_DONE $RM (by other)"; exit 0
fi
ML=$(python -c "import json;print(json.load(open('$D/meta.json'))['max_tokens'])")
mkdir -p configs/rt/exp
sed -e "s|base_model: .*|base_model: $MC|" -e "s|data: .*|data: $D/train.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" -e "s|max_length: .*|max_length: $ML|" configs/rt/rm_mc.yaml > configs/rt/exp/$name.yaml
grep -q "base_model: $MC" configs/rt/exp/$name.yaml || fail config
log "train $name"
python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > $A/$name.log 2>&1 || fail train
log "trained $name"
mkdir -p /root/autodl-fs/adapters/$name && rsync -a $CK/$name/adapter /root/autodl-fs/adapters/$name/ || fail save
fix_tok $CK/$name
vllm serve $CK/$name --served-model-name $name --max-model-len 16384 --gpu-memory-utilization 0.45 --port $PORT --max-logprobs 20 > $A/vllm-$name.log 2>&1 &
V=$!
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models | grep -q "$name" && break; kill -0 $V 2>/dev/null || fail serve; sleep 10; done
python -m ragtruth.rm_eval --rm $RM --model $name --base_url http://127.0.0.1:$PORT/v1 --concurrency 32 >> $LOG 2>&1 || { kill $V; fail eval; }
kill $V; sleep 5
log "MCRM_DONE $RM"
