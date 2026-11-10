#!/bin/bash
# 单字符选项 RM 训练 + 校准评测（2026-10-30，用户：三个 RM 都是 9B LoRA，三台 5090 并行）。
#   RM=rm1|rm2|rm3：数据 outputs/rt/rm/$RM/{train,calib,test}.jsonl（ragtruth.rm_data，本地建好后上传到文件存储）；
#   超参同 9B v2（configs/rt/rm_9b.yaml），从基座训；合并后用 vLLM 起服务，ragtruth.rm_eval 打分、拟合全局温度、出报告。
# 用法：RM=rm1 bash run_rmtrain.sh（工作目录 = 共享代码目录）。结束写 RMTRAIN_DONE。
set -u
cd ${CODE:-${FS_ROOT:-/root/autodl-fs}/code/critic_model}
RM=${RM:?}
LOG=${TMP_ROOT:-/root/autodl-tmp}/rmtrain.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${TMP_ROOT:-/root/autodl-tmp}/models; CK=${TMP_ROOT:-/root/autodl-tmp}/ckpt; B9=$M/Qwen3.5-9B
MM='{"image":0,"video":0}'
stop() { pkill -f '[v]llm serve'; sleep 10; pkill -9 -f '[V]LLM::EngineCore'; sleep 3; }
fix_tok() { cp $B9/{tokenizer_config.json,tokenizer.json,vocab.json,merges.txt,chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} "$1"/ 2>/dev/null; }
fail() { log "$1 FAILED"; exit 1; }
D=outputs/rt/rm/$RM
until [ -f $D/READY ]; do sleep 30; done   # 数据上传完会写 READY
ML=$(python -c "import json;print(json.load(open('$D/meta.json'))['max_tokens'])")
name=$RM-9b
sed -e "s|data: .*|data: $D/train.jsonl|" -e "s|output_dir: .*|output_dir: $CK/$name|" -e "s|max_length: .*|max_length: $ML|" \
  configs/rt/rm_9b.yaml > configs/rt/exp/$name.yaml
log "train $name (max_length $ML)"
python -u -m train.sft --config configs/rt/exp/$name.yaml --mode coldstart > ${TMP_ROOT:-/root/autodl-tmp}/$name.log 2>&1 || fail train
fix_tok $CK/$name; stop
nohup vllm serve $CK/$name --served-model-name $name --max-model-len 16384 --gpu-memory-utilization 0.9 --port 8000 \
  --limit-mm-per-prompt "$MM" --max-logprobs 20 > ${TMP_ROOT:-/root/autodl-tmp}/vllm-$name.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$name" && break; sleep 10; done
python -m ragtruth.rm_eval --rm $RM --model $name --base_url http://127.0.0.1:8000/v1 --concurrency 64 >> $LOG 2>&1 || fail eval
stop
log "RMTRAIN_DONE"
