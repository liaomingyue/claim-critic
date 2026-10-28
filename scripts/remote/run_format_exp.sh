#!/bin/bash
# 5090 上的格式对比实验流水线：等 9B v2 训完 → 评测 9B v2 → 依次训练并评测 0.6B 的 e2 / e3 / e5 / e5b。
# 每一步写 /root/autodl-tmp/exp.log；评测用 vLLM（Qwen3.5 需关掉多模态输入，并补齐分词器和处理器配置）。
set -u
cd /root/autodl-tmp/critic_model
LOG=/root/autodl-tmp/exp.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
export PYTHONHASHSEED=0 OPENAI_API_KEY=x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 克隆到新主机时系统盘环境可能不在：缺依赖就从华为云镜像重装（数据盘上的模型、代码、结果随克隆复制）
if ! python -c "import vllm, trl, peft, fla, rapidfuzz" 2>/dev/null; then
  log "reinstalling deps"
  pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple >/dev/null
  pip install -q -U vllm transformers trl==1.16.0 peft datasets modelscope accelerate rapidfuzz pyyaml flash-linear-attention >> /root/autodl-tmp/pip.log 2>&1
  log "deps installed"
fi

# 克隆出的系统盘会带回镜像里旧 torch 的残留 .py（如 _inductor/kernel/flex_attention.py），vLLM 导入时报 duplicate template name。
# 不在当前 torch RECORD 里的 .py 移到数据盘备份目录
if ! (cd /tmp && python -c "import vllm, torch._inductor.lowering") 2>/dev/null; then
  SP=$(python -c "import site; print(site.getsitepackages()[0])"); B=/root/autodl-tmp/torch_orphans
  (cd $SP && comm -23 <(find torch -name '*.py' | sort) <(cut -d, -f1 torch-*.dist-info/RECORD | grep '\.py$' | sort) |
    while read f; do mkdir -p $B/$(dirname $f); mv $f $B/$f; done)
  log "moved orphan torch files to $B"
fi

serve() {  # serve <ckpt> <name> <extra args...>
  pkill -f '[v]llm serve'; sleep 5
  nohup vllm serve "$1" --served-model-name "$2" --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 "${@:3}" \
    > /root/autodl-tmp/vllm-$2.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q "$2" && return 0; sleep 10; done
  log "vllm failed for $2"; return 1
}

if [ ! -f outputs/rt/eval/sft-9b-v2-notips/metrics.json ]; then
until grep -q SFT9B_V2_DONE /root/autodl-tmp/chain.log; do sleep 30; done
log "9B v2 trained"
B=/root/autodl-tmp/models/Qwen3.5-9B; C=/root/autodl-tmp/ckpt/sft-9b-v2-notips
cp $B/tokenizer_config.json $B/tokenizer.json $B/vocab.json $B/merges.txt $B/chat_template.jinja $B/preprocessor_config.json $B/video_preprocessor_config.json $C/ 2>/dev/null
serve $C sft-9b-v2-notips --limit-mm-per-prompt '{"image":0,"video":0}' && \
  python -m ragtruth.eval --model sft-9b-v2-notips --base_url http://127.0.0.1:8000/v1 --max_tokens 512 --concurrency 128 \
    --no_think --out outputs/rt/eval/sft-9b-v2-notips > /root/autodl-tmp/eval-sft-9b-v2-notips.log 2>&1
log "9B v2 evaluated"
pkill -f '[v]llm serve'; sleep 5
fi

[ -d /root/autodl-tmp/models/Qwen3-0.6B ] || modelscope download --model Qwen/Qwen3-0.6B --local_dir /root/autodl-tmp/models/Qwen3-0.6B > /dev/null 2>&1
for e in e2 e3 e5 e5b; do
  if [ -f outputs/rt/eval/sft-0.6b-$e/metrics.json ]; then log "skip $e (done)"; continue; fi
  log "train $e"
  python -u -m train.sft --config configs/rt/exp/sft_0.6b_$e.yaml --mode coldstart > /root/autodl-tmp/sft-0.6b-$e.log 2>&1 || { log "train $e FAILED"; continue; }
  case $e in
    e2) EV="--style verdict --max_tokens 64";;
    e3) EV="--style full --max_tokens 512";;
    e5) EV="--resp_style evidence --max_tokens 2048";;
    e5b) EV="--resp_style plain --max_tokens 256";;
  esac
  serve /root/autodl-tmp/ckpt/sft-0.6b-$e sft-0.6b-$e && \
    python -m ragtruth.eval --model sft-0.6b-$e --base_url http://127.0.0.1:8000/v1 --concurrency 128 --no_think $EV \
      --out outputs/rt/eval/sft-0.6b-$e > /root/autodl-tmp/eval-sft-0.6b-$e.log 2>&1
  pkill -f '[v]llm serve'; sleep 10   # 评测完立刻释放显存，否则下一组训练 OOM
  log "evaluated $e"
done
pkill -f '[v]llm serve'
log "ALL_DONE"
