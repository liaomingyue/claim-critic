#!/bin/bash
# 启动多 LoRA 的 vLLM 服务并等它就绪（2026-11-02，用户：做成固定脚本复用）。以后脚本都用它，不要自己写 vllm serve 和等待循环。
#   自动去掉 LoRA 里的视觉塔参数（strip_visual_lora.py，结果放 $A/lora/<名字>；vLLM 不接受，踩过：缺这步服务起不来）；
#   批量 token 上限默认 16384（vLLM 对 5090 的默认值只有 2048，一批只装得下一两个 prompt，显存空着请求却在排队，2026-11-02）；
#   前缀匹配细化到 16 token（重发同样的请求命中 82% → 99%；改不了线性注意力层状态的存储密度：块固定 528，--mamba-block-size 无效，换回答第一句只命中 34%）；
#   服务一退出就报错并打印日志尾，不等到超时（踩过：崩了还空等 20 分钟）。
#
#   bash scripts/remote/serve_loras.sh --gpu 1 --log $A/vllm.log [--port 8000] [--base <基座>] [--priority] [--mem 0.9] [--rank 32] critic=<目录> policy_v0=<目录> ...
#
# 就绪返回 0，服务留在后台；失败返回 1。工作目录要是代码根目录。
set -u
A=${A:-/root/autodl-tmp}
GPU=1; PORT=8000; BASE=$A/models/Qwen3.5-9B; LOG=$A/vllm.log; MEM=0.9; RANK=32; BATCH=16384; RETAIN=(); SCHED=(); LORAS=()
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --base) BASE=$2; shift 2 ;;
    --log) LOG=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --rank) RANK=$2; shift 2 ;;
    --batch) BATCH=$2; shift 2 ;;
    --vllm-args) read -r -a RETAIN <<< "$2"; shift 2 ;;   # 原样传给 vllm serve 的额外参数
    --priority) SCHED=(--scheduling-policy priority); shift ;;
    *=*) LORAS+=("$1"); shift ;;
    *) echo "serve_loras: bad arg $1"; exit 1 ;;
  esac
done
[ ${#LORAS[@]} -gt 0 ] || { echo "serve_loras: no loras"; exit 1; }
# 等上一个服务真正放开显存（踩过：pkill 后 sleep 8 不够，启动报显存不足）
for i in $(seq 1 30); do [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; sleep 3; done
# 双机模式（2026-11-04）：推理机的服务经 AutoDL 对外域名访问，VLLM_API_KEY 设了就要求密钥（单机默认不设，行为不变）
KEYARG=(); AUTH=()
if [ -n "${VLLM_API_KEY:-}" ] && [ "${VLLM_API_KEY}" != x ]; then KEYARG=(--api-key "$VLLM_API_KEY"); AUTH=(-H "Authorization: Bearer $VLLM_API_KEY"); fi
MODS=(); LAST=
for kv in "${LORAS[@]}"; do
  name=${kv%%=*}; src=${kv#*=}; dst=$A/lora/$name
  [ -f "$src/adapter_model.safetensors" ] || { echo "serve_loras: $name: no adapter at $src"; exit 1; }
  mkdir -p $A/lora; rm -rf "$dst"
  python scripts/strip_visual_lora.py "$src" "$dst" || { echo "serve_loras: strip $name FAILED"; exit 1; }
  MODS+=("$name=$dst"); LAST=$name
done
CUDA_VISIBLE_DEVICES=$GPU VLLM_ALLOW_RUNTIME_LORA_UPDATING=True nohup vllm serve "$BASE" --served-model-name base --max-model-len 16384 \
  --gpu-memory-utilization "$MEM" --port "$PORT" --limit-mm-per-prompt '{"image":0,"video":0}' --max-logprobs 20 --enable-lora --max-lora-rank "$RANK" \
  --max-num-batched-tokens "$BATCH" --prefix-match-unit 16 --max-loras ${MAX_LORAS:-8} --max-cpu-loras ${MAX_CPU_LORAS:-24} "${RETAIN[@]}" "${SCHED[@]}" "${KEYARG[@]}" --lora-modules "${MODS[@]}" > "$LOG" 2>&1 &
VP=$!
for i in $(seq 1 120); do
  curl -s -m 3 "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$LAST\"" && { echo "serve_loras: up on port $PORT after $((i * 10))s: ${LORAS[*]%%=*}"; exit 0; }
  kill -0 $VP 2>/dev/null || { echo "serve_loras: vllm exited"; grep -a -E "Error|error" "$LOG" | tail -3 | cut -c1-400; exit 1; }
  sleep 10
done
echo "serve_loras: not ready after 20 min"; tail -3 "$LOG" | cut -c1-400; exit 1
