#!/bin/bash
# 复现第 0 步：检查环境 → 下载发布的数据和 LoRA → 准备基座模型 → 把 LoRA 摆成训练脚本认识的目录结构。
#   bash scripts/repro/setup.sh [--local <发布目录>] [--data_repo U/claim-critic-data --model_repo U/claim-critic-models] [--base_src <已有基座目录>]
#   --base_src：机器上已经有 MiniCPM5-2B / Qwen3.5-9B（目录名同 HF 仓库名）时直接拷，不重新下载
# 需要：Linux、1 张 ≥ 32 GB 显存的 GPU（我们用 RTX 5090）、pip install -r requirements-gpu.txt、约 60 GB 磁盘。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd "$ROOT"
LOCAL=; DREPO=; MREPO=; BASE_SRC=
while [ $# -gt 0 ]; do case $1 in
  --local) LOCAL=$2; shift 2 ;; --data_repo) DREPO=$2; shift 2 ;; --model_repo) MREPO=$2; shift 2 ;; --base_src) BASE_SRC=$2; shift 2 ;;
  *) echo "setup: bad arg $1"; exit 1 ;; esac; done

rlog "== 1/4 环境检查"
command -v nvidia-smi > /dev/null || { echo "需要 NVIDIA GPU（nvidia-smi 不存在）"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 30000 ] || echo "警告：显存 ${mem} MiB < 32 GB，GRPO（R2）大概率放不下；R1 打分可以"
python - <<'PY' || exit 1
import importlib, sys
need = {"vllm": None, "torch": None, "transformers": None, "peft": None, "scipy": None, "huggingface_hub": None}
bad = []
for m in need:
    try:
        v = importlib.import_module(m).__version__; print(f"  {m:16s} {v}")
    except Exception as e:
        bad.append(m)
if bad:
    sys.exit(f"缺少：{bad}；先 pip install -r requirements-gpu.txt")
PY
command -v vllm > /dev/null || { echo "vllm 命令不在 PATH 里"; exit 1; }

rlog "== 2/4 发布的数据和 LoRA"
if [ -n "$LOCAL" ]; then python -m scripts.fetch_release --local "$LOCAL" --dest "$ROOT" | tail -3
elif [ -n "$DREPO" ]; then python -m scripts.fetch_release --data_repo "$DREPO" --model_repo "$MREPO" --dest "$ROOT" | tail -3
else echo "  跳过（没给 --local / --data_repo；假定已经解压过）"; fi
[ -f outputs/rt/minicpm/grpo300/answers.jsonl ] && [ -d adapters/rm1__core_r2_bfcl ] || { echo "数据或 LoRA 不全：先用 --local 或 --data_repo/--model_repo"; exit 1; }

rlog "== 3/4 基座模型 → $FS_ROOT/models"
get_base() {   # $1 = HF 仓库 $2 = 目录名 $3 = revision（可空）
  local d=$FS_ROOT/models/$2
  [ -f $d/config.json ] && { echo "  $2 已有"; return 0; }
  if [ -n "$BASE_SRC" ] && [ -f $BASE_SRC/$2/config.json ]; then rsync -a $BASE_SRC/$2/ $d/ && echo "  $2 从 $BASE_SRC 拷贝"; return; fi
  python - "$1" "$d" "$3" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, d, rev = sys.argv[1], sys.argv[2], sys.argv[3] or None
snapshot_download(repo, local_dir=d, revision=rev)
print("  downloaded", repo, rev or "main")
PY
}
get_base openbmb/MiniCPM5-2B MiniCPM5-2B f974000 || exit 1
get_base Qwen/Qwen3.5-9B Qwen3.5-9B "" || exit 1

rlog "== 4/4 LoRA 目录结构 → $FS_ROOT/adapters"
# 发布包里是 adapters/<名字>/；训练脚本按原实验的布局找：打分模型 <名字>-mc/adapter、9B critic <名字>-9b/adapter、
# RAG SFT rag-sft-core/adapter、GRPO 检查点 mc-grpo-*/stepN/policy。这里只建软链接，不拷权重。
for d in "$ROOT"/adapters/*/; do
  n=$(basename "$d")
  if [ -f "$d/adapter_config.json" ]; then
    case $n in
      rag-sft-*) t=$n/adapter ;;
      *_9b) t=$n-9b/adapter ;;
      *) t=$n-mc/adapter ;;
    esac
  else t=$n; fi
  mkdir -p "$(dirname "$FS_ROOT/adapters/$t")"; ln -sfn "${d%/}" "$FS_ROOT/adapters/$t"
done
ls "$FS_ROOT/adapters" | wc -l | xargs echo "  adapters linked:"
rlog "SETUP_DONE"
