#!/bin/bash
# 复现的路径约定。被 setup.sh / r1_rescore.sh / r2_rag.sh source。
#   ROOT     代码根目录（git clone 下来的仓库），默认是本脚本往上两级
#   FS_ROOT  "共享存储"：基座模型和 LoRA 放这里（$FS_ROOT/models、$FS_ROOT/adapters），默认 $ROOT/.repro/store
#   TMP_ROOT 快速本地盘：vLLM 日志、基座的本地副本、训练中间文件，默认 $ROOT/.repro/work
# 原实验跑在 AutoDL 上，这两个目录分别是 /root/autodl-fs 和 /root/autodl-tmp；scripts/remote/ 下的脚本不设变量时仍用这两个路径。
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROOT FS_ROOT=${FS_ROOT:-$ROOT/.repro/store} TMP_ROOT=${TMP_ROOT:-$ROOT/.repro/work}
export CODE=$ROOT PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
mkdir -p "$FS_ROOT/models" "$FS_ROOT/adapters" "$TMP_ROOT"
rlog() { echo "$(date '+%F %T') $*" | tee -a "$TMP_ROOT/repro.log"; }
