#!/bin/bash
# 客服 Agent（cs_agent）用的 τ²-bench：在共享盘上建一次，各机器共用。
#   τ²-bench 精简包（src + pyproject + data/tau2 里的 retail 和英文语言包，约 34 MB）由本地上传到 $FS/env/tau2_slim.tar（scripts/cs_pack_tau2.sh 打包）；
#   装进一个带系统包的 venv（$FS/env/venv-tau2，--system-site-packages）：vLLM、torch、transformers 用镜像里的，
#   τ² 的依赖（litellm 等）只装进 venv，不会改到镜像环境里 vLLM 依赖的版本。
#   τ² 声明要 Python ≥ 3.12，镜像是 3.10：--no-deps / --ignore-requires-python 装，tomllib 由 cs_agent/__init__.py 用 tomli 顶上（本地 114 个任务自检通过）。
# 用法：source scripts/remote/setup_tau2.sh   之后 $PY 是装好 τ² 的 python，TAU2_DATA_DIR 已设。
FS=${FS_ROOT:-/root/autodl-fs}; T2=$FS/env/tau2; VE=$FS/env/venv-tau2
mkdir -p $FS/env
(
  flock 8
  if [ ! -f $T2/pyproject.toml ]; then
    # 开机时随代码包上传（autodl_extra --with outputs/cs/tau2_slim.tar），落在代码目录里
    [ -f $FS/env/tau2_slim.tar ] || { [ -f outputs/cs/tau2_slim.tar ] && cp outputs/cs/tau2_slim.tar $FS/env/tau2_slim.tar; }
    [ -f $FS/env/tau2_slim.tar ] || { echo "setup_tau2: $FS/env/tau2_slim.tar missing"; exit 1; }
    mkdir -p $T2.partial && tar -xf $FS/env/tau2_slim.tar -C $T2.partial && mv $T2.partial $T2 || exit 1
  fi
  if [ ! -f $VE/.ok ]; then
    rm -rf $VE
    python -m venv --system-site-packages $VE || exit 1
    $VE/bin/pip install -q --no-deps --ignore-requires-python -e $T2 || exit 1
    # τ² 的依赖里镜像没有的（版本范围同它的 pyproject）；不加 --upgrade，镜像里已有的包保持原版本
    $VE/bin/pip install -q tomli loguru "deepdiff>=8.4.2" "addict>=2.4.0" "docstring-parser>=0.16" "litellm>=1.80.15,<1.82.7" "tenacity>=9.0.0" \
      "tabulate>=0.9.0" toml python-dotenv typer rich || exit 1
    $VE/bin/python -c "import cs_agent, tau2.domains.retail.environment as e; print('tau2 ok', len(e.get_tasks('base')))" || exit 1
    touch $VE/.ok
  fi
) 8>$FS/env/.tau2.lock || return 1 2>/dev/null || exit 1
PY=$VE/bin/python
export TAU2_DATA_DIR=$T2/data
