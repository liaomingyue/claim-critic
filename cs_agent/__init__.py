"""客服 Agent（τ²-bench retail）上的 GRPO 和盲区闭环。

τ²-bench 2.0 声明要 Python ≥ 3.12，但 retail 环境、工具和结果评测只用到 3.11 新增的 tomllib；
3.10 下装 tomli 顶上，就能和训练代码在同一个进程里跑（装法见 scripts/remote/setup_tau2.sh）。
"""
import sys

try:
    import tomllib  # noqa: F401
except ImportError:  # Python 3.10
    import tomli
    sys.modules["tomllib"] = tomli
