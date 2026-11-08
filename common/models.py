"""模型配置（configs/models/<名字>.yaml）：基座路径、加载方式、prompt 共享实现、聊天模板参数、结束符、LoRA、显存（README「多模型支持」）。

  python -m common.models env minicpm5-2b     # 打印 shell 变量，scripts/remote/model_env.sh 用 eval 读进去
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent.parent / "configs" / "models"
DEFAULT = "minicpm5-2b"


def load(name: str = DEFAULT) -> dict:
    p = DIR / f"{name}.yaml"
    if not p.exists():
        raise SystemExit(f"unknown model {name}: no {p} (have: {', '.join(sorted(x.stem for x in DIR.glob('*.yaml')))})")
    m = yaml.safe_load(p.read_text(encoding="utf-8"))
    m.setdefault("loader", "causal")
    m.setdefault("share_impl", "none")
    m.setdefault("chat_kwargs", {})
    m.setdefault("vllm", {})
    return m


def env(m: dict) -> dict[str, str]:
    v = m["vllm"]
    local = "/root/autodl-tmp/models/" + Path(m["path"]).name
    return {"MODEL_NAME": m["name"], "MODEL_FS": m["path"], "MODEL_LOCAL": local, "MODEL_SUFFIX": m["suffix"],
            "MODEL_CAUSAL": "1" if m["loader"] == "causal" else "0", "MODEL_SHARE_IMPL": m["share_impl"],
            "MODEL_SHARE": "0" if m["share_impl"] == "none" else "1", "MODEL_VLLM_MEM": str(v.get("gpu_mem", 0.45)),
            "MODEL_RM_BS": str(m.get("rm_batch", [4, 4])[0]), "MODEL_RM_GA": str(m.get("rm_batch", [4, 4])[1]),
            "MODEL_GRPO_GPUS": str(m.get("gpus", {}).get("grpo", 1)), "MODEL_CHECK_ADAPTER": m.get("check_adapter", "")}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "env":
        for k, val in env(load(sys.argv[2])).items():
            print(f"export {k}={shlex.quote(val)}")
    else:
        raise SystemExit("usage: python -m common.models env <name>")
