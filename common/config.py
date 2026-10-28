from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg and cfg.get("trace_dir"):  # 打开模型调用记录（common.trace）
        os.environ.setdefault("CRITIC_TRACE_DIR", str(cfg["trace_dir"]))
    return cfg
