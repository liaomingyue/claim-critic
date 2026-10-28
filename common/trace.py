"""模型调用记录：每次 LLM.chat 把输入 prompt、输出、模型和耗时追加写到 {trace_dir}/{stage}.jsonl，
供审核台展示每条样本的完整调用链。

- 开关：配置文件顶层写 trace_dir（common.config.load_config 读到后设置），或环境变量 CRITIC_TRACE_DIR。
- stage 和样本 id 由 map_resumable 在处理每条 item 时设置（stage = 输出文件名，如 reviews、positives、synth_batch1），
  阶段代码不用改。不在 map_resumable 里的调用记到 stage=adhoc。
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_local = threading.local()
_lock = threading.Lock()


def trace_dir() -> Path | None:
    d = os.environ.get("CRITIC_TRACE_DIR")
    return Path(d) if d else None


@contextmanager
def item(stage: str, item_id: str):
    prev = getattr(_local, "ctx", None)
    _local.ctx = {"stage": stage, "id": item_id, "call": 0}
    try:
        yield
    finally:
        _local.ctx = prev


def reset_usage() -> None:
    _local.usage = None


def add_usage(usage: dict) -> None:
    """后端在一次 chat 内累计真实用量（token 数、按标价折算的费用），record 时一并写入。"""
    acc = getattr(_local, "usage", None) or {}
    for k, v in usage.items():
        if isinstance(v, (int, float)):
            acc[k] = acc.get(k, 0) + v
    _local.usage = acc


def record(model: str, provider: str, messages: list[dict], outputs: list[str], secs: float) -> None:
    usage, _local.usage = getattr(_local, "usage", None), None
    d = trace_dir()
    if d is None:
        return
    ctx = getattr(_local, "ctx", None) or {"stage": "adhoc", "id": None, "call": 0}
    row = {"stage": ctx["stage"], "id": ctx["id"], "call": ctx["call"], "model": model, "provider": provider,
           "system": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
           "user": "\n\n".join(m["content"] for m in messages if m["role"] != "system"),
           "outputs": outputs, "secs": round(secs, 2), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "usage": usage}
    ctx["call"] += 1
    d.mkdir(parents=True, exist_ok=True)
    with _lock, open(d / f"{ctx['stage']}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load(stage_file: Path) -> dict[str, list[dict]]:
    """读一个阶段的调用记录，按样本 id 分组、按调用顺序排。"""
    out: dict[str, list[dict]] = {}
    if stage_file.exists():
        for line in open(stage_file, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["id"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: (r["ts"], r["call"]))
    return out
