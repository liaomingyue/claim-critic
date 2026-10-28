"""实验复现记录：每次训练 / 评测自动写一份 run manifest，记下复现这次实验需要的一切。

记录内容：
- 完整配置（训练 yaml 展开后的字典 / 评测命令行参数）；
- 数据指纹：每个输入文件的路径、sha256、行数，jsonl 里有 label / type / origin 字段时附带分布；
- 代码指纹：参与这次实验的源码文件逐个 sha256（项目不是 git 仓库，用文件哈希代替 commit）；
- 环境：Python 和关键库版本、GPU 型号、主机名、AutoDL 实例（环境变量里有就记）；
- 时间、产出位置（权重、LoRA 适配器、评测目录）、训练结果摘要（耗时、末尾 loss）。

manifest 写在产出目录里（<output_dir>/run_manifest.json、<评测目录>/eval_manifest.json），同时追加一行到
本机 outputs/runs/registry.jsonl。本地用 scripts/collect_runs.py 把各台机器的记录汇总成 docs/runs.md。
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

REGISTRY = Path("outputs/runs/registry.jsonl")
CODE_FILES = ["train/sft.py", "train/grpo_trl.py", "ragtruth/task.py", "ragtruth/resp.py", "ragtruth/eval.py",
              "ragtruth/metrics.py", "ragtruth/synth.py", "ragtruth/fulltext.py", "ragtruth/mine.py",
              "ragtruth/iterate.py", "ragtruth/build.py", "ragtruth/annotate.py", "common/llm.py", "common/manifest.py"]
LIBS = ["torch", "transformers", "trl", "peft", "vllm", "datasets", "accelerate"]


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def data_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "missing": True}
    out = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
    if p.suffix == ".jsonl":
        # content_sha：逐行解析后按规范形式（键排序）哈希，与换行符、空白无关。Windows 写出的文件是 \r\n，
        # 字节级 sha256 和服务器上同一份数据对不上（2026-10-27 补录时发现），跨机器核对数据用这个
        n, dist = 0, {k: collections.Counter() for k in ("label", "type", "origin", "task_type")}
        ch = hashlib.sha256()
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ch.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
                for k in dist:
                    if k in row and isinstance(row[k], str):
                        dist[k][row[k]] += 1
        out["rows"] = n
        out["content_sha"] = ch.hexdigest()
        out["distribution"] = {k: dict(v) for k, v in dist.items() if v}
    return out


def code_fingerprint(files: list[str] = CODE_FILES) -> dict:
    return {f: sha256(f)[:16] for f in files if Path(f).exists()}


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(),
           "autodl_instance": os.environ.get("AutoDLContainerUUID") or os.environ.get("AUTODL_INSTANCE")}
    for lib in LIBS:
        try:
            mod = __import__(lib)
            env[lib] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 没装的库记 None
            env[lib] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        env["gpu"] = None
    return env


def write(kind: str, name: str, out_dir: str | Path, config: dict, data: list[str], extra: dict | None = None,
          started: float | None = None) -> dict:
    """写 <out_dir>/<kind>_manifest.json 并追加到 outputs/runs/registry.jsonl；返回 manifest。kind = run | eval。"""
    m = {"kind": kind, "name": name, "out_dir": str(out_dir),
         "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)) if started else None,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
         "duration_min": round((time.time() - started) / 60, 1) if started else None,
         "argv": sys.argv, "config": config, "data": [data_fingerprint(d) for d in data],
         "code": code_fingerprint(), "env": environment(), **(extra or {})}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{kind}_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                                     encoding="utf-8")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
    return m
