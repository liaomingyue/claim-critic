"""下载并解压开源发布的数据和模型，核对 SHA-256（2026-11-10）。

  pip install huggingface_hub
  python -m scripts.fetch_release --data_repo <用户>/claim-critic-data --model_repo <用户>/claim-critic-models [--only data|models] [--parts eval core]
  python -m scripts.fetch_release --local /path/to/release        # 已经有本地副本（目录里有 MANIFEST.json 和两个子目录）时直接从本地解压

解压位置：数据解压到仓库根目录（outputs/...、external_data/...），模型解压到仓库根目录的 adapters/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ["eval", "core", "synth", "rm", "ragtruth"]
MODELS = ["critic", "rag", "gen", "rm"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_repo", default="")
    ap.add_argument("--model_repo", default="")
    ap.add_argument("--local", default="", help="本地发布目录（含 MANIFEST.json、claim-critic-data/、claim-critic-models/）")
    ap.add_argument("--only", choices=["data", "models"], default=None)
    ap.add_argument("--parts", nargs="*", default=None, help="只要这些部分，如 eval core critic rag")
    ap.add_argument("--dest", default=str(ROOT))
    a = ap.parse_args()
    want = [("claim-critic-data", p, ".tar.gz") for p in DATA if a.only in (None, "data")] + \
           [("claim-critic-models", p, ".tar") for p in MODELS if a.only in (None, "models")]
    if a.parts:
        want = [w for w in want if w[1] in a.parts]
    if a.local:
        base = Path(a.local)
        man = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
        paths = {f"{d}/{p}{ext}": base / d / f"{p}{ext}" for d, p, ext in want}
    else:
        from huggingface_hub import hf_hub_download
        man_path = hf_hub_download(a.data_repo, "MANIFEST.json", repo_type="dataset")
        man = json.loads(Path(man_path).read_text(encoding="utf-8"))
        paths = {}
        for d, p, ext in want:
            repo, kind = (a.data_repo, "dataset") if d == "claim-critic-data" else (a.model_repo, "model")
            paths[f"{d}/{p}{ext}"] = Path(hf_hub_download(repo, f"{p}{ext}", repo_type=kind))
    dest = Path(a.dest)
    for name, p in paths.items():
        if not p.exists():
            print("missing", name); continue
        print("extract", name)
        with tarfile.open(p) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:   # Python < 3.10.12 没有 filter 参数
                t.extractall(dest)
    bad = 0
    for f in man["files"]:
        if f["tar"] not in paths:
            continue
        q = dest / f["path"]
        if not q.exists() or q.stat().st_size != f["bytes"] or sha(q) != f["sha256"]:
            bad += 1
            print("MISMATCH", f["path"])
    print("FETCH_DONE", len(paths), "archives;", "mismatches", bad)


if __name__ == "__main__":
    main()
