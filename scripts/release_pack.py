"""开源发布打包（2026-11-10）：按 release_inventory 的清单，在 GPU 机上把数据和模型打成 tar，并写 MANIFEST.json（逐文件 SHA-256）。

  python -m scripts.release_pack [--fs /root/autodl-fs] [--out /root/autodl-tmp/release]

产出（--out 下）：
  claim-critic-data/{eval,core,synth,rm,ragtruth}.tar.gz   成员路径 = 仓库相对路径（outputs/...、external_data/...），在仓库根目录解压即可
  claim-critic-models/{critic,rag,gen,rm}.tar               成员路径 = adapters/<名字>/...（safetensors 不再压缩）
  MANIFEST.json                                            每个文件：所在 tar、路径、字节数、SHA-256、分组、说明
不打包：
  RAGTruth++（blue-guardrails/ragtruth-plus-plus，数据卡未写许可证）派生的测试集 / 校准集，用 ragtruth/rtpp.py 从原地址自建
  基座模型（MiniCPM5-2B、Qwen3.5-9B），只发 LoRA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from scripts.release_inventory import entries

C = "code/critic_model/"
DATA_TAR = {"data/eval": "eval", "data/core": "core", "data/synth": "synth", "data/rm": "rm"}
MODEL_TAR = {"models/critic": "critic", "models/rag": "rag", "models/gen": "gen", "models/rm": "rm"}
SKIP_NAMES = {"labels_raw.jsonl"}   # Claude 原始返回（已解析进 labels.jsonl）


def is_rtpp(p: Path) -> bool:
    """RAGTruth++ 派生的集合：第一行没有 origin、有 coarse / claude 字段（rm1/test.jsonl 和各 critic 目录下的 test.jsonl）。"""
    if p.suffix != ".jsonl" or p.name not in ("test.jsonl", "calib.jsonl"):
        return False
    with open(p, encoding="utf-8") as f:
        first = f.readline()
    try:
        x = json.loads(first)
    except ValueError:
        return False
    return "origin" not in x and "coarse" in x


def model_name(rel: str) -> str:
    """共享盘路径 → 发布后的 adapters/<名字>。"""
    if rel.startswith("adapters/"):
        parts = rel.split("/")
        name = parts[1].removesuffix("-mc").removesuffix("-9b")
        rest = "/".join(parts[2:])
        if rest.startswith("step"):
            return f"adapters/{name}/{rest.split('/')[0]}"
        return f"adapters/{name}"
    if "gen_v2_grpo/g1/ckpt/step50" in rel:
        return "adapters/gen-g1/step50"
    return "adapters/" + Path(rel).name


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="/root/autodl-fs")
    ap.add_argument("--out", default="/root/autodl-tmp/release")
    a = ap.parse_args()
    fs, out = Path(a.fs), Path(a.out)
    (out / "claim-critic-data").mkdir(parents=True, exist_ok=True)
    (out / "claim-critic-models").mkdir(parents=True, exist_ok=True)
    tars, manifest, skipped = {}, [], []

    def tar_for(group: str) -> tuple[tarfile.TarFile, str]:
        if group in DATA_TAR:
            name = f"claim-critic-data/{DATA_TAR[group]}.tar.gz"
            mode = "w:gz"
        else:
            name = f"claim-critic-models/{MODEL_TAR[group]}.tar"
            mode = "w"
        if name not in tars:
            tars[name] = tarfile.open(out / name, mode)
        return tars[name], name

    def add(group: str, src: Path, arc: str, desc: str) -> None:
        files = sorted(f for f in src.rglob("*") if f.is_file()) if src.is_dir() else [src]
        t, tname = tar_for(group)
        for f in files:
            if f.name in SKIP_NAMES or is_rtpp(f):
                skipped.append(str(f))
                continue
            member = arc if not src.is_dir() else f"{arc}/{f.relative_to(src).as_posix()}"
            t.add(f, arcname=member)
            manifest.append({"tar": tname, "path": member, "bytes": f.stat().st_size, "sha256": sha(f), "group": group, "desc": desc})

    for group, rel, desc in entries():
        src = fs / rel
        if not src.exists():
            skipped.append(f"MISSING {rel}")
            continue
        if group.startswith("data/"):
            add(group, src, rel.removeprefix(C), desc)
        else:
            add(group, src, model_name(rel), desc)
    # RAGTruth 原件（MIT）
    rt = fs / C / "external_data/ragtruth"
    if rt.exists():
        tars["claim-critic-data/ragtruth.tar.gz"] = tarfile.open(out / "claim-critic-data/ragtruth.tar.gz", "w:gz")
        for f in sorted(x for x in rt.rglob("*") if x.is_file()):
            arc = "external_data/ragtruth/" + f.relative_to(rt).as_posix()
            tars["claim-critic-data/ragtruth.tar.gz"].add(f, arcname=arc)
            manifest.append({"tar": "claim-critic-data/ragtruth.tar.gz", "path": arc, "bytes": f.stat().st_size, "sha256": sha(f),
                             "group": "data/ragtruth", "desc": "RAGTruth 原件（github.com/ParticleMedia/RAGTruth，MIT）"})
    for t in tars.values():
        t.close()
    sizes = {k: (out / k).stat().st_size for k in tars}
    (out / "MANIFEST.json").write_text(json.dumps({"files": manifest, "tars": sizes, "skipped": skipped}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in sorted(sizes.items()):
        print(f"{k:42s} {v / 2**20:9.1f} MB")
    print("files", len(manifest), "skipped", len(skipped))
    for s in skipped[:30]:
        print("  skip", s)


if __name__ == "__main__":
    main()
