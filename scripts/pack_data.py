"""打包交给别人复现用的数据（不进 git：outputs/ 和 external_data/ 在 .gitignore 里）。

只打「重跑也得不到同样结果」的东西：Claude 的标注、外部数据原件、评测集；能由代码重新生成的中间文件不打。
写 dist/claim_critic_data.tar 和 dist/MANIFEST.json（每个文件的行数、字节数、SHA-256）。还没生成的文件跳过并在清单里标 missing。

  python -m scripts.pack_data            # 打包
  python -m scripts.pack_data --check    # 只核对清单，不打包（拿到数据的人解压后跑这个）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# (路径, 说明)。目录会整个打进去。
FILES = [
    ("external_data/ragtruth/dataset/response.jsonl", "RAGTruth 原件（github.com/ParticleMedia/RAGTruth，MIT）：回答 + 人工幻觉片段"),
    ("external_data/ragtruth/dataset/source_info.jsonl", "RAGTruth 原件：原文"),
    ("external_data/ragtruth/LICENSE", "RAGTruth 许可证"),
    ("external_data/ragtruth_pp/hallucination_spans.csv", "RAGTruth++ 重标的幻觉片段"),
    ("outputs/rt/ansgrpo/train_all.jsonl", "第一步：831 个训练原文（ans_data split）"),
    ("outputs/rt/minicpm/grpo/train_ref.jsonl", "第一步：2B 初始策略的原回答 ref_text（采样有随机性，打包以免重采不同）"),
    ("outputs/rt/minicpm/s1/fix_raw.jsonl", "第二步：Claude 逐句改对（批量）"),
    ("outputs/rt/minicpm/s1/fix_single_raw.jsonl", "第二步：Claude 逐句改对（单条补漏）"),
    ("outputs/rt/core/label_raw.jsonl", "第二步（新增）：Claude 错误类型和满意度标注"),
    ("outputs/rt/core/sev_label_raw.jsonl", "第四步（新增）：Claude 单独标注的改写样本"),
    ("outputs/rt/minicpm/grpo300", "评测：验收集 a（answers.jsonl 原文-种子，final.jsonl Claude 金标）"),
    ("outputs/rt/minicpm/grpo300b", "评测：验收集 b"),
    ("outputs/rt/minicpm/grpo300c", "评测：验收集 c"),
    ("outputs/rt/rm/rm1/calib.jsonl", "评测：RAGTruth++ 校准 427 句"),
    ("outputs/rt/rm/rm1/test.jsonl", "评测：RAGTruth++ 测试 2306 句"),
]
KEEP_IN_DIR = ("answers.jsonl", "final.jsonl")   # 验收集目录里只要这两个，其余是各次实验的打分


def files_of(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.is_file() and f.name in KEEP_IN_DIR) if p.is_dir() else []


def info(f: Path) -> dict:
    b = f.read_bytes()
    return {"bytes": len(b), "lines": b.count(b"\n"), "sha256": hashlib.sha256(b).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    man_path = Path(a.out) / "MANIFEST.json"
    if a.check:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        bad = [k for k, v in man["files"].items() if v != "missing" and (not Path(k).exists() or info(Path(k))["sha256"] != v["sha256"])]
        print("ok" if not bad else f"{len(bad)} mismatched: {bad}")
        return
    man = {"files": {}, "notes": {}}
    got = []
    for path, note in FILES:
        fs = files_of(Path(path))
        man["notes"][path] = note
        if not fs:
            man["files"][path] = "missing"
            print("missing", path)
        for f in fs:
            man["files"][f.as_posix()] = info(f)
            got.append(f)
    Path(a.out).mkdir(exist_ok=True)
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    with tarfile.open(Path(a.out) / "claim_critic_data.tar", "w") as t:
        for f in got:
            t.add(f, arcname=f.as_posix())
        t.add(man_path, arcname="MANIFEST.json")
    print(f"packed {len(got)} files → {a.out}/claim_critic_data.tar")


if __name__ == "__main__":
    main()
