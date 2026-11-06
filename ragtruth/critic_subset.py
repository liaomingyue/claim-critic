"""从已有 critic 训练集里按来源剔除样本（2026-11-05，用户：RAGTruth 原版人工标注漏标很多，最好不要了；早期合成只抽查过）。
校准 / 测试集原样拷贝。--drop 写「来源」或「来源:标签」，例如 human:A 只删人工里标忠实的。

  python -m ragtruth.critic_subset --base rm1__mcgold_ref --name rm1__mcgold_ref_nohA --drop human:A
  python -m ragtruth.critic_subset --base rm1__mcgold_ref --name rm1__mcgold_ref_noh --drop human
  python -m ragtruth.critic_subset --base rm1__mcgold_ref --name rm1__mcgold_ref_nohs --drop human synth
纯人工 / 纯合成（2026-11-06，用户：对比纯合成和人工数据）：--keep 只留这些来源（前缀匹配），--cap_groups 来源前缀 N 只留该来源前 N 对（文件顺序；
错句 <x>-err 和配对原句 <x>-orig 算一对）：
  python -m ragtruth.critic_subset --base rm1__mcgold_ref --name rm1__mcgold_hum --keep human refuse_
  python -m ragtruth.critic_subset --base rm1__mcgold_ref_n1500 --name rm1__mcgold_syn --keep synth harvest_ refuse_ --cap_groups harvest_ 805
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil

from common.schema import read_jsonl
from ragtruth.rm_data import OUT


def dropped(r: dict, specs: list[tuple[str, str | None]]) -> bool:
    return any(r.get("origin") == o and (lab is None or r["label"] == lab) for o, lab in specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--drop", nargs="+", default=[], help="来源 或 来源:标签")
    ap.add_argument("--keep", nargs="+", default=[], help="只留这些来源（前缀匹配）")
    ap.add_argument("--cap_groups", nargs=2, default=None, metavar=("PREFIX", "N"), help="该来源前缀只留前 N 对（错句 + 配对原句）")
    a = ap.parse_args()
    assert a.drop or a.keep, "--drop 或 --keep 至少给一个"
    specs = [(s.split(":")[0], s.split(":")[1] if ":" in s else None) for s in a.drop]
    base, d = OUT / a.base, OUT / a.name
    rows = list(read_jsonl(base / "train.jsonl"))
    keep = [r for r in rows if not dropped(r, specs) and (not a.keep or any(str(r.get("origin")).startswith(k) for k in a.keep))]
    if a.cap_groups:
        pre, n = a.cap_groups[0], int(a.cap_groups[1])
        # 按「对」截：harvest 的错句 id = <x>-err、配对原句 = <x>-orig（group 是按原文的，一个原文可有 2 对，不能用 group 截 —— 2026-11-06 踩过）
        pair = lambda r: re.sub(r"-(err|orig)$", "", r["id"])
        seen: dict[str, None] = {}
        for r in keep:
            if str(r.get("origin")).startswith(pre):
                seen.setdefault(pair(r))
        ok = set(list(seen)[:n])
        keep = [r for r in keep if not str(r.get("origin")).startswith(pre) or pair(r) in ok]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    c = collections.Counter
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "drop": a.drop, "keep": a.keep, "cap_groups": a.cap_groups, "dropped": len(rows) - len(keep),
            "train": {"n": len(keep), "label": dict(sorted(c(r["label"] for r in keep).items()))},
            "by_origin": dict(sorted(c(f"{r.get('origin')}|{r['label']}" for r in keep).items()))}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CRITIC_SUBSET", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
