"""长句 / 列表句增强（按「合并句子导致 r2 漏判」的方向改，再重跑）。

背景：用 rm1__core_r2 当奖励接着训，策略把内容压进长句、分号列表句；r2 在 9B 显然错句上的召回随句长下降
（15–25 词 0.49、25–35 词 0.27、≥ 35 词 0.14）。一处错混在一长串正确内容里，critic 就看不出来。

做法（不调 Claude，标签全部来自种子集里 Claude 核实过的句子）：
  对核心集训练切分的每个错误对，把错句和它前后相邻的 1–2 句（改对后的句子，Claude 判为正确）拼成一句：
    错误版：相邻正确句 + 原错句 → 标签 = 错句的类型（B 矛盾 / C 无依据）
    正确版：同样的位置拼改对句 → A 忠实（防止 critic 学成「长句就是错」）
  拼法轮换：「; 」、「, and 」、「, while 」、「: a; b; c」（QA 列表式），去掉前句句号、后句常见开头词首字母小写。
  回答里其他句子不动，被拼的位置换成这一句，idx 指向它。holdout 原文不进训练（按 split）。

  python -m ragtruth.merge_aug --base rm1__core_r2 --name rm1__core_r3 [--max_pairs 1000]
  → outputs/rt/rm/rm1__core_r3/{train,calib,test}.jsonl + meta.json（run_mcrm.sh 直接训）；另写 merged_holdout.jsonl（holdout 原文的同法样本，只评测用）
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import zlib
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.critic_harvest import row
from ragtruth.rm_data import OUT, RM1_OPT

LOWER = {"the", "this", "it", "they", "he", "she", "these", "that", "there", "in", "its", "their", "a", "an", "his", "her",
         "both", "some", "many", "according", "after", "before", "during", "on", "at", "as", "while", "when", "we", "you", "one", "each"}
DROP_LEAD = re.compile(r"^(additionally|also|furthermore|moreover|however|meanwhile|then|finally|next|in addition|overall)\s*,?\s*", re.I)


CONJ = {"though", "although", "but", "and", "while", "when", "because", "after", "before", "as", "if", "since", "so", "yet"}


def low(s: str, src: str) -> str:
    """去掉连接副词；首词的小写形式在原文里出现过（不是专名）才转小写。"""
    s = DROP_LEAD.sub("", s.strip())
    if not s:
        return s
    w = re.sub(r"\W", "", s.split(" ", 1)[0])
    common = w.lower() in (LOWER | CONJ) or (w and not w.isupper() and re.search(r"\b" + re.escape(w.lower()) + r"\b", src))
    return s[0].lower() + s[1:] if common else s


def join(parts: list[str], style: int, task: str, src: str) -> str:
    body = [p.strip().rstrip(".!?;") for p in parts]
    if style == 3 and task == "QA":   # 列表式：「Based on the passages: a; b; c.」
        return "Based on the passages: " + "; ".join(low(b, src) for b in body) + "."
    out = body[0]
    for b in body[1:]:
        b = low(b, src)
        first = b.split(" ", 1)[0].lower().strip(",")
        out += ("; " if first in CONJ else ["; ", ", and ", ", while "][style % 3]) + b
    return out + "."


def spans(n: int, idx: int, rng: random.Random) -> tuple[int, int]:
    """包含 idx 的连续区间 [lo, hi]（1 起），长度 2 或 3。"""
    L = 3 if n >= 3 and rng.random() < 0.4 else 2
    cand = [(lo, lo + L - 1) for lo in range(max(1, idx - L + 1), idx + 1) if lo + L - 1 <= n]
    return rng.choice(cand) if cand else (idx, idx)


def build(p: dict, rng: random.Random) -> list[dict] | None:
    sents, idx = list(p["sentences"]), int(p["idx"])
    if len(sents) < 2 or "unable to answer" in " ".join(sents).lower():
        return None
    lo, hi = spans(len(sents), idx, rng)
    if lo == hi:
        return None
    style = rng.randrange(4)
    err = sents[:idx - 1] + [p["real"]] + sents[idx:]
    m_err = join(err[lo - 1:hi], style, p["task_type"], p["source"])
    m_fix = join(sents[lo - 1:hi], style, p["task_type"], p["source"])
    ctx_err = sents[:lo - 1] + [m_err] + sents[hi:]
    ctx_fix = sents[:lo - 1] + [m_fix] + sents[hi:]
    g = p["resp_id"]
    return [row(p["id"] + f"-merr{hi - lo + 1}", p["source"], ctx_err, lo, RM1_OPT[p["type"]], p["task_type"], g, "merge_error"),
            row(p["id"] + f"-mfix{hi - lo + 1}", p["source"], ctx_fix, lo, RM1_OPT["faithful"], p["task_type"], g, "merge_fixed")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="rm1__core_r2")
    ap.add_argument("--name", default="rm1__core_r3")
    ap.add_argument("--pairs", default="outputs/rt/core/pairs.jsonl")
    ap.add_argument("--max_pairs", type=int, default=1000, help="最多用几个训练错误对（每对出 2 行）")
    a = ap.parse_args()
    ps = list(read_jsonl(a.pairs))
    tr = sorted([p for p in ps if p["split"] == "train"], key=lambda p: zlib.crc32(p["id"].encode()))
    add, hold, c = [], [], collections.Counter()
    for p in tr:
        if len(add) >= 2 * a.max_pairs:
            break
        r = build(p, random.Random(p["id"]))
        if r:
            add += r
            c[(p["task_type"], r[0]["id"][-1])] += 1
    for p in ps:
        if p["split"] != "train":
            r = build(p, random.Random(p["id"]))
            hold += r or []
    base, d = OUT / a.base, OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    train = list(read_jsonl(base / "train.jsonl")) + add
    random.Random(0).shuffle(train)
    for name, rows in (("train.jsonl", train), ("merged_holdout.jsonl", hold)):
        with open(d / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "merge_added": len(add), "merge_holdout": len(hold),
            "merge_by_task_len": {f"{k[0]}-{k[1]}": v for k, v in sorted(c.items())},
            "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items())),
                      "origin": dict(sorted(collections.Counter(r["origin"] for r in train).items()))}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MERGE_AUG", json.dumps({k: meta[k] for k in ("merge_added", "merge_holdout", "merge_by_task_len", "train")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
