"""critic 静态能力评测（扩充 Claude 集；2026-11-05 起的主口径，原在 scratchpad cap3.py / cap3n.py，2026-11-06 收进仓库）。

三批验收回答（grpo300 / grpo300b / grpo300c，Claude 金标 v1 逐句）：显然错 vs 没错（轻微错不进），默认去掉拒答句
（GRPO 里拒答走 refuse 模型、不经 critic）。排序法 AUC，按原文 bootstrap（1000 次，90%），和 --base 配对比；误报 10% / 5% 时的召回。
critic 分数文件：<批>/critic_<名>.json（rm1__mcgold 是 critic2b_scores.json）。

  python -m ragtruth.critic_cap3 --critics rm1__mcgold_ref rm1__mcgold_hum rm1__mcgold_syn --base rm1__mcgold_ref --out outputs/rt/critic_cap_humsyn.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from common.schema import read_jsonl
from ragtruth.critic_cap import critic_file, scores

DIRS = ["outputs/rt/minicpm/grpo300", "outputs/rt/minicpm/grpo300b", "outputs/rt/minicpm/grpo300c"]


def auc(p, y, w):   # 加权（bootstrap 抽中次数）Mann-Whitney
    r = rankdata(np.repeat(p, w)); yy = np.repeat(y, w); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rec(p, y, w, fp):
    neg = np.sort(np.repeat(p[y == 0], w[y == 0]))[::-1]; t = neg[int(fp * len(neg))]
    return (np.repeat(p[y == 1], w[y == 1]) > t).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critics", nargs="+", required=True)
    ap.add_argument("--base", default="rm1__mcgold_ref")
    ap.add_argument("--with_refusal", action="store_true", help="保留拒答句（默认去掉）")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = list(dict.fromkeys([a.base, *a.critics]))
    rows = [dict(x, _set=d[-1]) for d in DIRS for x in read_jsonl(Path(d) / "final.jsonl")]
    sc = {c: {k: v for d in DIRS for k, v in (scores(critic_file(d, c)) or {}).items()} for c in names}
    xs = [x for x in rows if (a.with_refusal or "unable to answer" not in x["sentence"].lower())
          and ((x["is_error"] and x["severity"] == "clear") or not x["is_error"]) and all(x["id"] in sc[c] for c in names)]
    y = np.array([int(x["is_error"]) for x in xs]); P = {c: np.array([sc[c][x["id"]] for x in xs]) for c in names}
    us, inv = np.unique(np.array([str(x["source_id"]) for x in xs]), return_inverse=True)
    print("sentences", len(xs), "clear", int(y.sum()), "sources", len(us), "by set", dict(collections.Counter((x["_set"], x["is_error"]) for x in xs)))
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))[inv] for _ in range(a.n_boot)]
    one = np.ones(len(y), int)
    ab = np.array([auc(P[a.base], y, w) for w in W]); rb = np.array([rec(P[a.base], y, w, .10) for w in W])
    stat = {}
    for c in names:
        aa = np.array([auc(P[c], y, w) for w in W]); r10 = np.array([rec(P[c], y, w, .10) for w in W])
        stat[c] = dict(auc=auc(P[c], y, one), lo=np.percentile(aa, 5), hi=np.percentile(aa, 95), d=auc(P[c], y, one) - auc(P[a.base], y, one),
                       dlo=np.percentile(aa - ab, 5), dhi=np.percentile(aa - ab, 95), p=(aa - ab <= 0).mean(),
                       r10=rec(P[c], y, one, .10), dr=rec(P[c], y, one, .10) - rec(P[a.base], y, one, .10),
                       drlo=np.percentile(r10 - rb, 5), drhi=np.percentile(r10 - rb, 95), r5=rec(P[c], y, one, .05), fp05=float((P[c][y == 0] >= .5).mean()))
    print(f"critic | clearAUC [90%] | dAUC vs {a.base} [90%] P<=0 | rec@fp10 (d [90%]) | rec@fp5 | fp@0.5")
    for c in sorted(names, key=lambda c: -stat[c]["auc"]):
        s = stat[c]
        print(f"{c:28s} | {s['auc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] | {s['d']:+.3f} [{s['dlo']:+.3f},{s['dhi']:+.3f}] {s['p']:.3f} | "
              f"{s['r10']:.2f} ({s['dr']:+.2f} [{s['drlo']:+.2f},{s['drhi']:+.2f}]) | {s['r5']:.2f} | {s['fp05']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps({c: {k: float(v) for k, v in stat[c].items()} for c in names}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
