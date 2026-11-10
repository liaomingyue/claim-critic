"""critic 盲区的形态：Claude 金标（三套验收回答，去掉拒答句）里真错却没判出（P < 阈值）的句子，P 落在哪儿。
用法：python -m scripts.analysis.blind_dist [critic ...]"""
import sys
import collections as C
from pathlib import Path
import numpy as np
from common.schema import read_jsonl
from ragtruth.critic_cap import critic_file, scores
from ragtruth.critic_cap3 import DIRS

names = sys.argv[1:] or ["rm1__core", "rm1__core_r2"]
rows = [dict(x, _set=d[-1]) for d in DIRS for x in read_jsonl(Path(d) / "final.jsonl") if "unable to answer" not in x["sentence"].lower()]
BINS = [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5)]
for c in names:
    sc = {k: v for d in DIRS for k, v in (scores(critic_file(d, c)) or {}).items()}
    xs = [x for x in rows if x["id"] in sc]
    neg = np.array([sc[x["id"]] for x in xs if not x["is_error"]])
    t10 = float(np.quantile(neg, .90))          # 误报 10% 的阈值
    print(f"\n==== {c}  句子 {len(xs)}，没错 {len(neg)}；误报 10% 阈值 = {t10:.3f}；没错句子 P 中位数 {np.median(neg):.3f}")
    for sev in ("clear", "minor"):
        pos = [sc[x["id"]] for x in xs if x["is_error"] and x["severity"] == sev]
        miss = [p for p in pos if p < .5]
        print(f"  [{sev}] 共 {len(pos)}，P≥0.5 判出 {len(pos) - len(miss)}（{100 * (len(pos) - len(miss)) / len(pos):.0f}%），漏 {len(miss)}；误报 10% 阈值下漏 {sum(p < t10 for p in pos)}")
        print("    漏掉的 P 分布：" + "  ".join(f"[{lo},{hi}) {sum(lo <= p < hi for p in miss)}（{100 * sum(lo <= p < hi for p in miss) / max(len(miss), 1):.0f}%）" for lo, hi in BINS))
        print(f"    参照：没错句子 P 分布：" + "  ".join(f"[{lo},{hi}) {100 * ((neg >= lo) & (neg < hi)).mean():.0f}%" for lo, hi in BINS) + f"  ≥0.5 {100 * (neg >= .5).mean():.0f}%")
