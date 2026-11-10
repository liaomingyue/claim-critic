"""critic 各轮对比：召回（阈值 0.5 / 误报 10%）、分数分布、错句排进正确句子前 20% 的比例，按口径和类型。
  python -m scripts.analysis.critic_round_cmp rm1__core rm1__core_r2 rm1__core_r2_bfcl rm1__core_rd2
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from common.schema import read_jsonl
from ragtruth.critic_cap import critic_file, scores
from ragtruth.critic_cap3 import DIRS, rec

names = sys.argv[1:]
rows = [x for d in DIRS for x in read_jsonl(Path(d) / "final.jsonl")]
sc = {c: {k: v for d in DIRS for k, v in (scores(critic_file(d, c)) or {}).items()} for c in names}
rows = [x for x in rows if "unable to answer" not in x["sentence"].lower() and all(x["id"] in sc[c] for c in names)]
neg = [x for x in rows if not x["is_error"]]
negp = {c: np.sort([sc[c][x["id"]] for x in neg]) for c in names}
views = [("显然错", lambda x: x["severity"] == "clear"), ("任意错", lambda x: True)]
zh = {"certainty_or_scope_shift": "范围偏移", "near_synonym_shift": "近义偏移", "wrong_attribution": "张冠李戴", "merged_across_passages": "跨段拼接",
      "contradicts_source": "直接矛盾", "unsupported_inference": "无依据推断"}
views += [(z, (lambda m: lambda x: x["mechanism"] == m)(m)) for m, z in zh.items()]
print("召回：阈值 0.5 / 误报 10%（以误报 10% 为准）")
print("| 口径 | 错句 | " + " | ".join(names) + " |")
for v, keep in views:
    xs = [x for x in rows if not x["is_error"] or keep(x)]
    y = np.array([int(x["is_error"]) for x in xs]); w = np.ones(len(y), int)
    cells = []
    for c in names:
        p = np.array([sc[c][x["id"]] for x in xs])
        cells.append(f"{(p[y == 1] >= 0.5).mean():.2f} / {rec(p, y, w, .10):.2f}")
    print(f"| {v} | {int(y.sum())} | " + " | ".join(cells) + " |")
print("\n分数分布（P<0.1 完全看不见 / ≥0.5）与「排进正确句子前 20%」")
for v, keep in views:
    xs = [x for x in rows if x["is_error"] and keep(x)]
    cells = []
    for c in names:
        p = np.array([sc[c][x["id"]] for x in xs])
        cells.append(f"{(p < .1).mean():.0%} / {(p >= .5).mean():.0%} / 前20% {(p > np.quantile(negp[c], .8)).mean():.0%}")
    print(f"| {v} | " + " | ".join(cells) + " |")
print("| 正确句子 <0.1 / ≥0.5 | " + " | ".join(f"{(negp[c] < .1).mean():.0%} / {(negp[c] >= .5).mean():.1%}" for c in names) + " |")
