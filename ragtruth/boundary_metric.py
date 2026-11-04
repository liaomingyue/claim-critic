"""奖励对比的新口径（2026-11-03，用户：critic 越接近 0.5 越合适；Claude 抽查 2B P(判出) < 0.1 的 33 条只有 11 条是真错）：
「骗过 2B critic（P < 0.5）」混进了大量 RM3 误判的非错误句，改看边界样本：合格样本里 2B critic P(判出) 在 [lo, hi] 之间的，
以及其中 9B 金标 critic 判得出（P ≥ 0.5，确实是错、只是 2B 拿不准）的。读 gen_indep_eval 写的 {run}/indep_s{step}.jsonl。

  python -m ragtruth.boundary_metric outputs/rt/gentarget/grpo_seed5_a0 outputs/rt/gentarget/grpo_seed5_a1 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.7)
    a = ap.parse_args()
    print("组 | 步 | 合格 | 边界(0.3–0.7) 条数 / 占比 | 边界且 9B 判出 条数 | 边界里第一版 RM2 判自然 | P<0.1 条数")
    for run in a.runs:
        for f in sorted(Path(run).glob("indep_s*.jsonl"), key=lambda p: int(p.stem[7:])):
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            b = [r for r in rows if a.lo <= r["p_halluc_2b"] <= a.hi]
            v = [r for r in b if r["p_halluc_9b"] >= 0.5]
            nat = sum(r["p_natural_v1"] > 0.5 for r in b) / max(1, len(b))
            print(f"{Path(run).name} | {f.stem[7:]} | {len(rows)} | {len(b)} / {len(b) / max(1, len(rows)):.1%} | {len(v)} | {nat:.0%} | {sum(r['p_halluc_2b'] < 0.1 for r in rows)}")


if __name__ == "__main__":
    main()
