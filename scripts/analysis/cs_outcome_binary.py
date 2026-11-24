"""三分类结果模型（GOOD / FLAWED / FAIL）去掉 FLAWED、只用 GOOD 和 FAIL 重新归一成二分类 P(正确)，看区分数据库对 / 错的能力。
  python -m scripts.analysis.cs_outcome_binary outputs/rt/cs/critic_outcome/scores.jsonl
标签：数据库对（GOOD + FLAWED）= 正确，FAIL = 错误。
报告：AUC、0.5 阈值的准确率、最优阈值的准确率；再做一次 Platt 校准（logit(P) 上拟合 a·x + b，在 claude 集上拟合、test 集上评，
避免在同一批数据上调参），看校准后的准确率和期望校准误差（ECE）。
"""
from __future__ import annotations

import json
import math
import sys

from cs_agent.critic import auc


def ece(ps, ys, bins=10):
    tot, n = 0.0, len(ps)
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(ps[i] for i in idx) / len(idx) - sum(ys[i] for i in idx) / len(idx))
    return round(tot, 4)


def platt(xs, ys, iters=2000, lr=0.1):
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a * x + b)))
            ga += (p - y) * x
            gb += p - y
        a -= lr * ga / len(xs)
        b -= lr * gb / len(xs)
    return a, b


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    data = {}
    for r in rows:
        g, f = r["probs"]["GOOD"], r["probs"]["FAIL"]
        p = g / max(1e-9, g + f)
        data.setdefault(r["split"], []).append((p, 1 if r["gold"] in ("GOOD", "FLAWED") else 0))
    logit = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))
    fit = None
    if "claude" in data:
        xs, ys = [logit(p) for p, _ in data["claude"]], [y for _, y in data["claude"]]
        fit = platt(xs, ys)
    out = {}
    for sp, d in sorted(data.items()):
        ps, ys = [p for p, _ in d], [y for _, y in d]
        acc = lambda thr: sum((p >= thr) == bool(y) for p, y in d) / len(d)
        best = max((acc(t / 100), t / 100) for t in range(5, 96))
        rec = {"n": len(d), "pos_rate": round(sum(ys) / len(ys), 3), "auc": auc([p for p, y in d if y], [p for p, y in d if not y]),
               "acc@0.5": round(acc(0.5), 4), "best_acc": round(best[0], 4), "best_thr": best[1], "ece": ece(ps, ys),
               "mean_p_correct": round(sum(p for p, y in d if y) / max(1, sum(ys)), 3), "mean_p_wrong": round(sum(p for p, y in d if not y) / max(1, len(ys) - sum(ys)), 3)}
        if fit:
            cal = [1 / (1 + math.exp(-(fit[0] * logit(p) + fit[1]))) for p in ps]
            rec.update({"platt_a_b": [round(fit[0], 3), round(fit[1], 3)], "cal_acc@0.5": round(sum((c >= 0.5) == bool(y) for c, y in zip(cal, ys)) / len(ys), 4),
                        "cal_ece": ece(cal, ys)})
        out[sp] = rec
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
