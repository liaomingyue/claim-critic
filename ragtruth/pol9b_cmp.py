"""策略独立评测对比：各组在同一批验收原文-种子上的生成，用 9B 金标 critic 逐句打分（9B 只当评测标签）。
按回答 id 配对，按原文聚类 bootstrap（2000 次，90% 区间）。

  python -m ragtruth.pol9b_cmp --arm v0=outputs/rt/minicpm/r3full/eval:v0 --arm h3ctl=outputs/rt/minicpm/r3full/eval:s50 \
      --sets a b c --base v0 [--out x.json]
每组 <名>=<目录>:<前缀>，读 <目录>/<前缀>_<套>.jsonl 和 <前缀>_<套>_9b.json。
指标（每条回答）：clear = 显然错句数（9B p ≥ 0.9）、any_clear = 含显然错、any_err = 含任意错（≥ 0.5）、refuse = 拒答、chars、sents。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common.schema import read_jsonl
from train.ans_reward import REFUSAL

METRICS = ("any_clear", "clear", "any_err", "refuse", "chars", "sents")
SID = re.compile(r"-S\d+$")


def load_arm(d: str, prefix: str, sets: list[str]) -> dict:
    out = {}
    for s in sets:
        ans = Path(d) / f"{prefix}_{s}.jsonl"
        sc = Path(d) / f"{prefix}_{s}_9b.json"
        if not (ans.exists() and sc.exists()):
            continue
        ps: dict[str, list[float]] = {}
        for x in json.loads(sc.read_text(encoding="utf-8")):
            ps.setdefault(SID.sub("", x["id"]), []).append(float(x["p"]))
        for r in read_jsonl(ans):
            if r.get("finish") != "stop":
                continue
            p = ps.get(r["id"], [])
            refuse = bool(REFUSAL.search(r["response"] or ""))
            out[f"{s}:{r['id']}"] = {"src": f"{s}:{r['source_id']}", "task": r["task_type"],
                                     "any_clear": float(any(v >= 0.9 for v in p)), "clear": float(sum(v >= 0.9 for v in p)),
                                     "any_err": float(any(v >= 0.5 for v in p)), "refuse": float(refuse),
                                     "chars": float(len(r["response"] or "")), "sents": float(len(p))}
    return out


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired(a: dict, b: dict, keys: list[str], m: str, n_boot: int, seed: int) -> dict:
    by_src: dict[str, list[float]] = {}
    for k in keys:
        by_src.setdefault(a[k]["src"], []).append(b[k][m] - a[k][m])
    srcs = sorted(by_src)
    d = mean([v for s in srcs for v in by_src[s]])
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        vs = [v for s in (rng.choice(srcs) for _ in srcs) for v in by_src[s]]
        bs.append(mean(vs))
    bs.sort()
    return {"delta": round(d, 4), "ci": [round(bs[int(0.05 * n_boot)], 4), round(bs[int(0.95 * n_boot) - 1], 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="名=目录:前缀")
    ap.add_argument("--sets", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--base", default=None, help="和哪一组配对比（默认第一组）")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        arms[name] = load_arm(d, prefix, a.sets)
    base = a.base or next(iter(arms))
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    rep = {"answers_paired": len(common), "sources": len({arms[base][k]["src"] for k in common}), "arms": {}, "vs_" + base: {}}
    for n, v in arms.items():
        rep["arms"][n] = {"n_all": len(v)}
        for t in ("all", "QA", "Summary", "Data2txt"):
            ks = [k for k in common if t == "all" or v[k]["task"] == t]
            if ks:
                rep["arms"][n][t] = {m: round(mean([v[k][m] for k in ks]), 4) for m in METRICS} | {"n": len(ks)}
    for n in arms:
        if n != base:
            rep["vs_" + base][n] = {m: paired(arms[base], arms[n], common, m, a.n_boot, a.seed) for m in METRICS}
    if len(arms) > 2:   # 非 base 组两两
        names = [n for n in arms if n != base]
        rep["pairs"] = {f"{x}->{y}": {m: paired(arms[x], arms[y], common, m, a.n_boot, a.seed) for m in ("any_clear", "clear", "any_err", "refuse")}
                        for i, x in enumerate(names) for y in names[i + 1:]}
    s = json.dumps(rep, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()
