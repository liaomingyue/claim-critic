"""R1 报告：重新打分 vs 发布的打分。
  python -m scripts.repro.compare_r1 --cap3 <critic_cap3 输出，基线 rm1__core_r2_bfcl_re> --critics <名字...>
三部分：① 逐句分数差（同一句话两次打分的 |ΔP|）；② 每个 critic 的 AUC，原分数 vs 重打；③ 两条关键结论在重打分数上是否成立：
  表 7  补盲时用 Claude 核实（bfcl）比不核实（bfnc）好：显然错 / 任意错 AUC 差 > 0 且 90% 区间不含 0
  表 9  第二轮 rd2 比第一轮 bfcl 的任意错 AUC 低（静态指标变差），区间不含 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIRS = ["outputs/rt/minicpm/grpo300", "outputs/rt/minicpm/grpo300b", "outputs/rt/minicpm/grpo300c"]


def load(d, name):
    p = Path(d) / f"critic_{name}.json"
    return {r["id"]: float(r["p"]) for r in json.load(open(p, encoding="utf-8"))} if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap3", required=True)
    ap.add_argument("--critics", nargs="+", required=True)
    a = ap.parse_args()
    cap = json.load(open(a.cap3, encoding="utf-8"))
    print("① 逐句分数：同一句话，发布的分数 vs 重新打分")
    print(f"{'critic':22s} {'句数':>6s} {'平均|ΔP|':>9s} {'P95|ΔP|':>8s} {'相关':>6s}")
    for c in a.critics:
        o, r = {}, {}
        for d in DIRS:
            o.update(load(d, c)); r.update(load(d, c + "_re"))
        k = sorted(set(o) & set(r))
        if not k:
            print(f"{c:22s} 没有重叠的句子"); continue
        x = np.array([o[i] for i in k]); y = np.array([r[i] for i in k]); dd = np.abs(x - y)
        print(f"{c:22s} {len(k):6d} {dd.mean():9.4f} {np.quantile(dd, .95):8.4f} {np.corrcoef(x, y)[0, 1]:6.4f}")
    print("\n② AUC：发布的分数 → 重新打分")
    for view in ("clear", "any"):
        print(f"[{view}]")
        for c in a.critics:
            if c in cap[view] and c + "_re" in cap[view]:
                p, q = cap[view][c]["auc"], cap[view][c + "_re"]["auc"]
                print(f"  {c:22s} {p:.3f} → {q:.3f}  ({q - p:+.3f})")
    print("\n③ 关键结论（在重新打分的分数上）")
    ok = True
    clnc = json.load(open("results/critic_cap_blindfill_clnc.json", encoding="utf-8"))
    rd2 = json.load(open("results/critic_cap_rd2.json", encoding="utf-8"))
    for view in ("clear", "any"):   # bfnc_re 相对 bfcl_re 的差取反 = bfcl − bfnc
        v = cap[view].get("rm1__core_r2_bfnc_re")
        if not v:
            continue
        d, lo, hi = -v["d"], -v["dhi"], -v["dlo"]; o = clnc[view]["rm1__core_r2_bfcl"]
        good = lo > 0; ok &= good
        print(f"  表 7 核实 − 不核实 [{view}]：原 {o['d']:+.3f} [{o['dlo']:+.3f},{o['dhi']:+.3f}]  重打 {d:+.3f} [{lo:+.3f},{hi:+.3f}]  {'成立' if good else '不成立'}")
    v = cap["any"].get("rm1__core_rd2_re")
    if v:
        o = rd2["any"]["rm1__core_rd2"]; good = v["dhi"] < 0; ok &= good
        print(f"  表 9 rd2 − bfcl [any]：原 {o['d']:+.3f} [{o['dlo']:+.3f},{o['dhi']:+.3f}]  重打 {v['d']:+.3f} [{v['dlo']:+.3f},{v['dhi']:+.3f}]  {'成立' if good else '不成立'}")
    print("\nR1_CONCLUSIONS_HOLD" if ok else "\nR1_CONCLUSIONS_DIFFER")


if __name__ == "__main__":
    main()
