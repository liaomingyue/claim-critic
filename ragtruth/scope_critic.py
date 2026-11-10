"""给 critic 补「范围偏移」样本（约 40 次 Claude 调用）。

流程：生成器 v2（9B）在 genpool_div（和 sev 训练池、critic 评测原文都不重叠）上按范围偏移采样
  → 9B 基座写改动描述 → sev（v3c / v3c0 里范围偏移上更好的）+ RM2 v3b + 当前 critic r2 打分
  → 过滤：sev P(要求的那种) ≥ --sev、RM2 自然 ≥ --nat、critic 在 [--lo, --hi)，每条回答最多 1 条
  → Claude 核实（gen_v2_label 同一提示词，out_dir scope_label）
  → 两组等量加进 r2 训练集（每条错句配同位置原句）：
       cl  Claude 判真错 + 确为范围偏移 + 自然
       nc  同一过滤池里随机取同样条数（不看 Claude）
  python -m ragtruth.scope_critic items | pick_sev | filter | build
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

import numpy as np

from common.schema import read_jsonl

D = Path("outputs/rt/core")
MECH = "certainty_or_scope_shift"


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_items(a) -> None:
    xs = [x for x in read_jsonl(D / a.samples) if x.get("why") == "ok" and x["mechanism"] == MECH]
    out = [{**x, "aid": "9b/" + x["id"], "group": "9b"} for x in sorted(xs, key=lambda x: x["id"])]
    write(D / "scope_pool" / "items.jsonl", out)
    print("SCOPE_ITEMS", len(out), "responses", len({x["resp_id"] for x in out}), flush=True)


def auc(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), len(y) - y.sum()
    if not n1 or not n0:
        return float("nan")
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)   # 平均秩（远端没有 scipy）
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_pick_sev(a) -> None:
    """范围偏移要求下，各版 sev 区分「确为范围偏移的真错」vs 其他（全部 / 只看真错）的 AUC；全量 Claude 标注（gen_v2_label + r2）。"""
    lab = {}
    for sub in ("gen_v2_label", "gen_v2_label_r2"):
        it = {x["aid"]: x for x in read_jsonl(D / sub / "items.jsonl")}
        for r in read_jsonl(D / sub / "labels.jsonl"):
            if r["aid"] in it and it[r["aid"]]["mechanism"] == MECH:
                lab[r["aid"]] = r["claude"]
    best, res = None, {}
    for v in a.versions:
        f = D / "gen_v2_label" / f"model_scores_{v}.jsonl"
        if not f.exists():
            continue
        sc = {r["aid"]: r.get("v3_p_req") for r in read_jsonl(f)}
        ks = [k for k in lab if sc.get(k) is not None]
        ys = [bool(lab[k].get("matches")) for k in ks]
        e = [k for k in ks if lab[k].get("severity") in ("clear", "minor")]
        res[v] = {"n": len(ks), "auc_all": round(auc([sc[k] for k in ks], ys), 3),
                  "auc_err": round(auc([sc[k] for k in e], [bool(lab[k].get("matches")) for k in e]), 3)}
        if v != "v3b" and (best is None or res[v]["auc_all"] > res[best]["auc_all"]):
            best = v
    print("PICK_SEV", json.dumps(res), "best", best, flush=True)
    (D / "scope_pool" / "sev_pick.txt").write_text(best or "v3c", encoding="utf-8")


def cmd_filter(a) -> None:
    it = {x["aid"]: x for x in read_jsonl(D / "scope_pool" / "items.jsonl")}
    sc = {r["aid"]: r for r in read_jsonl(D / "scope_pool" / "scores.jsonl")}
    c = collections.Counter()
    ok = []
    for k in sorted(it):
        s = sc.get(k) or {}
        c["all"] += 1
        if (s.get("v3_p_req") or 0) < a.sev:
            continue
        c["sev"] += 1
        if (s.get("rm2v3_p_natural") or 0) < a.nat:
            continue
        c["sev+nat"] += 1
        p = s.get("rm1_p_halluc")
        if p is None or not (a.lo <= p < a.hi):
            c[f"critic<{a.lo}" if p is not None and p < a.lo else f"critic>={a.hi}"] += 1
            continue
        c["sev+nat+band"] += 1
        ok.append(k)
    rng = random.Random("scope_critic|filter")
    rng.shuffle(ok)
    keep, seen = [], set()
    for k in ok:
        if it[k]["resp_id"] in seen:
            continue
        seen.add(it[k]["resp_id"])
        keep.append(k)
    c["one_per_resp"] = len(keep)
    keep = keep[:a.n]
    write(D / "scope_label" / "items.jsonl", [{**it[k], "scores": sc[k]} for k in keep])
    print("SCOPE_FILTER", json.dumps(dict(c)), "to_claude", len(keep), flush=True)


def cmd_build(a) -> None:
    from ragtruth.critic_harvest import row
    from ragtruth.rm_data import OUT, RM1_OPT
    it = {x["aid"]: x for x in read_jsonl(D / "scope_label" / "items.jsonl")}
    lab = {r["aid"]: r["claude"] for r in read_jsonl(D / "scope_label" / "labels.jsonl")}
    good = sorted(k for k, c in lab.items() if k in it and c.get("severity") in ("clear", "minor") and c.get("matches") and c.get("natural") == "yes")
    pool = sorted(k for k in it if k in lab)   # 两组都只从 Claude 有结论的条目里取，排除「Claude 没返回」带来的差别
    nc = random.Random("scope_critic|nc").sample(pool, len(good))
    stat = {"labeled": len(pool), "claude_good": len(good),
            "claude_real": sum(lab[k].get("severity") in ("clear", "minor") for k in pool),
            "claude_match": sum(bool(lab[k].get("matches")) for k in pool),
            "nc_real": sum(lab[k].get("severity") in ("clear", "minor") for k in nc),
            "nc_good": len(set(nc) & set(good))}
    for arm, keys in [(k, v) for k, v in (("cl", good), ("nc", nc)) if k in a.arms]:
        add = []
        for k in keys:
            x = it[k]
            t = lab[k].get("type") if arm == "cl" and lab[k].get("type") in ("conflict", "baseless") else "conflict"
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            orig = x["sentences_orig"] if isinstance(x["sentences_orig"], list) else json.loads(x["sentences_orig"])
            g = "gh-" + str(x["source_id"])
            add.append(row(x["id"] + "-err", x["source"], ctx, int(x["idx"]), RM1_OPT[t], x["task_type"], g, f"scope_{arm}_error"))
            if not a.no_orig:   # 试试不配原句（配对原句和错句只差一两个词，疑似扰动矛盾判断）
                add.append(row(x["id"] + "-orig", x["source"], orig, int(x["idx"]), RM1_OPT["faithful"], x["task_type"], g, f"scope_{arm}_original"))
        base, d = OUT / a.base, OUT / f"{a.base}_sc{arm}{'_no' if a.no_orig else ''}"
        d.mkdir(parents=True, exist_ok=True)
        train = list(read_jsonl(base / "train.jsonl")) + add
        random.Random(f"scope_critic|{arm}|shuffle").shuffle(train)
        write(d / "train.jsonl", train)
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copy(base / sp, d / sp)
        meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "arm": arm, "added_errors": len(keys), **stat,
                "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SCOPE_BUILD", json.dumps(stat), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "pick_sev", "filter", "build"])
    ap.add_argument("--samples", default="samples_scope_9b.jsonl")
    ap.add_argument("--versions", nargs="*", default=["v3b", "v3c", "v3c0"])
    ap.add_argument("--sev", type=float, default=0.5)
    ap.add_argument("--nat", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--base", default="rm1__core_r2")
    ap.add_argument("--no_orig", action="store_true", help="build：只加错句，不配同位置原句")
    ap.add_argument("--arms", nargs="*", default=["cl", "nc"])
    a = ap.parse_args()
    {"items": cmd_items, "pick_sev": cmd_pick_sev, "filter": cmd_filter, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
