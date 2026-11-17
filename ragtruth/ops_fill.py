"""按操作合成的错误补 critic 盲区。

样本：错误生成器在训练集原文的回答上（genpool_train，不碰校准集和验收集的原文）每个回答 × 每种操作采 1 条，
把关模型打分后取盲区 = 真错、自然、操作三项都达标（≥ 0.5）且当前 critic 低于 0.5。同一回答的同一句改成同样的错句只留一条。
  select：盲区全量 → zone.jsonl；每种操作随机取若干条 → items.jsonl（交 Claude 核实：真错、自然）
  build ：在基线 critic 的训练集上只加错句、不配原句，三组：
            cl  = items 里 Claude 判真错且自然的
            nc  = items 里随机取同样条数（不看 Claude）
            all = 盲区全量（不看 Claude；--arms all 时不需要 labels）

  python -m ragtruth.ops_fill select --tag tr_st8 --round of1
  python -m ragtruth.ops_fill build --round of1 --base rm1__core_r2_bfcl --arms all
  python -m ragtruth.ops_fill build --round of1 --base rm1__core_r2_bfcl --arms cl nc
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from common.schema import read_jsonl

C = Path("outputs/rt/core")
O = Path("outputs/rt/ops_fill")
BASELESS = {"add_detail", "add_hedge", "add_link", "add_conclusion"}   # 多出原文没有的内容 → 无依据；其余算和原文矛盾


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def in_zone(p: dict) -> bool:
    return all(p.get(k) is not None for k in ("p_err", "p_nat", "p_op", "p_edit")) and p["p_err"] >= 0.5 and p["p_nat"] >= 0.5 and p["p_op"] >= 0.5 and p["p_edit"] < 0.5


def cmd_select(a) -> None:
    B = O / a.round
    P = {r["id"]: r for r in read_jsonl(C / f"samples_ops_{a.tag}_rm4.jsonl")}
    rows = sorted((r for r in read_jsonl(C / f"samples_ops_{a.tag}.jsonl") if r["why"] == "ok" and r["id"] in P), key=lambda r: r["id"])
    ok3 = [r for r in rows if P[r["id"]].get("p_edit") is not None and all((P[r["id"]].get(k) or 0) >= 0.5 for k in ("p_err", "p_nat", "p_op"))]
    random.Random(f"ops_fill|{a.round}|zone").shuffle(rows)
    seen, zone = set(), []
    for r in rows:
        k = (r["resp_id"], int(r["idx"]), " ".join(r["sentence"].lower().split()))
        if in_zone(P[r["id"]]) and k not in seen:
            seen.add(k)
            zone.append({**r, "idx": int(r["idx"]), **{k: v for k, v in P[r["id"]].items() if k != "id"}})
    n, items = collections.Counter(), []
    for r in zone:
        if n[r["op"]] < a.per_op:
            n[r["op"]] += 1
            items.append(r)
    write(B / "zone.jsonl", zone)
    write(B / "items.jsonl", items)
    print("OPS_FILL_SELECT", json.dumps({"samples": len(rows), "pass3": len(ok3), "zone": len(zone), "items": len(items), "sources": len({r["source_id"] for r in zone}),
                                         "zone_by_op": dict(sorted(collections.Counter(r["op"] for r in zone).items())), "items_by_op": dict(sorted(n.items()))}), flush=True)


def cmd_build(a) -> None:
    from ragtruth.critic_harvest import row
    from ragtruth.error_ops import OPS
    from ragtruth.rm_data import OUT, RM1_OPT
    B = O / a.round
    zone = {x["id"]: x for x in read_jsonl(B / "zone.jsonl")}
    items = sorted(x["id"] for x in read_jsonl(B / "items.jsonl"))
    base = OUT / a.base
    held = {r["group"] for sp in ("calib.jsonl", "test.jsonl") for r in read_jsonl(base / sp) if "group" in r}
    leak = sorted(k for k, x in zone.items() if str(x["source_id"]) in held)
    assert not leak, f"{len(leak)} zone samples come from calib / test sources"
    lab, stat = {}, {"zone": len(zone), "items": len(items)}
    if set(a.arms) & {"cl", "nc"}:
        lab = {r["id"]: r["c"] for r in read_jsonl(B / "labels.jsonl") if r["id"] in zone}
        good = sorted(k for k in items if k in lab and lab[k]["severity"] in ("clear", "minor") and lab[k]["natural"] == "yes")
        pool = sorted(k for k in items if k in lab)   # 只从 Claude 有结论的里取，排除「没返回」带来的差别
        nc = random.Random(f"ops_fill|{a.round}|nc").sample(pool, len(good))
        stat.update({"labeled": len(pool), "real": sum(lab[k]["severity"] in ("clear", "minor") for k in pool), "clear": sum(lab[k]["severity"] == "clear" for k in pool),
                     "natural": sum(lab[k]["natural"] == "yes" for k in pool), "cl": len(good), "nc_good": len(set(nc) & set(good)),
                     "by_op": {op: "%d/%d" % (sum(zone[k]["op"] == op for k in good), sum(zone[k]["op"] == op for k in pool)) for op in OPS}})
    keys = {"cl": lambda: good, "nc": lambda: nc, "all": lambda: sorted(zone)}
    for arm in a.arms:
        add = []
        for k in keys[arm]():
            x = zone[k]
            op = lab[k]["op"] if arm == "cl" and lab[k].get("op") in OPS else x["op"]
            add.append(row(k.replace("#", "_") + "-err", x["source"], x["context"], int(x["idx"]), RM1_OPT["baseless" if op in BASELESS else "conflict"],
                           x["task_type"], "of-" + str(x["source_id"]), f"ops_fill_{arm}_error"))
        d = OUT / f"{a.name}{arm}"
        d.mkdir(parents=True, exist_ok=True)
        train = list(read_jsonl(base / "train.jsonl")) + add
        random.Random(f"ops_fill|{a.round}|{arm}|shuffle").shuffle(train)
        write(d / "train.jsonl", train)
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copy(base / sp, d / sp)
        meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "arm": arm, "added_errors": len(add), **stat,
                "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        stat[f"added_{arm}"] = len(add)
    print("OPS_FILL_BUILD", json.dumps(stat), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "build"])
    ap.add_argument("--tag", default="tr_st8", help="select：outputs/rt/core/samples_ops_<tag>.jsonl 和 _rm4")
    ap.add_argument("--round", default="of1", help="outputs/rt/ops_fill/ 下的目录")
    ap.add_argument("--per_op", type=int, default=30, help="select：每种操作交 Claude 的条数")
    ap.add_argument("--base", default="rm1__core_r2_bfcl")
    ap.add_argument("--name", default="rm1__core_of1", help="build：新 critic 的名字前缀，后面接组名")
    ap.add_argument("--arms", nargs="+", default=["cl", "nc"], choices=["cl", "nc", "all"])
    a = ap.parse_args()
    {"select": cmd_select, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
