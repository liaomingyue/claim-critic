"""合适度模型：不带回答（suit_v1）vs 带一条正确回答（suit_ans）在同一批原文、同一套标签上比较。

带回答的数据（*_ans.jsonl）在 test / policy 上只保留有无错回答的原文，标签用其余回答算；不带回答的模型的预测按 id 取同样的原文、
用同样的标签重评。差值用按原文的 bootstrap（1000 次）给 90% 区间：宏平均 AUC、前 10 名倍数（lift10）。

  python -m ragtruth.suit_cmp --a outputs/rt/suit/suit_v1 --b outputs/rt/suit/suit_ans → results/suit_cmp_answer.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.suit_data import MECHS, OUT, evaluate


def preds(d: Path, split: str) -> dict[str, list[float]]:
    return {r["id"]: r["probs"] for r in read_jsonl(d / f"preds_{split}.jsonl")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="不带回答的模型目录")
    ap.add_argument("--b", required=True, help="带回答的模型目录")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="results/suit_cmp_answer.json")
    a = ap.parse_args()
    res = {}
    for split in ("holdout", "test", "policy"):
        rows = list(read_jsonl(OUT / f"{split}_ans.jsonl"))
        pa, pb = preds(Path(a.a), split), preds(Path(a.b), split)
        rows = [r for r in rows if r["id"] in pa and r["id"] in pb]
        def ev(rs, P):
            return evaluate({m: [P[r["id"]][j] for r in rs] for j, m in enumerate(MECHS)}, rs)
        ea, eb = ev(rows, pa), ev(rows, pb)
        rng = random.Random(0)
        d_auc, d_lift = [], []
        for _ in range(a.n_boot):
            bs = [rows[rng.randrange(len(rows))] for _ in rows]
            bs = [{**r, "id": f"{r['id']}#{i}"} for i, r in enumerate(bs)]
            pa2 = {r["id"]: pa[r["id"].split("#")[0]] for r in bs}
            pb2 = {r["id"]: pb[r["id"].split("#")[0]] for r in bs}
            xa, xb = ev(bs, pa2), ev(bs, pb2)
            if xa["macro_auc"] is not None and xb["macro_auc"] is not None:
                d_auc.append(xb["macro_auc"] - xa["macro_auc"])
            if xa.get("mean_lift10") is not None and xb.get("mean_lift10") is not None:
                d_lift.append(xb["mean_lift10"] - xa["mean_lift10"])
        ci = lambda v: [round(sorted(v)[int(0.05 * len(v))], 3), round(sorted(v)[int(0.95 * len(v)) - 1], 3)] if v else None
        res[split] = {"sources": len(rows),
                      "no_answer": {k: ea.get(k) for k in ("macro_auc", "macro_spearman_soft", "mean_lift10", "mean_lift20", "mean_brier")},
                      "with_answer": {k: eb.get(k) for k in ("macro_auc", "macro_spearman_soft", "mean_lift10", "mean_lift20", "mean_brier")},
                      "delta_auc": round(eb["macro_auc"] - ea["macro_auc"], 3), "delta_auc_ci90": ci(d_auc),
                      "delta_lift10": round((eb.get("mean_lift10") or 0) - (ea.get("mean_lift10") or 0), 2), "delta_lift10_ci90": ci(d_lift),
                      "by_mech_auc": {m: (ea["by_mech"][m]["auc"], eb["by_mech"][m]["auc"]) for m in MECHS}}
        print(split, json.dumps({k: res[split][k] for k in ("sources", "no_answer", "with_answer", "delta_auc", "delta_auc_ci90", "delta_lift10", "delta_lift10_ci90")}), flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
