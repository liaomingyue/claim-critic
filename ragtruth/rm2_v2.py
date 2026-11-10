"""RM2 第二版：自然度判别（RM2 的样本 = 经过 critic 和 sev 过滤的生成器错误 + 自然的错误样本，训完在新样本集上重跑）。

  A 自然：种子集训练切分的真实错误句（RAGTruth 模型自己写错的句子，Claude 核实），回答里其它句子是改对后的
  B 合成：9B 错误改写模型（旧格式，gen-core-9b）在训练池提示上的合格采样里，sev__gen_dne P(错) ≥ --dne 且 rm1__core P(判出) ≥ --rm1 的
         （= 过了 sev 和 critic 过滤的「像样的错」，GRPO 里正是要在这些里分自然与否）；按原文抽到和 A 同样多
  提示词同旧 RM2（RM2G_SYSTEM + discrim.render），按原文 80 / 10 / 10 切训练 / 校准 / 测试。
  主评测在 run 之外：生成器 v2 在留出原文上的 1154 条，对照 Claude 的「自然」标签（outputs/model_vs_claude.py）。

  python -m ragtruth.rm2_v2 --dne 0.8 --rm1 0.1
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dne", type=float, default=0.8)
    ap.add_argument("--rm1", type=float, default=0.1)
    ap.add_argument("--name", default="rm2__v2")
    a = ap.parse_args()
    from ragtruth import discrim
    from ragtruth.core import hsplit, write_rm
    from ragtruth.rm_data import RM2G_SYSTEM

    def row(rid, source, ctx, idx, lab, task, src, origin):
        return {"id": rid, "prompt": [{"role": "system", "content": RM2G_SYSTEM}, {"role": "user", "content": discrim.render(source, ctx, idx)}],
                "completion": lab, "label": lab, "task_type": task, "group": "gh-" + str(src), "origin": origin}

    nat = []
    for p in read_jsonl(D / "pairs.jsonl"):
        if p["split"] != "train":
            continue
        sents = p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"])
        k = int(p["idx"])
        ctx = sents[:k - 1] + [p["real"]] + sents[k:]
        nat.append(row("nat/" + p["id"], p["source"], ctx, k, "A", p["task_type"], p["source_id"], "natural_real_error"))
    dne = {r["aid"]: r["p_sev"] for r in read_jsonl(D / "gen_audit/sev_dne_scores_all.jsonl")}
    rm1 = {r["aid"]: r["p_halluc"] for r in read_jsonl(D / "gen_audit/critic_core_scores.jsonl")}
    syn, c = [], collections.Counter()
    for f in ("samples_9b_audit.jsonl", "samples_9b_rest.jsonl"):
        for x in read_jsonl(D / f):
            if x.get("why") != "ok":
                continue
            aid = "9b/" + x["id"]
            pd, pc = dne.get(aid), rm1.get(aid)
            if pd is None or pc is None:
                c["no_score"] += 1
                continue
            if pd["A"] + pd["B"] < a.dne or pc < a.rm1:
                c["filtered_out"] += 1
                continue
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            syn.append(row("syn/" + aid, x["source"], ctx, int(x["idx"]), "B", x["task_type"], x["source_id"], "synthetic_filtered"))
    rng = random.Random("rm2_v2")
    syn.sort(key=lambda r: r["id"])
    rng.shuffle(syn)
    per = collections.Counter()   # 每篇原文最多 4 条，免得少数原文占满
    keep = []
    for r in syn:
        if per[r["group"]] >= 4:
            continue
        per[r["group"]] += 1
        keep.append(r)
        if len(keep) >= len(nat):
            break
    parts = collections.defaultdict(list)
    for r in nat + keep:
        parts[("train", "calib", "test")[hsplit(r["group"], "rm2_v2", (80, 90))]].append(r)
    meta = write_rm(a.name, parts, 3072, {"natural": len(nat), "synthetic_pool": len(syn), "synthetic_kept": len(keep), "dne_thr": a.dne, "rm1_thr": a.rm1, **c})
    print("RM2_V2", json.dumps({k: meta[k] for k in ("train", "calib", "test", "natural", "synthetic_pool", "synthetic_kept")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
