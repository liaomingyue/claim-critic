"""定向造错生成器的数据加进 2B critic（2026-11-03，用户：数据生成和 critic 重训的脚本先准备）。

harvest（gen_target_data harvest）存下的样本里取合格（P(真错) = RM3 A + B ≥ gate、原句干净）且 2B critic P(判出) 在 [lo, hi) 的——
Claude 判真错：0.1–0.3 11/16、0.3–0.7 14/16（< 0.1 只有 11/33，不要）；这些正是 2B critic 拿不准、读起来又自然的错。
每条错句配一条同一位置的原句（9B 判干净）当忠实样本，免得 critic 学成「像合成的就是错」。标签用生成器给的类型（B 矛盾 / C 无依据）；
RM3 判 B（类型 / 理由不对）的类型可能标错，只影响 B / C 的分配，判错与否（P(B) + P(C)）不受影响。
校准集、测试集（RAGTruth++）沿用基线 critic 的，按回答切分不变；加的数据只进训练集，原文都是回答 GRPO 的训练原文，和验收的 100 条不重叠。

  python -m ragtruth.critic_harvest --harvest outputs/rt/gentarget/harvest_a7.jsonl --name rm1__mcgold_h1
  → outputs/rt/rm/rm1__mcgold_h1/{train,calib,test}.jsonl + meta.json（run_mcrm.sh 直接训）
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.rm_data import OUT, RM1_OPT, RM1_SYSTEM
from ragtruth.task import render_input


def row(rid: str, source: str, sentences: list[str], idx: int, label: str, task: str, group: str, origin: str) -> dict:
    prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": source, "sentences": sentences, "idx": idx})}]
    return {"id": rid, "prompt": prompt, "completion": label, "label": label, "task_type": task, "group": group, "origin": origin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--base", default="rm1__mcgold")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--gate", type=float, default=0.7)
    ap.add_argument("--max_add", type=int, default=1500, help="最多加几条错句（另配同样多的原句）")
    ap.add_argument("--per_prompt", type=int, default=2, help="同一个 prompt 最多取几条，防止一篇原文占太多")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default="", help="只用这些 harvest id（每行一个；清洗后的白名单），不再按区间筛")
    a = ap.parse_args()
    allow = {l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()} if a.ids else None
    keep, seen_sent, per = [], set(), collections.Counter()
    xs = sorted((x for x in read_jsonl(a.harvest) if x.get("p_halluc") is not None and x["p_correct"] >= a.gate and x["orig_p"] <= 0.5
                 and (x["id"] in allow if allow is not None else a.lo <= x["p_halluc"] < a.hi)), key=lambda x: x["id"])
    rng = random.Random(a.seed)
    rng.shuffle(xs)
    for x in xs:
        if x["sentence"] in seen_sent or per[x["prompt_id"]] >= a.per_prompt:
            continue
        seen_sent.add(x["sentence"])
        per[x["prompt_id"]] += 1
        keep.append(x)
        if len(keep) >= a.max_add:
            break
    base, d = OUT / a.base, OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    add = []
    for x in keep:
        g = "gh-" + str(x["source_id"])
        add.append(row(x["id"] + "-err", x["source"], x["context"], int(x["idx"]), RM1_OPT[x["type"]], x["task_type"], g, "harvest_error"))
        add.append(row(x["id"] + "-orig", x["source"], x["sentences_orig"], int(x["idx"]), RM1_OPT["faithful"], x["task_type"], g, "harvest_original"))
    train = list(read_jsonl(base / "train.jsonl")) + add
    rng.shuffle(train)
    with open(d / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "harvest": a.harvest, "band": [a.lo, a.hi], "gate": a.gate,
            "added_errors": len(keep), "ids_file": a.ids or None, "added_originals": len(keep), "base": a.base,
            "added_by_type": dict(collections.Counter(x["type"] for x in keep)), "added_by_task": dict(collections.Counter(x["task_type"] for x in keep)),
            "added_bins": {f"{lo}-{hi}": sum(lo <= x["p_halluc"] < hi for x in keep) for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7))},
            "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CRITIC_HARVEST", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
