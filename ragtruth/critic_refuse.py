"""critic 学会判拒答对不对（2026-11-04，用户：可以）。
验收回答（2B GRPO）里 2B critic P < 0.1 的明显错 30 句里 8 句是「Unable to answer based on given passages.」而原文其实能回答；
2B 对 22 句拒答一句都没判出来。原因在训练标签：rm1__mcgold 训练集里 30 句拒答 25 句标 A（b9 16 / human 8 / gold 1），
其中 b9 的 16 句按拒答 RM 的 Claude 标注（outputs/rt/refuse，能不能回答 yes / partial / no）有 10 句原文能回答。

  ① 训练集里检查句是拒答、原文 Claude 判 yes / partial 的 A 行改成 B（与原文矛盾）；
  ② 拒答 RM 训练集的原文（582 原版 + 近似改写 nearmiss 里的训练部分）各加一条拒答回答：yes / partial → B，no → A
     （只用 refuse 的 train 切分，和 critic 校准 / 测试集的原文去重）；
  ③ refuse 的 calib + test 切分做成拒答评测 test_refuse.jsonl（is_error = 原文能回答）。

  python -m ragtruth.critic_refuse
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil

from common.schema import read_jsonl
from ragtruth.critic_harvest import row
from ragtruth.rm_data import OUT, too_long

REFUSAL = "Unable to answer based on given passages."
REF_RE = re.compile(r"unable to answer|cannot be answered|can.t answer|cannot answer", re.I)


def sid_of(x: dict) -> str:
    m = re.search(r"-(\d+)", x.get("group") or x["id"])
    return m.group(1) if m else ""


def checked(x: dict) -> str:
    return x["prompt"][1]["content"].split("[sentence to check]")[-1].split(":", 1)[-1].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="rm1__mcgold")
    ap.add_argument("--name", default="rm1__mcgold_ref")
    ap.add_argument("--refuse", default="refuse", help="OUT 下的拒答 RM 数据目录")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base, d, rf = OUT / a.base, OUT / a.name, OUT / a.refuse
    d.mkdir(parents=True, exist_ok=True)
    held = {sid_of(x) for sp in ("calib.jsonl", "test.jsonl") for x in read_jsonl(base / sp)}
    split = {sp: list(read_jsonl(rf / f"{sp}.jsonl")) for sp in ("train", "calib", "test")}
    ans = {str(r["group"]): r["answerable"] for rs in split.values() for r in rs if not str(r.get("origin", "")).startswith("nearmiss")}
    train = list(read_jsonl(base / "train.jsonl"))
    relab = collections.Counter()
    for x in train:
        if x["task_type"] == "QA" and REF_RE.search(checked(x)):
            v = ans.get(sid_of(x))
            relab[(x["label"], x.get("origin"), v)] += 1
            if x["label"] == "A" and v in ("yes", "partial"):
                x["label"] = x["completion"] = "B"
                x["relabel"] = "refusal_answerable"

    def crow(r: dict, sp: str) -> dict:
        src = r["prompt"][1]["content"]
        lab = "A" if r["answerable"] == "no" else "B"
        o = row(f"rf-{sp}-{r['id']}", src, [REFUSAL], 1, lab, "QA", "rf-" + str(r["group"]), "refuse_" + r["answerable"])
        o["is_error"] = lab == "B"
        o["answerable"] = r["answerable"]
        return o

    ml = json.load(open(base / "meta.json", encoding="utf-8"))["max_tokens"]
    add = [crow(r, "train") for r in split["train"] if str(r["group"]) not in held]
    test = [crow(r, sp) for sp in ("calib", "test") for r in split[sp] if str(r["group"]) not in held]
    n0 = len(add) + len(test)
    add = [r for r in add if not too_long(r["prompt"], ml)]
    test = [r for r in test if not too_long(r["prompt"], ml)]
    train += add
    random.Random(a.seed).shuffle(train)
    with open(d / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(d / "test_refuse.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base,
            "refusal_rows_in_base": {"|".join(map(str, k)): v for k, v in sorted(relab.items(), key=str)},
            "relabeled": sum(1 for x in train if x.get("relabel")),
            "added": dict(collections.Counter(r["answerable"] for r in add)),
            "test_refuse": dict(collections.Counter(r["answerable"] for r in test)),
            "too_long_dropped": n0 - len(add) - len(test), "dropped_overlap": sum(str(r["group"]) in held for rs in split.values() for r in rs),
            "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CRITIC_REFUSE", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
