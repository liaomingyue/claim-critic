"""标注一致性与裁决。

- κ_verdict：两位标注者在 verdict 上的 Cohen's κ（主一致性指标，低于 0.6 要重写规范再标）。
- κ_type：两人都标 hallucination 的样本上，type 的 Cohen's κ。
- 不一致样本导出给第三人裁决；裁决结果合并成最终测试集。

用法：
  python -m annotation.kappa agree  --a ann_a.jsonl --b ann_b.jsonl --disagree_out to_adjudicate.jsonl
  python -m annotation.kappa merge  --a ann_a.jsonl --b ann_b.jsonl --adjudicated adj.jsonl --out test.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common.schema import Sample, read_jsonl, write_jsonl


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "need two equal-length non-empty label lists"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(ann_a: dict[str, dict], ann_b: dict[str, dict]) -> dict:
    ids = sorted(set(ann_a) & set(ann_b))
    va = [ann_a[i]["label"] for i in ids]
    vb = [ann_b[i]["label"] for i in ids]
    both_h = [i for i in ids if ann_a[i]["label"] == ann_b[i]["label"] == "hallucination"]
    res = {"n": len(ids), "kappa_verdict": cohen_kappa(va, vb),
           "raw_agreement": sum(x == y for x, y in zip(va, vb)) / len(ids)}
    if both_h:
        res["kappa_type"] = cohen_kappa([ann_a[i]["type"] for i in both_h], [ann_b[i]["type"] for i in both_h])
        res["n_both_hallucination"] = len(both_h)
    return res


def disagreements(ann_a: dict[str, dict], ann_b: dict[str, dict]) -> list[dict]:
    out = []
    for i in sorted(set(ann_a) & set(ann_b)):
        a, b = ann_a[i], ann_b[i]
        if a["label"] != b["label"] or a["type"] != b["type"]:
            out.append({"id": i, "refs": a["refs"], "claim": a["claim"],
                        "annotator_a": {k: a[k] for k in ("label", "type", "evidence", "reason")},
                        "annotator_b": {k: b[k] for k in ("label", "type", "evidence", "reason")}})
    return out


def merge(ann_a: dict[str, dict], ann_b: dict[str, dict], adjudicated: dict[str, dict]) -> list[dict]:
    """一致的取 A 的标注（evidence 取 A）；不一致的取裁决结果；缺裁决的报错。"""
    final = []
    for i in sorted(set(ann_a) & set(ann_b)):
        a, b = ann_a[i], ann_b[i]
        if a["label"] == b["label"] and a["type"] == b["type"]:
            s = dict(a)
        elif i in adjudicated:
            s = dict(a, **{k: adjudicated[i][k] for k in ("label", "type", "evidence", "reason")
                           if k in adjudicated[i]})
        else:
            raise KeyError(f"{i} disagrees but has no adjudication")
        s.update(split="test", origin="human")
        final.append(Sample.from_dict(s).to_dict())
    return final


def _load(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["agree", "merge"])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--disagree_out")
    ap.add_argument("--adjudicated")
    ap.add_argument("--out")
    args = ap.parse_args()
    a, b = _load(args.a), _load(args.b)

    if args.cmd == "agree":
        res = agreement(a, b)
        print(json.dumps(res, indent=2))
        if res["kappa_verdict"] < 0.6:
            print("kappa < 0.6: revise annotation/guidelines.md and re-annotate before continuing")
        if args.disagree_out:
            print(f"wrote {write_jsonl(args.disagree_out, disagreements(a, b))} disagreements")
    else:
        final = merge(a, b, _load(args.adjudicated))
        n_h = sum(s["label"] == "hallucination" for s in final)
        print(f"wrote {write_jsonl(args.out, final)} samples, hallucination ratio {n_h / len(final):.1%}")


if __name__ == "__main__":
    main()
