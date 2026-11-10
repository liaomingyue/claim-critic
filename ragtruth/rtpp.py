"""RAGTruth++（blue-guardrails/ragtruth-plus-plus）：RAGTruth 测试集 408 条回答的人工重标，幻觉片段 86 → 865。
用作循环外的金标准：原 RAGTruth 漏标多，精确率被系统性低估、召回也分不出高下。

- convert：408 条里 402 条能按原始 id 对上我们的测试集记录（同一回答文本、同一套切句），只把标签换成 RAGTruth++ 的片段，
  句子标注规则同 ragtruth.convert（片段与句子重叠 ≥ MIN_OVERLAP 即算；覆盖 ≥3 句的大片段记 coarse）。
  「Incorrect Refusal」（有依据却拒答）按 RAGTruth 的规则归入 baseless，单独计数。
- rescore：已有 critic 在全量测试集上的逐句预测直接拿来，在这 402 条上分别用原标签、RAGTruth++ 标签算指标，不用再跑 GPU。

用法：
  python -m ragtruth.rtpp convert --out outputs/external/ragtruth/test_pp.jsonl
  python -m ragtruth.rtpp rescore --out outputs/rt/analysis/rtpp_rescore.json
  python -m ragtruth.rtpp mcnemar --a sft-9b-critic-closed-gfree --b sft-9b-critic-closed-gfreedpo2

- mcnemar：两个 critic 在同一批句子上逐句配对比较（精确二项检验，双侧）。召回：只看金标准为幻觉的句子，
  比「A 抓到 B 漏」与「B 抓到 A 漏」；准确率：全部句子比对错。句级口径同 rescore（不计 coarse）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

from common.schema import read_jsonl, write_jsonl
from ragtruth.convert import coarse_sentences, label_sentences
from ragtruth.metrics import aggregate, bootstrap_f1, per_sample
from ragtruth.task import expand_claims

PP = Path("external_data/ragtruth_pp")


def cmd_convert(args) -> None:
    msgs = list(csv.DictReader(open(PP / "messages.csv", encoding="utf-8")))
    spans = collections.defaultdict(list)
    for s in csv.DictReader(open(PP / "hallucination_spans.csv", encoding="utf-8")):
        spans[s["message_stable_id"]].append({"start": int(s["start_pos"]), "end": int(s["end_pos"]),
                                              "label_type": s["label"], "meta": s.get("reasoning") or ""})
    test = {r["id"]: r for r in read_jsonl(args.test)}
    out, stats = [], collections.Counter()
    for m in msgs:
        if m["role"] != "assistant":
            continue
        oid = json.loads(m["meta"]).get("original_id")
        r = test.get(f"rt-{oid}")
        if r is None:
            stats["no_match"] += 1
            continue
        if r["response"] != m["text"]:
            stats["text_differs"] += 1
            continue
        labels = spans.get(m["stable_id"], [])
        sent_spans = [tuple(x) for x in r["sent_spans"]]
        bad = label_sentences(sent_spans, labels)
        stats["spans"] += len(labels)
        stats["refusal_spans"] += sum(x["label_type"] == "Incorrect Refusal" for x in labels)
        out.append({**r, "id": r["id"], "bad": {str(k): v for k, v in bad.items()},
                    "coarse": coarse_sentences(sent_spans, labels, bad),
                    "label": "hallucination" if bad else "faithful",
                    "spans": [[x["start"], x["end"], x["label_type"]] for x in labels],
                    "span_notes": [[x["start"], x["end"], " ".join(x["meta"].split())] for x in labels if x["meta"].strip()],
                    "origin": "ragtruth_pp", "orig_bad": r["bad"], "orig_coarse": r["coarse"]})
    write_jsonl(args.out, out)
    stats = {**stats, "responses": len(out),
             "bad_sentences_pp": sum(len(r["bad"]) - len(r["coarse"]) for r in out),
             "bad_sentences_orig": sum(len(r["orig_bad"]) - len(r["orig_coarse"]) for r in out),
             "by_task": dict(collections.Counter(r["task_type"] for r in out))}
    print(json.dumps(stats, ensure_ascii=False))


def _style(d: Path) -> str | None:
    """评测时用的输出格式（eval_manifest 里记着）；回答级的 critic（resp_style）不适用逐句重算。"""
    m = d / "eval_manifest.json"
    cfg = json.loads(m.read_text(encoding="utf-8")).get("config", {}) if m.exists() else {}
    if cfg.get("resp_style"):
        return None
    return cfg.get("style") or ("verdict" if "verdict" in d.name else "vfirst" if d.name.endswith("e4") else "full")


def _valid(gold_pp: dict) -> dict:
    """有效口径（ragtruth.pp_valid）：++ 新增幻觉里 Claude 判为 mild / nitpick / meta_wordcount 的句子不参与计分。"""
    from ragtruth.pp_valid import load
    lab = load()
    return {i: c for i, c in gold_pp.items() if lab.get(i, "real") == "real"} if lab else {}


def cmd_rescore(args) -> None:
    pp = list(read_jsonl(args.pp))
    orig = [{**r, "bad": r["orig_bad"], "coarse": r["orig_coarse"]} for r in pp]
    gold_pp = {c["id"]: c for s in pp for c in expand_claims(s)}
    gold_orig = {c["id"]: c for s in orig for c in expand_claims(s)}
    gold_valid = _valid(gold_pp)
    tags = (("orig", gold_orig), ("pp", gold_pp)) + ((("valid", gold_valid),) if gold_valid else ())
    res = {}
    for d in sorted(Path(args.eval_dir).glob("sft-*")):
        pred = d / "predictions.jsonl"
        style = _style(d)
        if not pred.exists() or style is None:
            continue
        preds = {p["id"]: p["output"] for p in read_jsonl(pred)}
        row = {}
        for tag, gold in tags:
            rows = [per_sample(c, preds[i], style) for i, c in gold.items() if i in preds]
            if len(rows) < 0.95 * len(gold):   # 预测不全（只评了一部分）的跳过
                break
            agg = aggregate(rows)["sentence"]
            row[tag] = {k: round(agg[k], 3) for k in ("precision", "recall", "f1")}
            row[tag]["f1_ci95"] = [round(x, 3) for x in bootstrap_f1(rows)["sentence"]]
        if len(row) == len(tags):
            res[d.name] = row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    names = {"orig": "orig", "pp": "RAGTruth++", "valid": "++ valid"}
    print(f"{'model':32s} " + " ".join(f"{names[t] + ' P/R/F1':22s}" for t, _ in tags))
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["pp"]["f1"]):
        print(f"{k:32s} " + " ".join(f"{v[t]['precision']:.3f}/{v[t]['recall']:.3f}/{v[t]['f1']:.3f}    " for t, _ in tags))


def binom_two_sided(b: int, c: int) -> float:
    """McNemar 精确检验：不一致对 b、c 在 H0 下 ~ Binomial(b + c, 0.5)。"""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_mcnemar(args) -> None:
    pp = list(read_jsonl(args.pp))
    orig = [{**r, "bad": r["orig_bad"], "coarse": r["orig_coarse"]} for r in pp]
    res = {}
    gold_pp = {c["id"]: c for s in pp for c in expand_claims(s) if not c.get("coarse")}
    golds = {"orig": {c["id"]: c for s in orig for c in expand_claims(s) if not c.get("coarse")}, "pp": gold_pp}
    if v := _valid(gold_pp):
        golds["valid"] = v
    for tag, gold in golds.items():
        rows = {}
        for m in (args.a, args.b):
            d = Path(args.eval_dir) / m
            preds = {p["id"]: p["output"] for p in read_jsonl(d / "predictions.jsonl")}
            rows[m] = {i: per_sample(c, preds[i], _style(d)) for i, c in gold.items() if i in preds}
        ids = sorted(set(rows[args.a]) & set(rows[args.b]))
        A, B = rows[args.a], rows[args.b]
        pos = [i for i in ids if A[i]["gold"] == "hallucination"]
        hit = lambda R, i: R[i]["pred"] == "hallucination"
        b = sum(hit(A, i) and not hit(B, i) for i in pos)
        c = sum(hit(B, i) and not hit(A, i) for i in pos)
        ok = lambda R, i: R[i]["pred"] == R[i]["gold"]
        b2 = sum(ok(A, i) and not ok(B, i) for i in ids)
        c2 = sum(ok(B, i) and not ok(A, i) for i in ids)
        res[tag] = {"sentences": len(ids), "positives": len(pos),
                    "recall": {"a": round(sum(hit(A, i) for i in pos) / max(1, len(pos)), 3),
                               "b": round(sum(hit(B, i) for i in pos) / max(1, len(pos)), 3),
                               "a_only": b, "b_only": c, "p": round(binom_two_sided(b, c), 4)},
                    "accuracy": {"a_only": b2, "b_only": c2, "p": round(binom_two_sided(b2, c2), 4)}}
    out = {"a": args.a, "b": args.b, **res}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    c.add_argument("--out", default="outputs/external/ragtruth/test_pp.jsonl")
    r = sub.add_parser("rescore")
    r.add_argument("--pp", default="outputs/external/ragtruth/test_pp.jsonl")
    r.add_argument("--eval_dir", default="outputs/rt/eval")
    r.add_argument("--out", default="outputs/rt/analysis/rtpp_rescore.json")
    m = sub.add_parser("mcnemar")
    m.add_argument("--a", required=True, help="outputs/rt/eval 下的目录名")
    m.add_argument("--b", required=True)
    m.add_argument("--pp", default="outputs/external/ragtruth/test_pp.jsonl")
    m.add_argument("--eval_dir", default="outputs/rt/eval")
    m.add_argument("--out", default="")
    args = ap.parse_args()
    {"convert": cmd_convert, "rescore": cmd_rescore, "mcnemar": cmd_mcnemar}[args.cmd](args)


if __name__ == "__main__":
    main()
