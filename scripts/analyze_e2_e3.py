"""e2（只判定）vs e3（证据 + 理由 + 判定）逐句对比：同一训练集、同一测试句，定位 e3 多出来的错误从哪来。"""
import collections
import json

from common.schema import read_jsonl
from ragtruth.task import evidence_in_source, expand_claims, parse_output, says_negative

gold = {c["id"]: c for s in read_jsonl("outputs/external/ragtruth/test.jsonl") for c in expand_claims(s) if not c["coarse"]}
e2 = {p["id"]: parse_output(p["output"], "verdict") for p in read_jsonl("outputs/rt/eval/sft-0.6b-e2/predictions.jsonl")}
e3 = {p["id"]: (p["output"], parse_output(p["output"], "full")) for p in read_jsonl("outputs/rt/eval/sft-0.6b-e3/predictions.jsonl")}
ids = [i for i in gold if i in e2 and i in e3]

cell = collections.Counter()
rows = []
for i in ids:
    g = gold[i]["label"] == "hallucination"
    p2 = e2[i].verdict == "hallucination"
    out3, q = e3[i]
    p3 = q.verdict == "hallucination"
    cell[(g, p2 == g, p3 == g)] += 1
    neg = says_negative(q.reason or "")
    loc = q.evidence is not None and evidence_in_source(q.evidence, gold[i]["source"])
    rows.append(dict(id=i, gold=g, e2_ok=p2 == g, e3_ok=p3 == g, e3_pred=p3, reason_neg=neg, ev_none=q.evidence_none,
                     ev_loc=loc, fmt=q.format_ok, task=gold[i]["task_type"], out=out3))

print("n", len(ids))
for g in (True, False):
    print("gold", "HAL" if g else "OK ", {f"e2{'✓' if a else '✗'} e3{'✓' if b else '✗'}": cell[(g, a, b)] for a in (True, False) for b in (True, False)})

def describe(sel, name):
    n = len(sel)
    if not n:
        return
    c = collections.Counter()
    for r in sel:
        c["reason_says_negative"] += r["reason_neg"]
        c["reason_agrees_with_own_verdict"] += r["reason_neg"] == r["e3_pred"]
        c["evidence_none"] += r["ev_none"]
        c["evidence_located"] += r["ev_loc"]
        c["format_bad"] += not r["fmt"]
    print(f"\n{name}: n={n}", {k: f"{v / n:.0%}" for k, v in c.items()}, dict(collections.Counter(r["task"] for r in sel)))

describe([r for r in rows if not r["gold"] and r["e2_ok"] and not r["e3_ok"]], "FP only e3 (gold OK, e2 right, e3 says HAL)")
describe([r for r in rows if r["gold"] and r["e2_ok"] and not r["e3_ok"]], "FN only e3 (gold HAL, e2 right, e3 says OK)")
describe([r for r in rows if not r["gold"] and r["e3_ok"]], "e3 TN (reference)")
describe([r for r in rows if r["gold"] and r["e3_ok"]], "e3 TP (reference)")
describe([r for r in rows if not r["gold"] and not r["e2_ok"] and not r["e3_ok"]], "FP both")

json.dump([r for r in rows if r["e2_ok"] != r["e3_ok"]], open("outputs/rt/analysis/e2_vs_e3_diff.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
