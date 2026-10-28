"""估计一个 critic 的真实精确率：从它的误报（人工判无、critic 判有）里抽样交给 Claude 独立裁定。

人工标注漏标多（诊断 A：0.6B 的误报里 40–60% Claude 判为真幻觉），按人工标签算的精确率偏低。
真实精确率 ≈ (TP + 误报数 × Claude 判为幻觉的比例) / critic 判为幻觉的总数；附 Wilson 95% 区间（只计抽样误差）。
人工标为幻觉的句子 Claude 基本同意（30 句里 29 句），召回按人工标签算即可。

用法：python -m ragtruth.arbitrate_fp --pred outputs/rt/eval/sft-9b-v2-notips --style full --n 60 \
          --out outputs/rt/analysis/arb_fp_9b_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.task import expand_claims, messages, parse_output


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--pred", required=True, help="评测目录（含 predictions.jsonl）")
    ap.add_argument("--style", default="full")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    claims = {c["id"]: c for s in read_jsonl(args.test) for c in expand_claims(s) if not c["coarse"]}
    pred = {p["id"]: parse_output(p["output"], args.style).verdict == "hallucination"
            for p in read_jsonl(f"{args.pred}/predictions.jsonl") if p["id"] in claims}
    gold = {i: claims[i]["label"] == "hallucination" for i in pred}
    fp = sorted(i for i in pred if pred[i] and not gold[i])
    tp = sum(pred[i] and gold[i] for i in pred)
    items = [{"id": i} for i in random.Random(args.seed).sample(fp, min(args.n, len(fp)))]

    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(item: dict) -> dict:
        out = llm.chat_one(messages(claims[item["id"]], tips=True))
        p = parse_output(out)
        return {"claude": p.verdict, "claude_type": p.type, "output": out, "task_type": claims[item["id"]]["task_type"]}

    map_resumable(items, run, args.out, concurrency=args.concurrency)

    rows = [r for r in kept(args.out) if r.get("claude") in ("faithful", "hallucination")]
    k, n = sum(r["claude"] == "hallucination" for r in rows), len(rows)
    n_pred, n_pos = tp + len(fp), sum(gold.values())
    lo, hi = wilson(k, n)
    est = lambda rate: (tp + len(fp) * rate) / n_pred  # noqa: E731
    summary = {"pred": args.pred, "n_flagged": n_pred, "tp_human": tp, "fp_human": len(fp), "sampled": n,
               "claude_says_hallucination": k, "fp_real_rate": round(k / n, 3) if n else None,
               "fp_real_rate_ci95": [round(lo, 3), round(hi, 3)],
               "precision_human": round(tp / n_pred, 3), "precision_est": round(est(k / n), 3) if n else None,
               "precision_est_ci95": [round(est(lo), 3), round(est(hi), 3)], "recall_human": round(tp / n_pos, 3),
               "by_task": {t: [sum(r["claude"] == "hallucination" for r in rows if r["task_type"] == t),
                               sum(r["task_type"] == t for r in rows)] for t in ("QA", "Summary", "Data2txt")}}
    Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
