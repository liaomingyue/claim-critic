"""两个 critic「各自多报」的句子交给 Claude 复核：人工标为无幻觉、只有 A 标为幻觉 / 只有 B 标为幻觉，两层各抽 n 条。

用来回答「A 多报的部分是人工漏标还是真误报」——两层里 Claude 判为真幻觉的比例一比，就知道 A 比 B 多标出的
句子是不是更多地落在人工漏标上。输入、判定要点、输出格式同 ragtruth.arbitrate_fp。

用法：python -m ragtruth.arbitrate_diff --a outputs/rt/eval/sft-9b-syn-critic --b outputs/rt/eval/sft-9b-syn-random \
          --n 50 --out outputs/rt/analysis/arb_diff_9b_critic_random.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.arbitrate_fp import wilson
from ragtruth.task import expand_claims, messages, parse_output


def flags(path: str, style: str, ids: set[str]) -> dict[str, bool]:
    return {p["id"]: parse_output(p["output"], style).verdict == "hallucination"
            for p in read_jsonl(f"{path}/predictions.jsonl") if p["id"] in ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--style", default="full")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    claims = {c["id"]: c for s in read_jsonl(args.test) for c in expand_claims(s) if not c["coarse"]}
    faithful = {i for i, c in claims.items() if c["label"] == "faithful"}
    pa, pb = flags(args.a, args.style, faithful), flags(args.b, args.style, faithful)
    strata = {"a_only": sorted(i for i in faithful if pa.get(i) and not pb.get(i)),
              "b_only": sorted(i for i in faithful if pb.get(i) and not pa.get(i))}
    rng = random.Random(args.seed)
    items = [{"id": i, "stratum": s} for s, ids in strata.items() for i in rng.sample(ids, min(args.n, len(ids)))]

    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(item: dict) -> dict:
        out = llm.chat_one(messages(claims[item["id"]], tips=True))
        p = parse_output(out)
        return {"stratum": item["stratum"], "claude": p.verdict, "claude_type": p.type, "output": out,
                "task_type": claims[item["id"]]["task_type"]}

    map_resumable(items, run, args.out, concurrency=args.concurrency)

    rows = [r for r in kept(args.out) if r.get("claude") in ("faithful", "hallucination")]
    summary = {"a": args.a, "b": args.b}
    for s, ids in strata.items():
        sub = [r for r in rows if r["stratum"] == s]
        k = sum(r["claude"] == "hallucination" for r in sub)
        lo, hi = wilson(k, len(sub))
        summary[s] = {"pool": len(ids), "sampled": len(sub), "claude_says_hallucination": k,
                      "real_rate": round(k / len(sub), 3) if sub else None, "ci95": [round(lo, 3), round(hi, 3)],
                      "est_real": round(k / len(sub) * len(ids)) if sub else None,
                      "by_task": {t: [sum(r["claude"] == "hallucination" for r in sub if r["task_type"] == t),
                                      sum(r["task_type"] == t for r in sub)] for t in ("QA", "Summary", "Data2txt")}}
    Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
