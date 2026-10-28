"""第三方裁定：35B 复核与人工标注不一致的句子，抽样交给 Claude Opus 独立判断，估计双方各自的错判率。

全量复核（ragtruth.adjudicate）发现 35B 判出的幻觉句占 26.8%，人工只有 7.5%，一致率 79.4%（2026-10-26）。
到底是人工漏标多，还是 35B 判得太严，需要第三方裁定：
- 分层抽样：「35B 判幻觉、人工判无」n_judge_only 句，「人工判幻觉、35B 判无」n_human_only 句（不含粗粒度句）；
- Claude 看不到人工标注和 35B 的判定，用和 critic 相同的输入、判定要点和输出格式独立判断；
- 汇总：每一层里 Claude 站在人工一边还是 35B 一边，据此估计人工漏标率和 35B 误判率。

用法：python -m ragtruth.arbitrate --judge outputs/rt/eval/judge_full.jsonl --out outputs/rt/eval/arbitration.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.task import expand_claims, messages, parse_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--judge", default="outputs/rt/eval/judge_full.jsonl", help="35B 全量复核结果")
    ap.add_argument("--n_judge_only", type=int, default=60)
    ap.add_argument("--n_human_only", type=int, default=20)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/rt/eval/arbitration.jsonl")
    args = ap.parse_args()

    gold = {c["id"]: c for s in read_jsonl(args.test) for c in expand_claims(s)}
    judge = {r["id"]: r for r in read_jsonl(args.judge) if r.get("judge") in ("faithful", "hallucination")}
    ids = sorted(i for i in judge if i in gold and not gold[i].get("coarse"))
    judge_only = [i for i in ids if gold[i]["label"] == "faithful" and judge[i]["judge"] == "hallucination"]
    human_only = [i for i in ids if gold[i]["label"] == "hallucination" and judge[i]["judge"] == "faithful"]
    rng = random.Random(args.seed)
    items = ([{"id": i, "stratum": "judge_only"} for i in rng.sample(judge_only, min(args.n_judge_only, len(judge_only)))]
             + [{"id": i, "stratum": "human_only"} for i in rng.sample(human_only, min(args.n_human_only, len(human_only)))])
    print(f"strata sizes: judge_only {len(judge_only)}, human_only {len(human_only)}; sampling {len(items)}")

    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(item: dict) -> dict:
        out = llm.chat_one(messages(gold[item["id"]], tips=True))
        p = parse_output(out)
        return {"stratum": item["stratum"], "claude": p.verdict, "claude_type": p.type, "output": out}

    map_resumable(items, run, args.out, concurrency=args.concurrency)

    rows = [r for r in kept(args.out) if r.get("claude") in ("faithful", "hallucination")]
    summary = {}
    for st in ("judge_only", "human_only"):
        sub = [r for r in rows if r["stratum"] == st]
        sides = collections.Counter(("human" if r["claude"] == gold[r["id"]]["label"] else "35B") for r in sub)
        summary[st] = {"n": len(sub), "claude_agrees_with_human": sides["human"], "claude_agrees_with_35B": sides["35B"],
                       "stratum_size": len(judge_only) if st == "judge_only" else len(human_only)}
    # 外推：judge_only 层里 Claude 判幻觉的比例 × 层大小 ≈ 人工漏标数；human_only 层里 Claude 判无的比例 ≈ 人工误标
    jo, ho = summary["judge_only"], summary["human_only"]
    if jo["n"]:
        summary["est_human_missed"] = round(jo["claude_agrees_with_35B"] / jo["n"] * jo["stratum_size"])
        summary["est_35B_false_alarms"] = round(jo["claude_agrees_with_human"] / jo["n"] * jo["stratum_size"])
    if ho["n"]:
        summary["est_human_false_labels"] = round(ho["claude_agrees_with_35B"] / ho["n"] * ho["stratum_size"])
    Path(args.out).with_name("arbitration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
