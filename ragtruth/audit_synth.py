"""合成错误句的有效率抽查：Claude 独立判断改后的句子是否真的与原文不符、类型是否和合成时指定的一致。

分两层抽样：难样本（critic 识破率 < threshold）和被识破的样本，对比两层的有效率——
若难样本有效率明显更低，说明「挑 critic 没识破的」同时挑进了无效样本（改了但其实不算错），
过 critic 后加入的那组训练集带着标签噪声，这正是人工修正（实验 4）要补的环节。
Claude 看不到合成时指定的类型和 critic 的输出，输入、判定要点、输出格式同 ragtruth.arbitrate。

用法：python -m ragtruth.audit_synth --synth outputs/rt/synth/r1.jsonl --mined outputs/rt/synth/r1_mined.jsonl \
          --out outputs/rt/analysis/audit_r1.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.arbitrate_fp import wilson
from ragtruth.synth import to_claims
from ragtruth.task import messages, parse_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--mined", required=True)
    ap.add_argument("--n_hard", type=int, default=40)
    ap.add_argument("--n_easy", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    synth = {r["id"]: r for r in read_jsonl(args.synth)}
    rate = {r["id"]: r["detect_rate"] for r in kept(args.mined) if r["id"] in synth}
    hard = sorted(i for i, x in rate.items() if x < args.threshold)
    easy = sorted(i for i, x in rate.items() if x >= args.threshold)
    rng = random.Random(args.seed)
    items = ([{"id": i, "stratum": "hard"} for i in rng.sample(hard, min(args.n_hard, len(hard)))]
             + [{"id": i, "stratum": "easy"} for i in rng.sample(easy, min(args.n_easy, len(easy)))])

    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(item: dict) -> dict:
        rec = synth[item["id"]]
        bad, _ = to_claims(rec)
        out = llm.chat_one(messages(bad, tips=True))
        p = parse_output(out)
        return {"stratum": item["stratum"], "detect_rate": rate[item["id"]], "want_type": rec["type"],
                "task_type": rec["task_type"], "tactic": rec["tactic"], "claude": p.verdict, "claude_type": p.type,
                "original": rec["original"], "sentence": rec["sentence"], "output": out}

    map_resumable(items, run, args.out, concurrency=args.concurrency)

    rows = [r for r in kept(args.out) if r.get("claude") in ("faithful", "hallucination")]
    summary = {}
    for st in ("hard", "easy"):
        sub = [r for r in rows if r["stratum"] == st]
        valid = sum(r["claude"] == "hallucination" for r in sub)
        same = sum(r["claude"] == "hallucination" and r["claude_type"] == r["want_type"] for r in sub)
        lo, hi = wilson(valid, len(sub))
        summary[st] = {"n": len(sub), "pool": len(hard if st == "hard" else easy), "valid": valid,
                       "valid_rate": round(valid / len(sub), 3) if sub else None, "valid_ci95": [round(lo, 3), round(hi, 3)],
                       "type_matches": same,
                       "invalid_by_tactic": dict(collections.Counter(r["tactic"] for r in sub if r["claude"] != "hallucination"))}
    Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
