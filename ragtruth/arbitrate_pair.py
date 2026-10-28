"""两个 critic 判定不一致的句子交给 Claude 独立裁定，估计人工标签噪声对两者差距的影响（诊断 A）。

e2（只判定）和 e3（证据 + 理由）在同一测试句上判定不同时，按真值和谁判对分四层：
  a_fp：真值无、A 判对、B 判幻觉（B 多出来的误报）；b_fp：真值无、B 判对、A 判幻觉；
  a_fn：真值有、A 判对、B 判无（B 多出来的漏报）；b_fn：真值有、B 判对、A 判无。
每层抽样交给 Claude（看不到人工标注和两个 critic 的输出，输入、判定要点、输出格式同 ragtruth.arbitrate），
按 Claude 站在人工一边的比例外推到整层，重算两个 critic 的句级 P/R/F1。只改不一致的句子，两者都判对 / 都判错的句子不动，
所以改正后的是「差距」的估计，不是绝对水平。

用法：python -m ragtruth.arbitrate_pair --a outputs/rt/eval/sft-0.6b-e2 --a_style verdict \
          --b outputs/rt/eval/sft-0.6b-e3 --b_style full --out outputs/rt/analysis/arb_e2_e3.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.metrics import _prf
from ragtruth.task import expand_claims, messages, parse_output

STRATA = {"a_fp": 30, "b_fp": 20, "a_fn": 15, "b_fn": 15}


def preds(path: str, style: str) -> dict[str, bool]:
    return {p["id"]: parse_output(p["output"], style).verdict == "hallucination"
            for p in read_jsonl(f"{path}/predictions.jsonl")}


def stratum(gold_hal: bool, a: bool, b: bool) -> str | None:
    if a == b:
        return None
    a_right = a == gold_hal
    return ("a_fn" if a_right else "b_fn") if gold_hal else ("a_fp" if a_right else "b_fp")


def corrected_prf(gold: dict[str, bool], pred: dict[str, bool], flip_rate: dict[str, float],
                  strata: dict[str, list[str]]) -> dict:
    """按每层「Claude 不同意人工」的比例，把该层句子的真值按比例翻转后重算 P/R/F1（期望值）。"""
    tp = fp = fn = 0.0
    in_stratum = {i: s for s, ids in strata.items() for i in ids}
    for i, g in gold.items():
        p = pred[i]
        q = flip_rate.get(in_stratum.get(i), 0.0)   # 真值翻转的概率
        tp += p * ((1 - q) * g + q * (not g))
        fp += p * ((1 - q) * (not g) + q * g)
        fn += (not p) * ((1 - q) * g + q * (not g))
    return _prf(tp, tp + fp, tp + fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--a", required=True)
    ap.add_argument("--a_style", default="verdict")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b_style", default="full")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    claims = {c["id"]: c for s in read_jsonl(args.test) for c in expand_claims(s) if not c["coarse"]}
    pa, pb = preds(args.a, args.a_style), preds(args.b, args.b_style)
    ids = sorted(i for i in claims if i in pa and i in pb)
    gold = {i: claims[i]["label"] == "hallucination" for i in ids}
    strata = collections.defaultdict(list)
    for i in ids:
        s = stratum(gold[i], pa[i], pb[i])
        if s:
            strata[s].append(i)
    rng = random.Random(args.seed)
    items = [{"id": i, "stratum": s} for s, n in STRATA.items() for i in rng.sample(strata[s], min(n, len(strata[s])))]
    print({s: len(v) for s, v in strata.items()}, f"sampling {len(items)}")

    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(item: dict) -> dict:
        out = llm.chat_one(messages(claims[item["id"]], tips=True))
        p = parse_output(out)
        return {"stratum": item["stratum"], "claude": p.verdict, "claude_type": p.type, "output": out}

    map_resumable(items, run, args.out, concurrency=args.concurrency)

    rows = [r for r in kept(args.out) if r.get("claude") in ("faithful", "hallucination")]
    summary = {"stratum_sizes": {s: len(v) for s, v in strata.items()}}
    flip = {}
    for s in STRATA:
        sub = [r for r in rows if r["stratum"] == s]
        agree = sum((r["claude"] == "hallucination") == gold[r["id"]] for r in sub)
        flip[s] = (len(sub) - agree) / len(sub) if sub else 0.0
        summary[s] = {"n": len(sub), "claude_agrees_with_human": agree, "flip_rate": round(flip[s], 3)}
    for name, p in (("a", pa), ("b", pb)):
        sub = {i: p[i] for i in ids}
        summary[f"{name}_human"] = corrected_prf(gold, sub, {}, strata)
        summary[f"{name}_corrected"] = corrected_prf(gold, sub, flip, strata)
    Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
