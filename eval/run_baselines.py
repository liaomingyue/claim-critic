"""非 LLM-prompt 类基线：MiniCheck、RefChecker。两者只给 verdict，没有 evidence 和 type，
所以只参与 F1 和按类型召回，不参与定位准确率。

零样本大模型、同基座 SFT、critic-0 三个基线直接用 eval/run_eval.py。

两个库的接口随版本变化，requirements 里各自 pin 版本，升级时核对这里的调用。

用法：python -m eval.run_baselines --test data_release/test.jsonl --which minicheck --out outputs/eval/minicheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.output_format import format_output
from common.schema import read_jsonl
from eval.metrics import aggregate, bootstrap_f1, per_sample


def _as_output(verdict: str) -> str:
    """把二分类结果包成 critic 格式，复用同一套指标代码。evidence 为空，定位项自然不计分。"""
    return format_output(verdict, "ref-0", "", "none" if verdict == "faithful" else "missing_condition",
                         think="baseline")


def run_minicheck(samples: list[dict], model_name: str) -> list[str]:
    from minicheck.minicheck import MiniCheck

    scorer = MiniCheck(model_name=model_name, enable_prefix_caching=False)
    docs = ["\n".join(s["refs"].values()) for s in samples]
    claims = [s["claim"] for s in samples]
    pred_label, _raw_prob, _, _ = scorer.score(docs=docs, claims=claims)
    return ["faithful" if y == 1 else "hallucination" for y in pred_label]


def run_refchecker(samples: list[dict], model_name: str) -> list[str]:
    from refchecker import LLMChecker

    checker = LLMChecker(model=model_name)
    # claim 级：每条 claim 作为单个待检陈述，参考文献拼接为一个 reference
    labels = checker.check(batch_claims=[[s["claim"]] for s in samples],
                           batch_references=["\n".join(s["refs"].values()) for s in samples],
                           max_reference_segment_length=0)
    # 只有 Entailment 视为 faithful；Neutral（无依据）和 Contradiction 都算 hallucination
    return ["faithful" if lb[0] == "Entailment" else "hallucination" for lb in labels]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--which", choices=["minicheck", "refchecker"], required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = list(read_jsonl(args.test))
    if args.which == "minicheck":
        verdicts = run_minicheck(samples, args.model_name or "Bespoke-MiniCheck-7B")
    else:
        verdicts = run_refchecker(samples, args.model_name or "gpt-4o")

    rows = [per_sample(s, _as_output(v)) for s, v in zip(samples, verdicts)]
    for r in rows:
        r["loc_ok"] = None
    res = aggregate(rows)
    res["f1_ci95"] = bootstrap_f1(rows)
    res["model"] = args.which
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
