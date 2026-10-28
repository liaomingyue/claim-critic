"""测试集复核：critic 与人工标注不一致的句子（误报 + 漏报），交给更强的模型独立重判，算「复核后」的指标。

错误分析发现约 1/3 的误报其实是人工漏标的幻觉，另有一部分漏报来自整段大片段的粗标注（2026-10-26），
只按原始标注算会低估 critic。这里：
- 只复核不一致的句子（gold ≠ critic 预测），外加 n_control 条一致的句子作对照，估计复核模型自身的错判率；
- 复核模型看不到人工标注和 critic 的输出，用和 critic 相同的判定格式（system prompt 带 TIPS）独立判断；
- 复核后的真值 = 不一致句子用复核模型的判定，其余句子保留人工标注。原始指标和复核后指标都报。

用法：
  vllm serve /root/autodl-tmp/models/Qwen3.5-35B-A3B-GPTQ-Int4 --served-model-name judge ...
  python -m ragtruth.adjudicate --pred outputs/rt/eval/sft-0.6b-claim/predictions.jsonl --model judge \
      --out outputs/rt/eval/sft-0.6b-claim/adjudicated.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.metrics import aggregate, per_sample
from ragtruth.task import expand_claims, messages, parse_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--pred", required=True, help="critic 的 predictions.jsonl（ragtruth.eval 产出）")
    ap.add_argument("--model", default="judge")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n_control", type=int, default=300)
    ap.add_argument("--think", action="store_true", help="复核模型开思考模式")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = {c["id"]: c for s in read_jsonl(args.test) for c in expand_claims(s)}
    preds = {p["id"]: p["output"] for p in read_jsonl(args.pred) if p["id"] in gold}
    rows = {i: per_sample(gold[i], o) for i, o in preds.items()}
    disagree = [i for i, r in rows.items() if r["gold"] != r["pred"]]
    agree = sorted(set(rows) - set(disagree))
    control = random.Random(0).sample(agree, min(args.n_control, len(agree)))
    todo = [{"id": i, "control": i in set(control)} for i in disagree + control]
    print(f"{len(disagree)} disagreements + {len(control)} controls to adjudicate")

    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, temperature=0.0, max_tokens=args.max_tokens,
                        concurrency=args.concurrency, timeout=600))
    extra = {} if args.think else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    def judge(item: dict) -> dict:
        out = llm.chat_one(messages(gold[item["id"]], tips=True), **extra)
        p = parse_output(out.split("</think>")[-1])
        return {"control": item["control"], "judge": p.verdict, "judge_type": p.type, "output": out}

    map_resumable(todo, judge, args.out, concurrency=args.concurrency)
    res = {r["id"]: r for r in kept(args.out)}

    ctrl = [r for r in res.values() if r["control"] and r["judge"] in ("faithful", "hallucination")]
    ctrl_agree = sum(r["judge"] == rows[r["id"]]["gold"] for r in ctrl) / len(ctrl) if ctrl else None
    flips = {"fp_judge_says_hallucination": 0, "fp_total": 0, "fn_judge_says_faithful": 0, "fn_total": 0}
    corrected = []
    for i, r in rows.items():
        g = dict(gold[i])
        j = res.get(i)
        if i in set(disagree) and j and j["judge"] in ("faithful", "hallucination"):
            key = "fp" if r["gold"] == "faithful" else "fn"
            flips[f"{key}_total"] += 1
            if j["judge"] != r["gold"]:
                flips["fp_judge_says_hallucination" if key == "fp" else "fn_judge_says_faithful"] += 1
                g["label"] = j["judge"]
                g["type"] = j["judge_type"] if j["judge"] == "hallucination" else "none"
                g["coarse"] = False
        corrected.append(per_sample(g, preds[i]))
    summary = {"original": aggregate(list(rows.values())), "adjudicated": aggregate(corrected),
               "control_agreement_with_human": ctrl_agree, "n_control": len(ctrl), **flips}
    Path(args.out).with_name("metrics_adjudicated.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("original", "adjudicated")}, indent=2))
    for k in ("original", "adjudicated"):
        print(k, "sentence", summary[k]["sentence"], "response", summary[k]["response"])


if __name__ == "__main__":
    main()
