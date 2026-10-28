"""评测指标。输入是 (gold Sample dict, 模型原始输出) 对。

- F1（hallucination 类）：主指标。无法解析出 verdict 的输出按「与真值相反」计，避免格式崩坏反而占便宜。
- 按类型召回：gold 为某类错误的样本中，预测为 hallucination 的比例。
- 定位准确率：verdict 正确的样本中，预测 evidence 与人工 evidence 在同一 ref 内字符区间有重叠的比例；
  gold 证据为 none（引用错误中的无中生有）时没有原文可引，预测写 none 即算定位正确。
- 格式合规率：parse_output 判定合规的比例。
"""
from __future__ import annotations

import random

from common.output_format import evidence_span, parse_evidence, parse_output
from common.schema import ERROR_TYPES, is_unsourced


def _overlap(a: tuple[str, int, int] | None, b: tuple[str, int, int] | None) -> bool:
    return bool(a and b and a[0] == b[0] and a[1] < b[2] and b[1] < a[2])


def per_sample(gold: dict, output: str) -> dict:
    p = parse_output(output)
    pred = p.verdict if p.verdict in ("faithful", "hallucination") else None
    if pred is None:  # 无法解析：按错误处理
        pred = "faithful" if gold["label"] == "hallucination" else "hallucination"
    gold_span = evidence_span(*parse_evidence(gold.get("evidence", "")), gold["refs"])
    pred_span = evidence_span(p.evidence_ref, p.evidence_text, gold["refs"])
    if is_unsourced(gold):
        loc_ok = p.evidence_none
    else:
        loc_ok = _overlap(gold_span, pred_span) if gold_span else None
    return {
        "id": gold["id"],
        "gold": gold["label"],
        "gold_type": gold.get("type", "none"),
        "pred": pred,
        "pred_type": p.type,
        "format_ok": p.format_ok,
        "loc_ok": loc_ok,
    }


def f1_halluc(rows: list[dict]) -> dict:
    tp = sum(r["gold"] == "hallucination" and r["pred"] == "hallucination" for r in rows)
    fp = sum(r["gold"] == "faithful" and r["pred"] == "hallucination" for r in rows)
    fn = sum(r["gold"] == "hallucination" and r["pred"] == "faithful" for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def aggregate(rows: list[dict]) -> dict:
    res = f1_halluc(rows)
    res["accuracy"] = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
    for t in ERROR_TYPES:
        sub = [r for r in rows if r["gold_type"] == t]
        res[f"recall_{t}"] = sum(r["pred"] == "hallucination" for r in sub) / len(sub) if sub else None
    correct = [r for r in rows if r["gold"] == r["pred"] and r["loc_ok"] is not None]
    res["loc_acc"] = sum(r["loc_ok"] for r in correct) / len(correct) if correct else None
    res["format_rate"] = sum(r["format_ok"] for r in rows) / len(rows)
    res["n"] = len(rows)
    return res


def bootstrap_f1(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """F1 的 95% bootstrap 置信区间。1000 条测试集上差 1–2 个点可能只是噪声，报告里要带上。"""
    rng = random.Random(seed)
    scores = sorted(f1_halluc([rng.choice(rows) for _ in rows])["f1"] for _ in range(n_boot))
    return scores[int(0.025 * n_boot)], scores[int(0.975 * n_boot) - 1]


def compare(ours: dict, base: dict, key: str = "f1") -> dict:
    """绝对差值和相对提升都报。"""
    a, b = ours[key], base[key]
    return {"abs": a - b, "rel": (a - b) / b if b else None}
