"""RAGTruth 评测指标。输入是 (gold 句子样本, 模型原始输出) 对；回答级结果由同一回答的逐句结果汇总。

- 句级 P/R/F1（hallucination 类）：主指标之一，critic 直接优化的目标。不计大片段标出的句子（coarse，见 ragtruth.convert）。
  无法解析出 verdict 的输出按「与真值相反」计，避免格式崩坏反而占便宜。
- 回答级 P/R/F1：任一句判为幻觉即整条回答为幻觉，对应 RAGTruth 论文的 response-level，可与已发表结果比较。
  按任务（QA / Summary / Data2txt）和生成模型拆开。
- 类型：找对的幻觉句中 conflict / baseless 也对的比例；句级召回按 conflict / baseless 拆开。
- 证据：evidence_located_rate = 给出的原文片段中逐字可定位的比例（允许「开头 ... 结尾」）；
  conflict_evidence_rate = 真值为 conflict 的句子给出了可定位片段的比例。
"""
from __future__ import annotations

import random
from collections import defaultdict

from ragtruth.task import evidence_in_source, parse_output


def per_sample(gold: dict, output: str, style: str = "full") -> dict:
    p = parse_output(output, style)
    pred = p.verdict if p.verdict in ("faithful", "hallucination") else None
    if pred is None:
        pred = "faithful" if gold["label"] == "hallucination" else "hallucination"
    located = p.evidence is not None and evidence_in_source(p.evidence, gold["source"])
    return {
        "id": gold["id"], "response_id": gold["response_id"], "task_type": gold["task_type"], "model": gold["model"],
        "gold": gold["label"], "gold_type": gold["type"], "pred": pred, "pred_type": p.type,
        "format_ok": p.format_ok, "ev_quoted": p.evidence is not None, "ev_located": located,
        "coarse": bool(gold.get("coarse")),
    }


def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def prf(rows: list[dict]) -> dict:
    tp = sum(r["gold"] == r["pred"] == "hallucination" for r in rows)
    return _prf(tp, sum(r["pred"] == "hallucination" for r in rows), sum(r["gold"] == "hallucination" for r in rows))


def responses(rows: list[dict]) -> list[dict]:
    """逐句结果 → 回答级结果。"""
    by = defaultdict(list)
    for r in rows:
        by[r["response_id"]].append(r)
    out = []
    for rid, rs in by.items():
        out.append({"response_id": rid, "task_type": rs[0]["task_type"], "model": rs[0]["model"],
                    "gold": "hallucination" if any(r["gold"] == "hallucination" for r in rs) else "faithful",
                    "pred": "hallucination" if any(r["pred"] == "hallucination" for r in rs) else "faithful"})
    return out


def aggregate(rows: list[dict]) -> dict:
    """句级指标不计大片段标出的句子（coarse），回答级用全部句子。"""
    all_rows, rows = rows, [r for r in rows if not r["coarse"]]
    res = {"sentence": prf(rows)}
    res["sentence"]["accuracy"] = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
    resp = responses(all_rows)
    res["response"] = prf(resp)
    res["response"]["accuracy"] = sum(r["gold"] == r["pred"] for r in resp) / len(resp)
    for kind in ("conflict", "baseless"):
        sub = [r for r in rows if r["gold_type"] == kind]
        res[f"sentence_recall_{kind}"] = sum(r["pred"] == "hallucination" for r in sub) / len(sub) if sub else None
    found = [r for r in rows if r["gold"] == r["pred"] == "hallucination"]
    res["type_accuracy"] = sum(r["pred_type"] == r["gold_type"] for r in found) / len(found) if found else None
    quoted = [r for r in rows if r["ev_quoted"]]
    res["evidence_located_rate"] = sum(r["ev_located"] for r in quoted) / len(quoted) if quoted else None
    conflict = [r for r in rows if r["gold_type"] == "conflict"]
    res["conflict_evidence_rate"] = sum(r["ev_located"] for r in conflict) / len(conflict) if conflict else None
    for key in ("task_type", "model"):
        res[f"response_f1_by_{key}"] = {v: prf([r for r in resp if r[key] == v])["f1"]
                                        for v in sorted({r[key] for r in resp})}
    res["format_rate"] = sum(r["format_ok"] for r in all_rows) / len(all_rows)
    res["n_sentences"], res["n_coarse_excluded"], res["n_responses"] = len(rows), len(all_rows) - len(rows), len(resp)
    return res


def bootstrap_f1(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """句级和回答级 F1 的 95% bootstrap 置信区间，按回答重采样（同一回答的句子不独立）。"""
    by = defaultdict(list)
    for r in rows:
        by[r["response_id"]].append(r)
    groups = list(by.values())
    rng = random.Random(seed)
    sent, resp = [], []
    for _ in range(n_boot):
        sample = [r for g in (rng.choice(groups) for _ in groups) for r in g]
        sent.append(prf(sample)["f1"])
        resp.append(prf(responses(sample))["f1"])
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"sentence": (sorted(sent)[lo], sorted(sent)[hi]), "response": (sorted(resp)[lo], sorted(resp)[hi])}
