"""人工审核记录的存储、校验、统计与回流。纯函数为主，审核服务（review/server.py）和
训练集组装（data/build_trainset.py）共用。

存储：outputs/review/<dataset>.jsonl，只追加不改写；同一 id 以最后一条为准，保留完整修改历史。
一条记录：
  {id, dataset, decision: accept|reject, flags: [...], edits: {claim?, cited?, label?, type?, evidence?, reason?},
   note, reviewer, ts, base_hash}
base_hash 是审核时原样本内容的哈希。原样本被重跑改变后，旧审核的 edits 不再套用（标为 stale），reject 仍然生效。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from common.glossary import default_glossary
from common.output_format import evidence_span, normalize, parse_evidence
from common.schema import Sample, append_jsonl, cited_ids, cited_refs, read_jsonl

EDITABLE = ("claim", "cited", "label", "type", "evidence", "reason")
DECISIONS = ("accept", "reject")
POSITIVE_FLAGS = ("judge_wrong", "evidence_off")
NEGATIVE_FLAGS = ("type_wrong", "not_an_error", "evidence_off")
FLAG_LABELS = {
    "judge_wrong": "强模型判错（其实有幻觉）",
    "type_wrong": "类型标错",
    "not_an_error": "不是真错误",
    "evidence_off": "证据错位",
}


# ---------- 数据集登记 ----------

def datasets(cfg: dict, max_batch: int = 3) -> dict[str, dict]:
    """可审核的数据集：名称 -> {path, audit, flags, threshold}。审核记录总是记在主数据集名下，
    打开 audit 子集时也一样，所以审核结果能直接回流到主文件。"""
    out = {
        "positives": {"path": cfg["label_positive"]["out"], "audit": cfg["label_positive"]["audit_out"],
                      "flags": POSITIVE_FLAGS, "threshold": ("judge_wrong", cfg["label_positive"]["max_audit_error"])},
    }
    for b in range(1, max_batch + 1):
        out[f"synth_batch{b}"] = {"path": cfg["synth_negative"]["out"].format(batch=b), "audit": None,
                                  "flags": NEGATIVE_FLAGS, "threshold": None}
        out[f"hard_batch{b}"] = {"path": cfg["mine_hard"]["out"].format(batch=b),
                                 "audit": cfg["mine_hard"]["audit_out"].format(batch=b),
                                 "flags": NEGATIVE_FLAGS,
                                 "threshold": ("not_an_error", cfg["mine_hard"]["max_not_error_ratio"])}
    out["coldstart"] = {"path": cfg["build_trainset"]["coldstart_out"], "audit": None,
                        "flags": POSITIVE_FLAGS + NEGATIVE_FLAGS, "threshold": None}
    return out


def review_path(review_dir: str | Path, dataset: str) -> Path:
    return Path(review_dir) / f"{dataset}.jsonl"


# ---------- 读写 ----------

def content_hash(sample: dict) -> str:
    key = json.dumps({k: sample.get(k) for k in EDITABLE}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def load_reviews(review_dir: str | Path, dataset: str) -> dict[str, dict]:
    """id -> 最新一条审核记录。"""
    path = review_path(review_dir, dataset)
    latest: dict[str, dict] = {}
    if path.exists():
        for r in read_jsonl(path):
            latest[r["id"]] = r
    return latest


def history(review_dir: str | Path, dataset: str, sample_id: str) -> list[dict]:
    path = review_path(review_dir, dataset)
    return [r for r in read_jsonl(path) if r["id"] == sample_id] if path.exists() else []


def make_review(sample: dict, dataset: str, decision: str, flags: list[str], edits: dict,
                note: str = "", reviewer: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"bad decision {decision!r}")
    edits = {k: v for k, v in edits.items() if k in EDITABLE and v != sample.get(k)}
    return {"id": sample["id"], "dataset": dataset, "decision": decision, "flags": sorted(set(flags)),
            "edits": edits, "note": note, "reviewer": reviewer,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "base_hash": content_hash(sample)}


def save_review(review_dir: str | Path, review: dict) -> None:
    append_jsonl(review_path(review_dir, review["dataset"]), review)


# ---------- 校验 ----------

def validate(sample: dict, parent: dict | None = None, min_edit: int = 1, max_edit: int = 40) -> dict:
    """对（可能被人工修改过的）样本做与流水线相同的校验。errors 阻止保存为 accept，warnings 只提示。

    返回 errors、warnings，以及证据在归一化原文中的位置（供前端高亮）。
    """
    from data.synth_negative import edit_distance, tokens  # 延迟导入，避免循环依赖

    errors, warnings = [], []
    try:
        Sample.from_dict(sample)
    except (TypeError, ValueError) as e:
        errors.append(str(e))

    label, type_ = sample.get("label"), sample.get("type")
    refs = sample.get("refs", {})
    cited = cited_ids(sample) if refs else []
    evidence = (sample.get("evidence") or "").strip()
    ref_id, snippet = parse_evidence(evidence)
    span = evidence_span(ref_id, snippet, refs)
    if evidence.lower() == "none":
        if type_ != "citation_error":
            errors.append("证据 none 只用于引用错误中的无中生有（哪篇文献都没有出处）")
    elif not evidence:
        errors.append("缺少证据")
    elif ref_id is None:
        errors.append('证据格式应为 ref-N: "原文片段"')
    elif span is None:
        errors.append(f"证据没有逐字出现在 {ref_id} 中")
    elif len(normalize(snippet)) < 8:
        errors.append("证据少于 8 个字符")
    elif type_ == "citation_error" and ref_id in cited:
        errors.append("引用错误的证据应出自内容真正的出处（未被引用的文献）；哪篇都没有出处时填 none")
    elif label == "faithful" and cited and ref_id not in cited:
        errors.append("faithful 的证据应出自被引用的文献")
    if type_ == "citation_error" and not cited and evidence.lower() != "none":
        errors.append("挂载错误需要标注引用（cited 不能为空）；没标引用的句子只可能是无中生有，证据填 none")
    if type_ == "term_error" and refs and not default_glossary().mistranslations(cited_refs(sample), sample.get("claim", ""), refs):
        warnings.append("术语词典没有命中误译：确认是术语错误，必要时把这对译名补进 configs/glossary.tsv")

    if sample.get("label") == "hallucination" and not (sample.get("reason") or "").strip():
        errors.append("hallucination 样本需要填写原因")

    distance = None
    if parent and label == "hallucination":
        distance = edit_distance(tokens(parent["claim"]), tokens(sample.get("claim", "")))
        if type_ == "citation_error" and evidence.lower() != "none":
            if distance:
                warnings.append("挂载错误应只改引用、不改 claim 文本")
            if cited == cited_ids(parent):
                warnings.append("挂载错误的引用与原 claim 相同，没有挂错")
        elif not (min_edit <= distance <= max_edit):
            warnings.append(f"与原 claim 的编辑距离为 {distance}，不在 [{min_edit}, {max_edit}] 内")

    return {"errors": errors, "warnings": warnings, "edit_distance": distance,
            "evidence_span": list(span) if span else None}


# ---------- 回流 ----------

def apply_reviews(rows: list[dict], reviews: dict[str, dict]) -> tuple[list[dict], Counter]:
    """拒绝的样本剔除；accept 且有 edits 的套用修改；原样本已变化的旧修改不套用（stale）。"""
    out, stats = [], Counter()
    for r in rows:
        rv = reviews.get(r["id"])
        if rv is None:
            out.append(r)
            stats["unreviewed"] += 1
            continue
        if rv["decision"] == "reject":
            stats["rejected"] += 1
            continue
        if rv.get("edits"):
            if rv.get("base_hash") != content_hash(r):
                stats["stale_edits_skipped"] += 1
                out.append(r)
                continue
            r = {**r, **rv["edits"], "meta": {**r.get("meta", {}), "human_edited": True,
                                              "reviewer": rv.get("reviewer", "")}}
            if r["label"] == "faithful":
                r["type"], r["reason"] = "none", r.get("reason", "")
            stats["edited"] += 1
        else:
            stats["accepted"] += 1
        out.append(r)
    return out, stats


def reviewed(review_dir: str | Path, dataset: str, rows: list[dict]) -> list[dict]:
    """build_trainset 用：读审核记录并回流，打印统计。"""
    out, stats = apply_reviews(rows, load_reviews(review_dir, dataset))
    if len(stats) > 1 or "unreviewed" not in stats:
        print(f"[review] {dataset}: {dict(stats)}")
    return out


# ---------- 统计 ----------

def summary(rows: list[dict], reviews: dict[str, dict], audit_ids: set[str] | None,
            threshold: tuple[str, float] | None) -> dict:
    """审核进度与阈值判定。阈值只在抽检子集上算（抽检是随机的，才能代表整批）。"""
    ids = {r["id"] for r in rows}
    done = {i: rv for i, rv in reviews.items() if i in ids}
    flags = Counter(f for rv in done.values() for f in rv.get("flags", []))
    res = {"total": len(rows), "reviewed": len(done),
           "rejected": sum(rv["decision"] == "reject" for rv in done.values()),
           "edited": sum(bool(rv.get("edits")) for rv in done.values()),
           "flags": dict(flags)}
    if audit_ids is not None:
        audit_done = [rv for i, rv in done.items() if i in audit_ids]
        res["audit_total"], res["audit_reviewed"] = len(audit_ids), len(audit_done)
        if threshold and audit_done:
            flag, limit = threshold
            ratio = sum(flag in rv.get("flags", []) for rv in audit_done) / len(audit_done)
            res["threshold"] = {"flag": flag, "ratio": ratio, "limit": limit, "over": ratio > limit,
                                "final": len(audit_done) == len(audit_ids)}
    return res
