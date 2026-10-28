"""可视化应用的业务逻辑：读文件、聚合统计、生成表格和 HTML。纯函数，不依赖 gradio，单测见 tests/test_app_stats.py。

文件不存在时返回 None（或空列表），由 UI 层显示「尚未生成」。
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

from common.output_format import evidence_span, normalize
from common.schema import ERROR_TYPES, read_jsonl
from eval.metrics import compare
from train.reward import score_components, total_reward

NOT_READY = "尚未生成"
POSITIVE_AUDIT_MAX_ERROR = 0.03
HARD_AUDIT_MAX_NOT_ERROR = 0.10
METRIC_COLUMNS = ["model", "f1", "f1_ci95", "precision", "recall"] + \
                 [f"recall_{t}" for t in ERROR_TYPES] + ["loc_acc", "format_rate", "n"]
_REF_LINE = re.compile(r"^\s*\(?(ref-\d+)\)?\s*[:：]\s*(.*)$")


# ---------- 读文件 ----------

def load_kept(path: str | Path) -> list[dict] | None:
    """读 JSONL 并去掉 map_resumable 的丢弃占位行；文件不存在返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    return [r for r in read_jsonl(path) if not r.get("_dropped")]


def sample_files(cfg: dict, max_batch: int = 3) -> dict[str, str]:
    """看板里可浏览的数据文件：名称 -> 路径（不检查是否存在）。"""
    files = {
        "claims": cfg["split_claims"]["out"],
        "positives": cfg["label_positive"]["out"],
    }
    for b in range(1, max_batch + 1):
        files[f"synth_batch{b}"] = cfg["synth_negative"]["out"].format(batch=b)
        files[f"hard_batch{b}"] = cfg["mine_hard"]["out"].format(batch=b)
    return files


# ---------- 数据看板 ----------

def pipeline_counts(cfg: dict, max_batch: int = 3) -> list[dict]:
    rows = []
    for name, path in sample_files(cfg, max_batch).items():
        data = load_kept(path)
        rows.append({"stage": name, "path": path, "n": NOT_READY if data is None else len(data)})
    return rows


def hard_distribution(cfg: dict, max_batch: int = 3) -> list[dict]:
    """各批难样本按 (batch, round, type) 计数，供柱状图使用。"""
    rows = []
    for b in range(1, max_batch + 1):
        data = load_kept(cfg["mine_hard"]["out"].format(batch=b)) or []
        for (rnd, t), n in sorted(Counter((r.get("round", 0), r.get("type")) for r in data).items()):
            rows.append({"batch": b, "round": rnd, "type": t, "n": n})
    return rows


def hard_meta_distribution(rows: list[dict], bins: int = 4) -> dict[str, list[dict]]:
    """难样本的 detect_rate 分桶和 rewrites 次数分布。"""
    rates = Counter()
    rewrites = Counter()
    for r in rows:
        meta = r.get("meta") or {}
        if "detect_rate" in meta:
            # 检出率落在 [i/bins, (i+1)/bins) 桶里，1.0 归入最后一桶
            i = min(int(meta["detect_rate"] * bins), bins - 1)
            rates[f"{i / bins:.2f}-{(i + 1) / bins:.2f}"] += 1
        if "rewrites" in meta:
            rewrites[int(meta["rewrites"])] += 1
    return {
        "detect_rate": [{"bucket": k, "n": v} for k, v in sorted(rates.items())],
        "rewrites": [{"rewrites": k, "n": v} for k, v in sorted(rewrites.items())],
    }


def _ratio_row(file: str, metric: str, values: list, threshold: float | None, bad=False) -> dict:
    """values 中 None 表示尚未人工检查；ratio = 等于 bad 的比例（只在已检查的样本上算）。"""
    checked = [v for v in values if v is not None]
    ratio = sum(v == bad for v in checked) / len(checked) if checked else None
    return {"file": file, "metric": metric, "n": len(values), "checked": len(checked),
            "ratio": ratio, "threshold": threshold,
            "over": bool(threshold is not None and ratio is not None and ratio > threshold)}


def audit_summary(cfg: dict, max_batch: int = 3) -> list[dict]:
    """人工抽检统计，数据来自审核台写的审核记录（review/store.py），只在抽检子集里已审核的样本上算。
    正样本看 judge_wrong（阈值 3%），难样本看三项问题标记（not_an_error 阈值 10%）。"""
    from review import store

    review_dir = cfg.get("review", {}).get("dir", "outputs/review")
    rows = []
    for name, ds in store.datasets(cfg, max_batch).items():
        audit = load_kept(ds["audit"]) if ds["audit"] else None
        if audit is None:
            continue
        reviews = store.load_reviews(review_dir, name)
        values = [reviews.get(r["id"]) for r in audit]
        limit = ds["threshold"][1] if ds["threshold"] else None
        for flag in ds["flags"]:
            flagged = [None if rv is None else flag in rv.get("flags", []) for rv in values]
            is_limit = bool(ds["threshold"] and ds["threshold"][0] == flag)
            rows.append(_ratio_row(ds["audit"], flag, flagged, limit if is_limit else None, bad=True))
    return rows


def filter_samples(rows: list[dict], type_: str | None = None, round_: int | None = None,
                   limit: int = 200) -> list[dict]:
    """样本浏览器：按 type / round 过滤，返回便于展示的扁平行。"""
    out = []
    for r in rows:
        if type_ and r.get("type") != type_:
            continue
        if round_ is not None and r.get("round") != round_:
            continue
        out.append({
            "id": r.get("id"), "label": r.get("label", ""), "type": r.get("type", ""),
            "round": r.get("round", ""), "claim": r.get("claim", ""),
            "refs": "\n".join(f"({k}) {v}" for k, v in (r.get("refs") or {}).items()),
            "evidence": r.get("evidence", ""), "reason": r.get("reason", ""),
        })
        if len(out) >= limit:
            break
    return out


# ---------- 评测对比 ----------

def scan_metrics(eval_dir: str | Path) -> list[dict]:
    """扫描 eval_dir/*/metrics.json；model 字段缺失时用目录名。"""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return []
    results = []
    for p in sorted(eval_dir.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        m.setdefault("model", p.parent.name)
        m["_dir"] = p.parent.name
        results.append(m)
    return results


def metrics_table(results: list[dict], baseline: str | None = None) -> list[dict]:
    """对比表。指定 baseline（model 名）时，加上 F1 相对它的绝对差和相对提升。"""
    base = next((r for r in results if r["model"] == baseline), None) if baseline else None
    rows = []
    for r in results:
        row = {c: r.get(c) for c in METRIC_COLUMNS}
        if isinstance(row["f1_ci95"], (list, tuple)):
            row["f1_ci95"] = f"[{row['f1_ci95'][0]:.3f}, {row['f1_ci95'][1]:.3f}]"
        if base is not None:
            d = compare(r, base)
            row["f1_abs_vs_base"] = d["abs"]
            row["f1_rel_vs_base"] = d["rel"]
        rows.append(row)
    return rows


# ---------- 在线判定 Demo ----------

def parse_refs(text: str) -> dict[str, str]:
    """参考文献输入：JSON 对象，或每行 `ref-1: 文本`。"""
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return {str(k): str(v) for k, v in json.loads(text).items()}
    refs: dict[str, str] = {}
    for line in text.splitlines():
        m = _REF_LINE.match(line)
        if m:
            refs[m.group(1)] = m.group(2).strip()
    return refs


def highlight_refs(refs: dict[str, str], ref_id: str | None, snippet: str | None) -> str:
    """把参考文献渲染成 HTML，证据片段用 <mark> 高亮。

    evidence_span 返回的是归一化文本上的位置，所以展示的也是归一化后的文本。
    """
    span = evidence_span(ref_id, snippet, refs)
    parts = []
    for rid, text in refs.items():
        t = normalize(text)
        if span and span[0] == rid:
            s, e = span[1], span[2]
            body = f"{html.escape(t[:s])}<mark>{html.escape(t[s:e])}</mark>{html.escape(t[e:])}"
        else:
            body = html.escape(t)
        parts.append(f"<p><b>({html.escape(rid)})</b> {body}</p>")
    return "\n".join(parts)


def judge_table(results: list[dict]) -> list[dict]:
    return [{"#": i, "verdict": r.get("verdict"), "type": r.get("type") or "",
             "claim": r.get("claim", ""), "reason": r.get("reason", ""),
             "evidence_ok": bool(r.get("evidence_ok"))}
            for i, r in enumerate(results, 1)]


def render_claims_html(results: list[dict]) -> str:
    """每条 claim 一个可展开块；hallucination 标红，展开后显示原因和高亮证据的参考文献。"""
    blocks = []
    for i, r in enumerate(results, 1):
        bad = r.get("verdict") != "faithful"
        color = "#c0392b" if bad else "#1e8449"
        summary = (f"<span style='color:{color};font-weight:600'>[{html.escape(str(r.get('verdict')))}]</span> "
                   f"{i}. {html.escape(r.get('claim', ''))}")
        detail = []
        if r.get("type") and r.get("type") != "none":
            detail.append(f"<p><b>类型：</b>{html.escape(str(r['type']))}</p>")
        if r.get("reason"):
            detail.append(f"<p><b>原因：</b>{html.escape(r['reason'])}</p>")
        detail.append(f"<p><b>证据可定位：</b>{'是' if r.get('evidence_ok') else '否'}</p>")
        detail.append(highlight_refs(r.get("refs") or {}, r.get("evidence_ref"), r.get("evidence_text")))
        style = "border-left:4px solid " + color + ";padding:4px 8px;margin:6px 0"
        blocks.append(f"<details style='{style}'><summary>{summary}</summary>{''.join(detail)}</details>")
    return "\n".join(blocks) or "<p>没有切出带引用的 claim。</p>"


def reward_breakdown(output: str, label: str, type_: str, refs: dict[str, str],
                     cited: list[str] | None = None) -> dict:
    """Reward 计算器：三项得分、总分和解析错误原因。"""
    if label == "faithful":
        type_ = "none"
    comp = score_components(output, label, type_, refs, cited)
    return {"label": comp["label"], "format": comp["format"], "loc": comp["loc"],
            "total": total_reward(comp), "parse_error": comp["parse_error"] or "(无)"}
