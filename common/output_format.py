"""critic 输出格式的解析与校验。reward、评测、难样本挖掘共用这一份实现。

格式：
<think>...</think>
<verdict>hallucination | faithful</verdict>
<evidence>ref-2: "<原文中逐字出现的片段>"</evidence>     # citation_error 且哪篇都没有出处时为 none
<type>ERROR_TYPES 之一 | none</type>
<reason><一句话></reason>        # faithful 时可省
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from common.schema import LABELS, TYPES

TAG_ORDER = ("think", "verdict", "evidence", "type", "reason")
_TAG_RE = re.compile(r"<(think|verdict|evidence|type|reason)>(.*?)</\1>", re.S)
_EVIDENCE_RE = re.compile(r'^\s*(ref-\d+)\s*[:：]\s*["“「](.*)["”」]\s*$', re.S)


@dataclass
class ParsedOutput:
    think: str | None = None
    verdict: str | None = None
    evidence_ref: str | None = None
    evidence_text: str | None = None
    evidence_none: bool = False     # <evidence>none</evidence>，只允许出现在 citation_error
    type: str | None = None
    reason: str | None = None
    format_ok: bool = False
    error: str = ""


def parse_evidence(raw: str) -> tuple[str | None, str | None]:
    m = _EVIDENCE_RE.match(raw)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_output(text: str) -> ParsedOutput:
    out = ParsedOutput()
    found = [(m.group(1), m.group(2).strip(), m.start()) for m in _TAG_RE.finditer(text)]
    tags = [t for t, _, _ in found]
    values = {t: v for t, v, _ in found}

    out.think = values.get("think")
    out.verdict = values.get("verdict")
    out.type = values.get("type")
    out.reason = values.get("reason")
    if "evidence" in values:
        out.evidence_none = values["evidence"].strip().lower() == "none"
        out.evidence_ref, out.evidence_text = parse_evidence(values["evidence"])

    # 格式检查：每个标签至多一次、顺序正确、必需标签齐全、取值合法、think 非空
    if len(tags) != len(set(tags)):
        out.error = "duplicate tag"
        return out
    if tags != [t for t in TAG_ORDER if t in tags]:
        out.error = "tag order"
        return out
    required = ["think", "verdict", "evidence", "type"]
    if out.verdict == "hallucination":
        required.append("reason")
    missing = [t for t in required if t not in values]
    if missing:
        out.error = f"missing {missing}"
        return out
    if not out.think:
        out.error = "empty think"
        return out
    if out.verdict not in LABELS:
        out.error = f"bad verdict {out.verdict!r}"
        return out
    if out.type not in TYPES:
        out.error = f"bad type {out.type!r}"
        return out
    if (out.verdict == "faithful") != (out.type == "none"):
        out.error = "verdict/type mismatch"
        return out
    if out.evidence_none:
        if out.type != "citation_error":
            out.error = "evidence none only for citation_error"
            return out
    elif out.evidence_ref is None:
        out.error = "bad evidence syntax"
        return out
    out.format_ok = True
    return out


def normalize(text: str) -> str:
    """逐字比对前的归一化：NFKC、统一引号、压缩空白。不做大小写和标点删除。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def evidence_locatable(ref_id: str | None, snippet: str | None, refs: dict[str, str],
                       min_chars: int = 8) -> bool:
    """evidence 片段是否逐字出现在指定 ref 中。min_chars 防止用极短片段（如一个词）刷定位分。"""
    if not ref_id or not snippet or ref_id not in refs:
        return False
    s = normalize(snippet)
    return len(s) >= min_chars and s in normalize(refs[ref_id])


def evidence_span(ref_id: str | None, snippet: str | None,
                  refs: dict[str, str]) -> tuple[str, int, int] | None:
    """返回 (ref_id, start, end)，位置基于归一化后的文本；找不到返回 None。"""
    if not ref_id or not snippet or ref_id not in refs:
        return None
    s, r = normalize(snippet), normalize(refs[ref_id])
    i = r.find(s) if s else -1
    return None if i < 0 else (ref_id, i, i + len(s))


def format_output(verdict: str, evidence_ref: str | None, evidence_text: str | None, type_: str,
                  reason: str = "", think: str = "") -> str:
    """构造一条合规输出（SFT 目标、单测用）。evidence_ref 为 None 时输出 <evidence>none</evidence>。"""
    evidence = "none" if evidence_ref is None else f'{evidence_ref}: "{evidence_text}"'
    parts = [
        f"<think>{think}</think>",
        f"<verdict>{verdict}</verdict>",
        f"<evidence>{evidence}</evidence>",
        f"<type>{type_}</type>",
    ]
    if reason or verdict == "hallucination":
        parts.append(f"<reason>{reason}</reason>")
    return "\n".join(parts)
