"""把带引用的综述正文切成 claim。训练数据构建和线上推理（infer/judge.py）共用这一份代码。

规则：按句切分；句首的引用标记归还上一句；过短的片段并入上一句（合并从句）；
没有引用的句子默认丢弃（多是过渡句、总起句）；keep_uncited=True 时保留，推理时用来查「没标引用、
其实是编出来的」句子。claim 文本去掉引用标记，引用的 ref id 存在 cited。

每条 claim 的 refs = 被引用的文献 + 同组最相关的 context_k 篇对照文献（common/context.py），
critic 才能区分引用错误的两种情况：内容在对照文献里（挂载错误）和哪篇都没有（无中生有）。

用法：python -m data.split_claims --reviews outputs/reviews.jsonl --out outputs/claims.jsonl [--context_k 2]
"""
from __future__ import annotations

import argparse
import re

from common.context import select_context
from common.schema import read_jsonl, write_jsonl

CITE_RE = re.compile(r"\[(ref-\d+(?:\s*[,，;；]\s*ref-\d+)*)\]")
# 英文缩写后的句点不切
_ABBREV = ("et al.", "e.g.", "i.e.", "vs.", "Fig.", "fig.", "approx.", "No.", "Dr.")
_PLACEHOLDER = "\u0000"
_SENT_END = re.compile(r"(?<=[.!?。！？])\s*(?=\S)")


def _protect(text: str) -> str:
    for a in _ABBREV:
        text = text.replace(a, a.replace(".", _PLACEHOLDER))
    # 小数点（p<0.05、1.5 倍）
    return re.sub(r"(?<=\d)\.(?=\d)", _PLACEHOLDER, text)


def split_sentences(text: str) -> list[str]:
    raw = _SENT_END.split(_protect(text))
    sents = [s.replace(_PLACEHOLDER, ".").strip() for s in raw if s.strip()]
    # 句首的引用标记（"... outcomes. [ref-1] Next ..."）归还上一句
    out: list[str] = []
    for s in sents:
        m = re.match(r"^((?:\s*\[ref-[^\]]+\])+)\s*(.*)$", s, re.S)
        if m and out:
            out[-1] = out[-1] + " " + m.group(1).strip()
            s = m.group(2).strip()
            if not s:
                continue
        out.append(s)
    return out


def extract_refs(sentence: str) -> list[str]:
    ids: list[str] = []
    for m in CITE_RE.finditer(sentence):
        for rid in re.split(r"\s*[,，;；]\s*", m.group(1)):
            if rid not in ids:
                ids.append(rid)
    return ids


def strip_citations(sentence: str) -> str:
    s = CITE_RE.sub("", sentence)
    s = re.sub(r"\s+([.,;:!?。，；：！？])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _n_tokens(s: str) -> int:
    return len(re.findall(r"[一-鿿]|\w+", s))


def split_claims(text: str, min_tokens: int = 6, keep_uncited: bool = False) -> list[dict]:
    """返回 [{claim, ref_ids}]。keep_uncited 时没有引用的句子也返回（ref_ids 为空）。"""
    merged: list[str] = []
    for s in split_sentences(text):
        # 过短且无引用的片段视为上一句的从句
        if merged and _n_tokens(strip_citations(s)) < min_tokens and not extract_refs(s):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    claims = []
    for s in merged:
        ref_ids = extract_refs(s)
        if ref_ids or (keep_uncited and _n_tokens(strip_citations(s)) >= min_tokens):
            claims.append({"claim": strip_citations(s), "ref_ids": ref_ids})
    return claims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context_k", type=int, default=2, help="每条 claim 附带的未引用对照文献篇数")
    args = ap.parse_args()

    rows = []
    for rv in read_jsonl(args.reviews):
        for i, c in enumerate(split_claims(rv["text"])):
            if any(rid not in rv["refs"] for rid in c["ref_ids"]):
                continue  # 引用了不存在的文献，生成模型自己幻觉出来的 id；推理时由规则直接判 citation_error
            rows.append({
                "id": f"{rv['id']}-c{i:03d}",
                "source_group": rv["group_id"],
                "split": rv["split"],
                "refs": select_context(c["claim"], c["ref_ids"], rv["refs"], args.context_k),
                "cited": c["ref_ids"],
                "claim": c["claim"],
                "meta": {k: rv[k] for k in ("intent", "question") if rv.get(k)},
            })
    print(f"wrote {write_jsonl(args.out, rows)} claims")


if __name__ == "__main__":
    main()
