"""推理：一篇带引用的综述 + 参考文献 → 逐条 claim 判定 → Markdown 报告。
切分器、对照文献选取、术语表与训练数据构建用的是同一份代码（data/split_claims.py、common/context.py、
common/glossary.py），critic 上线时看到的输入和训练时同分布。

两条规则在 critic 之外兜底：引用了不存在的文献编号直接判 citation_error；术语词典预检的结果单列一栏，
和 critic 的判定并排给人看，不覆盖 critic。

输入 JSON：{"text": "综述正文，句末带 [ref-1] 引用", "refs": {"ref-1": "...", ...}}

用法：python -m infer.judge --input review.json --model critic-3 --out report.md
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common.context import select_context
from common.glossary import default_glossary
from common.llm import LLM, LLMConfig
from common.output_format import parse_output
from common.prompts import critic_messages
from common.schema import cited_refs
from data.split_claims import split_claims
from train.reward import evidence_ok


def judge_review(llm: LLM, text: str, refs: dict[str, str], concurrency: int = 16, context_k: int = 2,
                 check_uncited: bool = False) -> list[dict]:
    """check_uncited：没标引用的句子也送检（critic 在所有文献里都找不到支持就判 citation_error、证据 none）。"""
    glossary = default_glossary()
    claims = []
    for c in split_claims(text, keep_uncited=check_uncited):
        claims.append({"claim": c["claim"], "cited": c["ref_ids"],
                       "missing_refs": [r for r in c["ref_ids"] if r not in refs],
                       "refs": select_context(c["claim"], c["ref_ids"], refs, context_k)})

    def run(c: dict) -> dict:
        # 规则预检：参考文献里的英文术语在 claim 里被译成了词典中的误译
        term_hits = [f"{t.en} 应为「{t.zh}」，写成了「{w}」" for t, w in glossary.mistranslations(cited_refs(c), c["claim"], c["refs"])]
        if c["missing_refs"]:
            # 引用了输入里不存在的文献编号：不送 critic，规则直接判引用挂载错误
            return {**c, "verdict": "hallucination", "type": "citation_error", "by_rule": True,
                    "reason": f"引用了不存在的参考文献 {', '.join(c['missing_refs'])}", "evidence_ok": False,
                    "term_check": term_hits}
        p = parse_output(llm.chat_one(critic_messages(c)))
        return {**c, "verdict": p.verdict, "type": p.type, "reason": p.reason or "", "by_rule": False,
                "evidence": "none" if p.evidence_none else f"{p.evidence_ref}: {p.evidence_text}",
                "evidence_ref": p.evidence_ref, "evidence_text": p.evidence_text,
                "evidence_ok": p.format_ok and p.verdict in ("faithful", "hallucination")
                and evidence_ok(p, p.verdict, p.type, c["refs"], c["cited"]),
                "format_ok": p.format_ok, "term_check": term_hits}

    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(run, claims))


def to_markdown(results: list[dict]) -> str:
    n_h = sum(r["verdict"] == "hallucination" for r in results)
    lines = [f"# 忠实性检查报告", "", f"共 {len(results)} 条 claim，疑似幻觉 {n_h} 条。", "",
             "| # | 判定 | 类型 | 引用 | claim | 原因 | 证据合规 | 术语预检 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(results, 1):
        claim = r["claim"].replace("|", "\\|")
        reason = (r.get("reason") or "").replace("|", "\\|")
        located = "是" if r.get("evidence_ok") else "否"
        cited = ", ".join(r.get("cited") or []) or "无"
        terms = "；".join(r.get("term_check") or []).replace("|", "\\|")
        lines.append(f"| {i} | {r['verdict']} | {r.get('type') or ''} | {cited} | {claim} | {reason} | {located} | {terms} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context_k", type=int, default=2, help="与 configs/data.yaml 的 split_claims.context_k 保持一致")
    ap.add_argument("--check_uncited", action="store_true", help="没标引用的句子也送检")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        doc = json.load(f)
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, temperature=0.0, max_tokens=1024))
    results = judge_review(llm, doc["text"], doc["refs"], context_k=args.context_k,
                           check_uncited=args.check_uncited)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(to_markdown(results))
    print(f"{sum(r['verdict'] == 'hallucination' for r in results)}/{len(results)} flagged -> {args.out}")


if __name__ == "__main__":
    main()
