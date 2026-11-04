"""RAGTruth++ 「有效幻觉」子集（2026-10-29，用户同意，约 $20–25）：++ 比原标注多出的 542 句幻觉里，人工抽查 40 句只有约一半是真幻觉，
其余是吹毛求疵 / 常识等价、或开头套话里报的字数。让 Claude 逐句按 4 类重标，评测时同时报告严格召回（++ 全部）和有效召回。

  real            真幻觉：张冠李戴、跨段拼接、数字 / 时间 / 确定性被改、与原文矛盾、无依据的具体断言
  mild            技术上无依据但影响很小、可争议（轻微加码）
  nitpick         不算错：同义改写、常识等价（缩写 / 别称）、无害的填充词、合理推断，或原文其实支持（标注错）
  meta_wordcount  回答自述（「Here is a summary in N words」之类）里的字数 / 格式声明

Claude 看到原文、整条回答（逐句编号）、要判的句子和 ++ 标出的片段及其标签；只判这 542 句，原标注就有的 87 句视为有效。
有效口径（rtpp rescore / mcnemar 的 valid 列）：正例 = 原标注已有 + 本文件判 real 的；判 mild / nitpick / meta_wordcount 的句子不参与计分。

用法：python -m ragtruth.pp_valid label --out outputs/rt/analysis/pp_valid.jsonl
     python -m ragtruth.pp_valid agree --manual outputs/rt/analysis/pp_new542_sample40.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.task import expand_claims

OUT = "outputs/rt/analysis/pp_valid.jsonl"
CATS = ("real", "mild", "nitpick", "meta_wordcount")

SYSTEM = """You are auditing hallucination annotations for a retrieval-augmented generation benchmark.
A response was written from a source (a question with passages, a news article to summarize, or structured business data).
Annotators marked spans in one sentence of the response as hallucinated. Decide whether that sentence really contains a hallucination, and put it in exactly one category:

real: a careful reader would count it as a factual error against the source. Examples: facts attributed to the wrong person, entity, passage or step; details from different parts of the source spliced together wrongly; a changed number, date, quantity or order; certainty strengthened or weakened ("may" becoming "likely"); a direct contradiction; a specific claim the source does not support.
mild: technically not supported by the source, but the change is small and reasonable people could disagree, such as a slight embellishment that barely changes the meaning.
nitpick: not an error. Examples: a paraphrase; an equivalence any reader knows (an abbreviation, a well-known alternative name); a harmless filler word; a reasonable inference from the source; or the source actually supports the sentence, so the annotation is wrong.
meta_wordcount: the flagged part is the response describing itself, such as "Here is a summary in 106 words", and the issue is the stated word count or format.

Check the source yourself. Do not assume the annotation is right.
Answer in this format:
<reason>one or two sentences</reason>
<category>real|mild|nitpick|meta_wordcount</category>"""


def _L(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


def new_sentences(pp_path: str) -> list[dict]:
    """++ 判幻觉、原标注没标的逐句样本（不含粗粒度句），附 ++ 片段。"""
    out = []
    for r in read_jsonl(pp_path):
        ob = {int(k) for k in r["orig_bad"]}
        spans = _L(r["spans"])
        for c in expand_claims(r):
            if c.get("coarse") or c["label"] != "hallucination" or c["idx"] in ob:
                continue
            a, b = _L(r["sent_spans"])[c["idx"] - 1]
            sp = [(r["response"][s:e], lab) for s, e, lab in spans if s < b and e > a]
            out.append({**c, "pp_spans": sp})
    return sorted(out, key=lambda c: c["id"])


def render(c: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(c["sentences"], 1))
    spans = "\n".join(f'- "{t}" ({lab})' for t, lab in c["pp_spans"]) or "- (whole sentence)"
    return (f"<source>\n{c['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Sentence to judge: [S{c['idx']}] {c['sentences'][c['idx'] - 1]}\n\nAnnotated spans:\n{spans}")


def parse(out: str) -> str | None:
    m = re.findall(r"<category>\s*([a-z_]+)\s*</category>", out)
    return m[-1] if m and m[-1] in CATS else None


def load(path: str = OUT) -> dict[str, str]:
    """id → 类别（只含解析成功的）。"""
    return {r["id"]: r["category"] for r in kept(path) if r.get("category")} if Path(path).exists() else {}


def cmd_label(args) -> None:
    items = new_sentences(args.pp)
    print(json.dumps({"sentences": len(items), "by_task": dict(collections.Counter(c["task_type"] for c in items))}))
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(c: dict) -> dict:
        out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(c)}])
        return {"id": c["id"], "task_type": c["task_type"], "pp_type": c["type"], "pp_spans": c["pp_spans"],
                "sentence": c["sentences"][c["idx"] - 1], "category": parse(out), "output": out}

    map_resumable(items, run, args.out, concurrency=args.concurrency)
    res = list(kept(args.out))
    summ = {"n": len(res), "unparsed": sum(r["category"] is None for r in res),
            "by_category": dict(collections.Counter(r["category"] for r in res)),
            "by_task": {t: dict(collections.Counter(r["category"] for r in res if r["task_type"] == t))
                        for t in sorted({r["task_type"] for r in res})}}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


def cmd_agree(args) -> None:
    lab = load(args.labels)
    man = json.load(open(args.manual, encoding="utf-8"))
    pairs = [(m["judgement"], lab[m["id"]]) for m in man if m["id"] in lab]
    valid = lambda x: x == "real"
    print(json.dumps({"n": len(pairs), "exact": round(sum(a == b for a, b in pairs) / len(pairs), 3),
                      "valid_agree": round(sum(valid(a) == valid(b) for a, b in pairs) / len(pairs), 3),
                      "confusion": dict(collections.Counter(f"{a}->{b}" for a, b in pairs))}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label")
    a.add_argument("--pp", default="outputs/external/ragtruth/test_pp.jsonl")
    a.add_argument("--model", default="claude-opus-5-5")
    a.add_argument("--effort", default="high")
    a.add_argument("--concurrency", type=int, default=8)
    a.add_argument("--out", default=OUT)
    g = sub.add_parser("agree")
    g.add_argument("--labels", default=OUT)
    g.add_argument("--manual", required=True)
    args = ap.parse_args()
    {"label": cmd_label, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    main()
