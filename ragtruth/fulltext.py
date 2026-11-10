"""合成错误句的全文核对：改后句「改动或加入的信息」在原文别处是否其实有依据。有依据 = 假错误，剔除。

起因：9B critic 识破不了的合成样本里约 30% 无效，几乎都是「35B 以为加了原文没有的信息，
其实在原文别处有」（Data2txt 的另一条评论、摘要原文的后半段）。合成时的复核只对照了原句附近，没有逐处查全文。
这一步专门让模型先说清改了什么，再在整份原文里逐段找它。

两条迭代序列共用同一个过滤后的合成池（ragtruth.iterate），过滤不是对比变量。

用法：python -m ragtruth.fulltext --synth outputs/rt/synth/r1.jsonl --out outputs/rt/synth/r1_ft.jsonl
"""
from __future__ import annotations

import argparse
import collections

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth.synth import parse_json_obj

PROMPT = """You are checking whether an edited sentence really contains an error with respect to a source.

[source]
{source}

[original sentence] {orig}
[edited sentence] {new}

Step 1: state exactly what information the edited sentence changes or adds compared to the original sentence.
Step 2: search the ENTIRE source for that information: every passage, every review, every field, not only the part the original sentence was based on. Paraphrases, synonyms and simple inferences count as support.
Step 3: decide whether the changed or added information is stated or directly supported anywhere in the source.

Output only JSON: {{"changed": "...", "supported_somewhere": true or false, "quote": "the supporting text from the source, or empty"}}"""


def check_one(gen: LLM, rec: dict, extra: dict | None = None) -> dict:
    text = gen.chat_one([{"role": "user", "content": PROMPT.format(
        source=rec["source"].strip(), orig=rec["original"], new=rec["sentence"])}], temperature=0.0, **(extra or {}))
    obj = parse_json_obj(text) or {}
    sup = obj.get("supported_somewhere")
    return {"changed": obj.get("changed"), "supported": sup if isinstance(sup, bool) else None,
            "quote": obj.get("quote")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gen")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    gen = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=512, concurrency=args.concurrency,
                        timeout=600))
    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    recs = list(read_jsonl(args.synth))
    raw = args.out.replace(".jsonl", ".raw.jsonl")
    map_resumable(recs, lambda r: {"fulltext": check_one(gen, r, extra)}, raw, concurrency=args.concurrency)
    ft = {r["id"]: r["fulltext"] for r in kept(raw)}
    ok = [{**r, "fulltext": ft[r["id"]]} for r in recs if r["id"] in ft and ft[r["id"]]["supported"] is False]
    print(f"fulltext: kept {write_jsonl(args.out, ok)}/{len(recs)}; "
          f"removed by type {dict(collections.Counter(r['type'] for r in recs if ft.get(r['id'], {}).get('supported')))}; "
          f"unparsed {sum(ft.get(r['id'], {}).get('supported') is None for r in recs)}")


if __name__ == "__main__":
    main()
