"""对每组文献生成综述：先出大纲，再按大纲写正文，每句标注引用的 ref id。每组生成 versions 版。

按意图构造的组（data.build_groups）带用户问题和意图：综述围绕问题写，并加上该意图的写作要求。

文献是英文，综述默认写中文（gen_review.language: zh）：中文综述才会出现术语翻译错误（term_error），
这也是实际使用场景。生成时不给术语词典，让译名错误自然出现，由 critic 和词典预检去查。

用法：python -m data.gen_review --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import re

from common.config import load_config
from common.llm import LLM, LLMConfig, map_resumable
from common.schema import read_jsonl
from data.intents import INTENTS

OUTLINE_PROMPT = """You are writing a short literature review based ONLY on the references below.{task}
Write an outline of 3-5 sections. For each section give a one-line focus and the reference ids it will use.

{refs}"""

BODY_PROMPT = """Write the review following the outline.{task} Rules:
- Use only information in the references.{intent_rule}
- Every factual sentence must end with its citation(s), like [ref-1] or [ref-2, ref-3].
- Plain paragraphs, no headings, no bullet points. {length}
- Output only the review itself: no notes, explanations or comments about the outline or the references.
{language_rule}

References:
{refs}

Outline:
{outline}"""


LANGUAGE_RULES = {
    "zh": "- Write in Simplified Chinese. Translate medical terms into standard Chinese terms; abbreviations "
          "(HbA1c, eGFR) and drug names may stay in English. Put the citation before the Chinese full stop, "
          "like ……下降了 1.1% [ref-1]。",
    "en": "",
}
LENGTH = {"zh": "500-1000 Chinese characters.", "en": "300-600 words."}


def task_text(g: dict) -> tuple[str, str]:
    """有意图的组（data.build_groups）：综述要回答用户问题，并按意图的写法写；旧的按主题分组没有这两项。"""
    if not g.get("question"):
        return "", ""
    return (f"\nThe review answers this user question: {g['question']}",
            f"\n- {INTENTS[g['intent']]['writing_rule']}")


_TRAILER = re.compile(r"\n\s*(?:-{3,}|\*{3,}|说明[:：]|注[:：]|备注[:：]|Note[:：])[\s\S]*$")


def strip_trailer(text: str) -> str:
    """去掉模型在正文后附加的说明（实测 Claude 会在 --- 之后解释为何没写大纲里的某一点），否则会被切成 claim。"""
    return _TRAILER.sub("", text).strip()


def fmt_refs(refs: dict[str, str]) -> str:
    return "\n\n".join(f"[{rid}] {text}" for rid, text in refs.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--only", default="", help="只跑这些文献组，逗号分隔的 group_id（试跑用）")
    args = ap.parse_args()
    only = {x for x in args.only.split(",") if x}
    cfg = load_config(args.config)
    gcfg = cfg["gen_review"]
    llm = LLM(LLMConfig.from_dict(cfg["generator"]))

    items = []
    groups_path = (cfg.get("build_groups") or cfg["fetch"])["out"]
    for g in read_jsonl(groups_path):
        if only and g["group_id"] not in only:
            continue
        for v in range(gcfg["versions"]):
            items.append({"id": f"{g['group_id']}-v{v}", "group": g})

    def gen(item: dict) -> dict:
        g = item["group"]
        refs = fmt_refs(g["refs"])
        task, intent_rule = task_text(g)
        outline = llm.chat_one([{"role": "user", "content": OUTLINE_PROMPT.format(refs=refs, task=task)}],
                               temperature=gcfg["temperature"])
        lang = gcfg.get("language", "zh")
        body = BODY_PROMPT.format(refs=refs, outline=outline, task=task, intent_rule=intent_rule,
                                  length=LENGTH[lang], language_rule=LANGUAGE_RULES[lang])
        text = strip_trailer(llm.chat_one([{"role": "user", "content": body}], temperature=gcfg["temperature"]))
        return {"group_id": g["group_id"], "split": g["split"], "refs": g["refs"],
                "intent": g.get("intent"), "question": g.get("question"),
                "outline": outline, "text": text}

    map_resumable(items, gen, gcfg["out"], concurrency=llm.cfg.concurrency)


if __name__ == "__main__":
    main()
