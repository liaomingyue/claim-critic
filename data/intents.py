"""用户意图：决定检索哪类文献、问题怎么问、综述怎么写。

文献类型（kinds）由 data.study_type 按标题和摘要的规则判断，不用 Europe PMC 的 PUB_TYPE 标签：
那个标签来自 MEDLINE 人工标引，新文章和很多开放获取期刊的文章没有，按它过滤会漏掉大部分临床研究。

问「前沿研究」只该引临床研究，问「用药」要有药品说明书 —— 选哪些文献由意图决定，
而不是一个主题下什么都混在一起。诊疗规范类暂不做：PMC 里 CC-BY 的指南几乎为 0。
"""
from __future__ import annotations


INTENTS: dict[str, dict] = {
    "frontier": {
        "label": "前沿研究",
        "kinds": ["clinical"],
        "recent_years": 3,
        "drug_label": False,
        "question_hint": "asks about the newest clinical research findings on a specific question "
                         "(e.g. 某药在某人群中的最新研究结果)",
        "writing_rule": "Answer with the newest clinical study findings. For each finding state the study design, "
                        "population and main result with its numbers.",
    },
    "progress": {
        "label": "最新进展",
        "kinds": ["meta", "clinical"],
        "recent_years": 3,
        "drug_label": False,
        "question_hint": "asks what the recent progress or current state of evidence is on a topic "
                         "(e.g. 某疗法近几年有什么进展)",
        "writing_rule": "Summarize how the evidence has developed, where studies agree or conflict, and prefer pooled "
                        "estimates from meta-analyses when available.",
    },
    "overview": {
        "label": "主题科普",
        "kinds": ["review", "meta"],
        "recent_years": None,
        "drug_label": False,
        "question_hint": "asks for an explanation of a disease, mechanism or treatment concept for a non-specialist "
                         "(e.g. 某病是什么、为什么要做某检查)",
        "writing_rule": "Explain for a general reader: definition, mechanism, main evidence and current practice. "
                        "Use plain language and only the key numbers.",
    },
    "medication": {
        "label": "用药",
        "kinds": ["clinical"],
        "recent_years": None,
        "drug_label": True,
        "question_hint": "asks a practical medication question about one specific drug: indication, dosing, "
                         "contraindications, interactions or adverse effects (e.g. 肾功能不全时某药怎么用)",
        "writing_rule": "Answer the medication question. Take indications, dosing, contraindications and warnings "
                        "from the drug label reference; use clinical studies only for supporting evidence.",
    },
}

QUESTION_PROMPT = """You are building a dataset of questions that Chinese-speaking clinicians or patients ask a medical \
assistant. Topic: {topic}. Intent: {label} — the user {hint}.

Write {n} distinct questions for this topic and intent. Each question must be answerable from published clinical \
literature in English. For each question also give 2-4 English search keywords that must all appear in the title or \
abstract of a relevant paper (short, specific terms; no phrases longer than 3 words){drug_note}.

Output only a JSON array, no other text:
[{{"question": "<question in Simplified Chinese>", "keywords": ["...", "..."]{drug_field}}}]"""

DRUG_NOTE = ", and the English generic name of the one drug the question is about"
DRUG_FIELD = ', "drug": "<generic name>"'


def question_prompt(topic: str, intent: str, n: int) -> str:
    spec = INTENTS[intent]
    drug = spec["drug_label"]
    return QUESTION_PROMPT.format(topic=topic, label=spec["label"], hint=spec["question_hint"], n=n,
                                  drug_note=DRUG_NOTE if drug else "", drug_field=DRUG_FIELD if drug else "")
