"""RAGTruth 回答级 critic：整条回答一次输入、一次输出，判出所有不被原文支持的句子（格式对比实验 ①/⑤/⑤b）。

两种输出（style）：
- plain：只给有问题的句子和最终判定
    <errors>
    S2: conflict
    </errors>                    # 无问题时 <errors>none</errors>
    <verdict>hallucination | faithful</verdict>
- evidence：先给每一句一行原文证据（结构对所有回答相同，不出现标签），再给有问题的句子和最终判定
    <analysis>
    S1: "<原文逐字片段>"         # 原文没有相关内容或句子没有事实内容时写 none
    S2: none
    </analysis>
    <errors>...</errors>
    <verdict>...</verdict>

样本是回答样本（ragtruth.convert 产出）；evidence 风格还需要 ragtruth.annotate 补写的 analysis。
评测把回答级输出展开成逐句预测，指标与逐句 critic 完全一致（ragtruth.metrics）。
"""
from __future__ import annotations

import re

from ragtruth.task import expand_claims, gold_bad

STYLES = ("plain", "evidence")

_RULES = """Find every sentence that contains information not supported by the source:
- conflict: the sentence contradicts the source.
- baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.
Judge only against the source, not against world knowledge. Sentences without factual content (greetings, restating the task) are supported."""

SYSTEM = {
    "plain": f"""You check whether an assistant's response is faithful to the source material it was given.

The source is the exact input the assistant received (instructions plus material). The response is split into numbered sentences.
{_RULES}

Output exactly this format and nothing else:
<errors>
S<number>: conflict or baseless
</errors>
<verdict>hallucination or faithful</verdict>
List one line per unsupported sentence. When no sentence is unsupported, write <errors>none</errors> and <verdict>faithful</verdict>.""",
    "evidence": f"""You check whether an assistant's response is faithful to the source material it was given.

The source is the exact input the assistant received (instructions plus material). The response is split into numbered sentences.
First, for every sentence in order, copy word for word the part of the source that the sentence corresponds to, contradicts, or is most closely related to (for a long passage, its beginning and end joined by " ... "); write none if the source has nothing related or the sentence has no factual content.
Then {_RULES[0].lower()}{_RULES[1:]}

Output exactly this format and nothing else:
<analysis>
S1: "exact text from the source" or none
(one line per sentence, S1 to the last sentence)
</analysis>
<errors>
S<number>: conflict or baseless
</errors>
<verdict>hallucination or faithful</verdict>
When no sentence is unsupported, write <errors>none</errors> and <verdict>faithful</verdict>.""",
}

_ERR_RE = re.compile(r"^\s*S(\d+)\s*[:：]\s*(conflict|baseless)\s*$", re.I)
_AN_RE = re.compile(r'^\s*S(\d+)\s*[:：]\s*(none|["“].*["”])\s*$', re.I)


def render_input(s: dict) -> str:
    lines = ["[source]", s["source"].strip(), "", "[response]"]
    return "\n".join(lines + [f"[S{i}] {t}" for i, t in enumerate(s["sentences"], 1)])


def messages(s: dict, style: str = "plain") -> list[dict]:
    return [{"role": "system", "content": SYSTEM[style]}, {"role": "user", "content": render_input(s)}]


def target_text(s: dict, style: str = "plain") -> str:
    gb = gold_bad(s)
    parts = []
    if style == "evidence":
        lines = []
        for i, a in enumerate(s["analysis"], 1):
            ev = a.get("evidence")
            lines.append(f"S{i}: none" if ev is None else f'S{i}: "{ev}"')
        parts += ["<analysis>", *lines, "</analysis>"]
    if gb:
        parts += ["<errors>", *(f"S{k}: {v}" for k, v in sorted(gb.items())), "</errors>",
                  "<verdict>hallucination</verdict>"]
    else:
        parts += ["<errors>none</errors>", "<verdict>faithful</verdict>"]
    return "\n".join(parts)


def parse(text: str, n: int, style: str = "plain") -> dict:
    """返回 {verdict, bad: {句号: 类型}, format_ok}。"""
    verdict = re.search(r"<verdict>(.*?)</verdict>", text, re.S)
    verdict = verdict.group(1).strip().lower() if verdict else None
    errors = re.search(r"<errors>(.*?)</errors>", text, re.S)
    bad, ok = {}, verdict in ("faithful", "hallucination") and errors is not None
    if errors and errors.group(1).strip().lower() != "none":
        for line in errors.group(1).strip().splitlines():
            m = _ERR_RE.match(line)
            if m and 1 <= int(m.group(1)) <= n:
                bad[int(m.group(1))] = m.group(2).lower()
            elif line.strip():
                ok = False
    if style == "evidence":
        an = re.search(r"<analysis>(.*?)</analysis>", text, re.S)
        ids = [int(m.group(1)) for m in map(_AN_RE.match, an.group(1).strip().splitlines()) if m] if an else []
        ok = ok and ids == list(range(1, n + 1))
    ok = ok and (verdict == "hallucination") == bool(bad)
    return {"verdict": verdict, "bad": bad if verdict == "hallucination" else {}, "format_ok": ok}


def sentence_rows(s: dict, output: str, style: str = "plain") -> list[dict]:
    """回答级输出 → 逐句预测行，字段与 ragtruth.metrics.per_sample 一致（不含证据定位）。"""
    p = parse(output, len(s["sentences"]), style)
    rows = []
    for c in expand_claims(s):
        pred_type = p["bad"].get(c["idx"])
        rows.append({"id": c["id"], "response_id": s["id"], "task_type": s["task_type"], "model": s["model"],
                     "gold": c["label"], "gold_type": c["type"],
                     "pred": "hallucination" if pred_type else "faithful", "pred_type": pred_type or "none",
                     "format_ok": p["format_ok"], "ev_quoted": False, "ev_located": False,
                     "coarse": bool(c.get("coarse"))})
    return rows


def full_responses(annotated: list[dict]) -> list[dict]:
    """evidence 风格的训练数据：每一句都有合格训练目标的回答（与 plain 风格用同一批回答，只差证据行）。"""
    out = []
    for s in annotated:
        if not s.get("analysis"):
            continue
        valid = {c["idx"]: c for c in expand_claims(s, require_analysis=True)}
        if len(valid) == len(s["sentences"]):
            out.append({**s, "analysis": [{"evidence": valid[i]["evidence"], "reason": valid[i]["reason"]}
                                          for i in range(1, len(s["sentences"]) + 1)]})
    return out
