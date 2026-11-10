"""RAGTruth 主线的 critic 任务：逐句检测。每个句子是一条样本（claim），输入给出完整源材料、完整回答，
并单独标出待检测的那一句；critic 只判断这一句。整条回答的结论由逐句结果汇总（任一句有问题即为幻觉）。

对应「基于正确回答合成错误回答 → critic 判断 → 漏检的加入训练」这条主线：合成时改动某一句，
这一句就是一条带真值的负样本。

回答样本（ragtruth.convert 产出）：id, source, sentences, sent_spans, bad（{句号: conflict|baseless}）, label,
  spans, span_notes, response, task_type, model, split, origin, 以及 ragtruth.annotate 补写的 analysis
  （每句 {"evidence": 原文片段 | None, "reason": 英文说明}）。
句子样本（expand_claims 产出）：id = <回答 id>-S<句号>，response_id, idx（1 起）, source, sentences,
  label（faithful|hallucination）, type（conflict|baseless|none）, evidence, reason, 以及回答的分组字段。

输出格式：先定位证据、再说明关系，类型和判定只在最后出现。证据和说明部分的写法对有无问题的句子完全相同。
<evidence>"<原文逐字片段，长片段可写成 开头 ... 结尾>"</evidence>   # 原文完全没有相关内容、或句子没有事实内容时写 none
<reason>...</reason>
<type>conflict | baseless | none</type>                              # none = 无幻觉
<verdict>hallucination | faithful</verdict>

纯函数，reward 由 TRL / verl 直接调用；单测见 tests/test_ragtruth.py。
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from common.output_format import normalize

TYPES = ("conflict", "baseless", "none")
MIN_EVIDENCE_CHARS = 8
ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")

SYSTEM = """You check whether one sentence of an assistant's response is faithful to the source material the assistant was given.

The source is the exact input the assistant received (instructions plus material). The full response is shown for context, split into numbered sentences; you check only the sentence given under [sentence to check].

Work in this order:
1. evidence: copy, word for word, the part of the source that the sentence corresponds to, contradicts, or is most closely related to. For a long passage you may copy its beginning and its end joined by " ... ". Write none only if the source has nothing related to the sentence, or the sentence has no factual content (greetings, restating the task).
2. reason: state how the sentence relates to the evidence: what the source says and what the sentence says.
3. type:
   - conflict: the sentence contradicts the source.
   - baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.
   - none: the sentence is supported by the source, or has no factual content.
4. verdict: hallucination if the type is conflict or baseless, otherwise faithful.
Judge only against the source, not against world knowledge.

Output exactly this format and nothing else:
<evidence>"exact text from the source" or none</evidence>
<reason>...</reason>
<type>conflict, baseless or none</type>
<verdict>hallucination or faithful</verdict>"""

# 判定要点：针对 0.6B 逐句模型在测试集上的主要错误（错误分析）。critic 的 system prompt 可选带上
# （messages(tips=True)，做「有 / 无 tips」对比），补写 evidence / reason 的 prompt 总是带上。
TIPS = """Tips:
- Structured data (JSON): True/False means the feature is offered or not; times such as 17:30-21:0 mean 5:30 PM to 9:00 PM, and a correct conversion is supported; 0:0-0:0 means no listed hours.
- Paraphrases, faithful summaries and numbers that follow directly from the source (for example by simple arithmetic) are supported.
- Check every item of a list or every detail separately: one added item or detail that the source does not contain makes the sentence baseless.
- The type must follow from the reason: if the reason says the source does not contain or contradicts something in the sentence, the type cannot be none."""

_TAG_RE = re.compile(r"<(evidence|reason|type|verdict)>(.*?)</\1>", re.S)
TAG_ORDER = ("evidence", "reason", "type", "verdict")


def render_input(c: dict) -> str:
    lines = ["[source]", c["source"].strip(), "", "[response]"]
    lines += [f"[S{i}] {t}" for i, t in enumerate(c["sentences"], 1)]
    lines += ["", "[sentence to check]", f"S{c['idx']}: {c['sentences'][c['idx'] - 1]}"]
    return "\n".join(lines)


# 只判定的逐句格式（格式对比实验 ②）：不写 evidence / reason，直接给类型和判定
SYSTEM_VERDICT = """You check whether one sentence of an assistant's response is faithful to the source material the assistant was given.

The source is the exact input the assistant received (instructions plus material). The full response is shown for context, split into numbered sentences; you check only the sentence given under [sentence to check].

type:
- conflict: the sentence contradicts the source.
- baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.
- none: the sentence is supported by the source, or has no factual content.
verdict: hallucination if the type is conflict or baseless, otherwise faithful.
Judge only against the source, not against world knowledge.

Output exactly this format and nothing else:
<type>conflict, baseless or none</type>
<verdict>hallucination or faithful</verdict>"""
# 先判定后解释（诊断实验 ④，与 e3 同数据只换顺序）：type → verdict → evidence → reason。
# 用来区分「写理由的能力不够」和「先写理由再下结论时理由写偏、结论跟着偏」；判定在最前，不作为正式格式。
SYSTEM_VFIRST = """You check whether one sentence of an assistant's response is faithful to the source material the assistant was given.

The source is the exact input the assistant received (instructions plus material). The full response is shown for context, split into numbered sentences; you check only the sentence given under [sentence to check].

Work in this order:
1. type:
   - conflict: the sentence contradicts the source.
   - baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.
   - none: the sentence is supported by the source, or has no factual content.
2. verdict: hallucination if the type is conflict or baseless, otherwise faithful.
3. evidence: copy, word for word, the part of the source that the sentence corresponds to, contradicts, or is most closely related to. For a long passage you may copy its beginning and its end joined by " ... ". Write none only if the source has nothing related to the sentence, or the sentence has no factual content (greetings, restating the task).
4. reason: state how the sentence relates to the evidence: what the source says and what the sentence says.
Judge only against the source, not against world knowledge.

Output exactly this format and nothing else:
<type>conflict, baseless or none</type>
<verdict>hallucination or faithful</verdict>
<evidence>"exact text from the source" or none</evidence>
<reason>...</reason>"""
STYLES = ("full", "verdict", "vfirst")   # full = evidence → reason → type → verdict；verdict = 只给 type → verdict
STYLE_ORDER = {"full": TAG_ORDER, "verdict": ("type", "verdict"), "vfirst": ("type", "verdict", "evidence", "reason")}


def system_prompt(tips: bool = False, style: str = "full") -> str:
    base = {"verdict": SYSTEM_VERDICT, "vfirst": SYSTEM_VFIRST}.get(style, SYSTEM)
    return base.replace("\n\nOutput exactly", f"\n\n{TIPS}\n\nOutput exactly") if tips else base


def messages(c: dict, tips: bool = False, style: str = "full") -> list[dict]:
    return [{"role": "system", "content": system_prompt(tips, style)}, {"role": "user", "content": render_input(c)}]


def format_output(evidence: str | None, reason: str, type_: str) -> str:
    ev = "none" if evidence is None else f'"{evidence}"'
    verdict = "faithful" if type_ == "none" else "hallucination"
    return f"<evidence>{ev}</evidence>\n<reason>{reason}</reason>\n<type>{type_}</type>\n<verdict>{verdict}</verdict>"


def target_text(c: dict, style: str = "full") -> str:
    head = f"<type>{c['type']}</type>\n<verdict>{'faithful' if c['type'] == 'none' else 'hallucination'}</verdict>"
    if style == "verdict":
        return head
    if style == "vfirst":
        ev = "none" if c["evidence"] is None else f'"{c["evidence"]}"'
        return f"{head}\n<evidence>{ev}</evidence>\n<reason>{c['reason']}</reason>"
    return format_output(c["evidence"], c["reason"], c["type"])


@dataclass
class Parsed:
    evidence: str | None = None
    evidence_none: bool = False
    reason: str | None = None
    type: str | None = None
    verdict: str | None = None
    format_ok: bool = False
    error: str = ""


def parse_output(text: str, style: str = "full") -> Parsed:
    out = Parsed()
    order = STYLE_ORDER[style]
    found = [(m.group(1), m.group(2).strip()) for m in _TAG_RE.finditer(text)]
    tags = [t for t, _ in found]
    values = dict(found)
    out.reason = values.get("reason")
    out.type = (values.get("type") or "").lower() or None
    out.verdict = (values.get("verdict") or "").lower() or None
    raw_ev = values.get("evidence")
    ev_ok = False
    if raw_ev is not None:
        if raw_ev.lower() == "none":
            out.evidence_none, ev_ok = True, True
        else:
            m = re.fullmatch(r'["“](.*)["”]', raw_ev, re.S)
            if m and m.group(1).strip():
                out.evidence, ev_ok = m.group(1), True
    if style == "verdict":   # 只判定格式没有 evidence / reason
        ev_ok, out.reason = True, out.reason or "-"
    if len(tags) != len(set(tags)) or tags != [t for t in order if t in tags]:
        out.error = "tag order/duplicate"
    elif [t for t in order if t not in values]:
        out.error = f"missing {[t for t in order if t not in values]}"
    elif not ev_ok:
        out.error = "bad evidence syntax"
    elif not out.reason:
        out.error = "empty reason"
    elif out.type not in TYPES:
        out.error = f"bad type {out.type!r}"
    elif out.verdict not in ("faithful", "hallucination"):
        out.error = f"bad verdict {out.verdict!r}"
    elif (out.verdict == "faithful") != (out.type == "none"):
        out.error = "verdict/type mismatch"
    out.format_ok = not out.error
    return out


_QUOTES = re.compile(r"[\"'`‘’“”]")


def _canon(text: str) -> str:
    """逐字比对前的归一化：common.output_format.normalize + 小写 + 单双引号不区分
    （模型摘原文时常把 "American Pie" 写成 'American Pie'，内容其实一致）。"""
    return _QUOTES.sub("'", normalize(text).lower())


def evidence_in_source(evidence: str, source: str, min_chars: int = MIN_EVIDENCE_CHARS) -> bool:
    """证据是否逐字出现在原文中（归一化空白、引号，不区分大小写和单双引号）。
    允许「开头 ... 结尾」的省略写法：各段按顺序逐字出现在原文中，首段和末段各至少 min_chars 个字符。"""
    src = _canon(source)
    parts = [_canon(p) for p in ELLIPSIS.split(evidence.strip())]
    parts = [p for p in parts if p]
    if not parts or len(parts[0]) < min_chars or len(parts[-1]) < min_chars:
        return False
    pos = 0
    for p in parts:
        i = src.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def repair_evidence(evidence: str, source: str, min_score: float = 90.0) -> str | None:
    """补写的证据和原文只差标点、大小写、个别词时，换成原文中对应的逐字片段；差得多返回 None。
    只用于整理训练目标（ragtruth.build），不用于给 critic 打分。"""
    if evidence_in_source(evidence, source):
        return evidence
    if ELLIPSIS.search(evidence):
        return None
    from rapidfuzz import fuzz

    al = fuzz.partial_ratio_alignment(evidence.lower(), source.lower())
    if al is None or al.score < min_score:
        return None
    fixed = source[al.dest_start:al.dest_end].strip()
    return fixed if evidence_in_source(fixed, source) else None


# 说明里表达「原文没有 / 与原文不符」的措辞。训练目标要求说明与类型一致：
# 类型为 none 的说明里不能出现（否则就是「说明说有问题、结论说没问题」，测试时 0.6B 学出了 244 句这样的输出）；
# 类型为 conflict / baseless 的说明里必须出现（说清差异在哪）。
# 分两级：NEGATIVE 明确表示「原文没有 / 与原文不符」；CONTRAST 只是对照措辞（while the sentence…、rather than…），
# 中性或支持性的说明里也常用（「原文列出店名，while the sentence 说它是餐厅」）。
# 「does not contain factual content」是在说句子本身没有事实内容（寒暄、复述任务），不算。
_NO_FACT = re.compile(r"(?:does not|do not|doesn't) (?:contain|make|include|present) (?:any )?(?:specific |verifiable )?"
                      r"(?:factual|verifiable) (?:content|claims?|information|statements?)"
                      r"|(?:contains?|has|makes) no (?:specific |verifiable )?(?:factual|verifiable) (?:content|claims?|information)"
                      r"|no factual (?:content|claims?)|not a (?:specific )?factual claim", re.I)
NEGATIVE = re.compile(
    r"\b(?:does|do|did) not (?:explicitly |directly |specifically |actually )?"
    r"(?:mention|state|say|specify|include|provide|contain|support|list|indicate|confirm|refer"
    r"|describe|discuss|use|give|show|report)"
    r"|\bnot (?:explicitly )?(?:mentioned|stated|specified|included|provided|contained|supported|listed|found|present|in the source)"
    r"|\bno (?:mention|information|evidence|indication|data)\b|\bnowhere\b|\babsent\b"
    r"|\bcontradict|\bincorrect|\binconsistent\b|\bdiffers? from\b|\bmismatch|\bunsupported\b"
    r"|\bnot derived\b|\bnew information\b|\bcannot be verified\b", re.I)
CONTRAST = re.compile(r"\b(?:while|but|whereas) the sentence\b|\brather than\b|\binstead\b|\bwhereas\b"
                      r"|\bonly (?:mentions|states|says|lists|provides)\b|\badds?\b|\badding\b", re.I)
# 结论词，以及补写模型偶尔写进说明的「ground truth / annotator」（泄露了补写时看到的真值）
LABEL_WORDS = re.compile(r"hallucinat|faithful|conflict|baseless|ground truth|annotator", re.I)


_NEGATED_NEG = re.compile(r"without (?:adding |introducing |including )?(?:any )?(?:unsupported|new|extra) \w+", re.I)


def says_negative(reason: str) -> bool:
    """去掉「没有事实内容」「without adding unsupported details」这类否定的否定后，是否明确说原文没有 / 不符。"""
    return bool(NEGATIVE.search(_NEGATED_NEG.sub("", _NO_FACT.sub("", reason))))


def reason_consistent(reason: str, type_: str) -> bool:
    """none：说明里不能明确说「原文没有 / 不符」；conflict / baseless：说明里要说出差异（明确否定或对照措辞）。"""
    if type_ == "none":
        return not says_negative(reason)
    return says_negative(reason) or bool(CONTRAST.search(reason))


def sentence_target_ok(evidence: str | None, reason: str, type_: str, source: str) -> bool:
    """逐句训练目标是否合格：证据逐字可定位（conflict 必须有证据），说明非空、不含结论词、与类型一致。"""
    if evidence is None:
        if type_ == "conflict":
            return False
    elif not evidence_in_source(evidence, source):
        return False
    return bool(reason) and not LABEL_WORDS.search(reason) and reason_consistent(reason, type_)


def gold_bad(s: dict) -> dict[int, str]:
    return {int(k): v for k, v in s["bad"].items()}


def expand_claims(s: dict, require_analysis: bool = False) -> list[dict]:
    """回答样本 → 逐句样本。有 analysis 时带上每句的 evidence / reason（训练目标）。
    require_analysis：只保留训练目标合格的句子（证据先经 repair_evidence 修正），按句取舍，
    一条回答里个别句子的证据没摘准不影响其他句子。"""
    gb = gold_bad(s)
    analysis = s.get("analysis")
    if require_analysis and not analysis:
        return []
    keep = {k: s[k] for k in ("source", "sentences", "task_type", "model", "split", "origin") if k in s}
    coarse = set(s.get("coarse") or [])
    out = []
    for i in range(1, len(s["sentences"]) + 1):
        a = analysis[i - 1] if analysis else {}
        ev, reason, type_ = a.get("evidence"), a.get("reason", ""), gb.get(i, "none")
        if require_analysis:
            # 大片段标出的句子不进训练；补写模型标了「标签存疑」且说明确实与真值相反的句子也不进训练。
            # 35B 试跑时会给说明与真值一致的句子也标 disputed（寒暄句、它其实同意的句子），这种按普通句子处理。
            if i in coarse or (a.get("disputed") and not reason_consistent(reason, type_)):
                continue
            if isinstance(ev, str):
                ev = repair_evidence(ev, s["source"]) or ev
            if not sentence_target_ok(ev, reason, type_, s["source"]):
                continue
        out.append({**keep, "id": f"{s['id']}-S{i}", "response_id": s["id"], "idx": i,
                    "label": "hallucination" if i in gb else "faithful", "type": type_,
                    "evidence": ev, "reason": reason, "coarse": i in coarse})
    return out


def balance(claims: list[dict], faithful_per_bad: float | None, seed: int = 0) -> list[dict]:
    """保留全部有问题的句子，无问题的句子按比例抽样；None 表示不下采样（保持自然比例）。
    1:2 下采样时训练集幻觉句占 33%、测试集只有 8%，模型误报偏多（错误分析）。"""
    bad = [c for c in claims if c["label"] == "hallucination"]
    good = [c for c in claims if c["label"] == "faithful"]
    rng = random.Random(seed)
    if faithful_per_bad is not None:
        good = rng.sample(good, min(len(good), round(len(bad) * faithful_per_bad)))
    out = bad + good
    rng.shuffle(out)
    return out


# ---- reward：总分 = 1.0 × 标签 + 0.3 × 格式 + 0.3 × 类型 + 0.2 × 证据 ----
# 类型以标签正确为前提，faithful 判对时类型记满分；证据以格式合规为前提：
# 给了片段就必须逐字可定位，真值为 conflict 时必须给片段（写 none 不得分）。
DEFAULT_WEIGHTS = {"label": 1.0, "format": 0.3, "type": 0.3, "evidence": 0.2}


def score_components(output: str, c: dict) -> dict[str, float]:
    p = parse_output(output)
    label_ok = p.verdict == c["label"]
    type_ok = label_ok and p.type == c["type"]
    if not p.format_ok:
        ev_ok = False
    elif p.evidence_none:
        ev_ok = c["type"] != "conflict"
    else:
        ev_ok = evidence_in_source(p.evidence, c["source"])
    return {"label": float(label_ok), "format": float(p.format_ok), "type": float(type_ok),
            "evidence": float(ev_ok), "pred_hallucination": float(p.verdict == "hallucination")}


def total_reward(comp: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return sum(w[k] * comp[k] for k in w)


def trl_reward(completions, gold, **kwargs) -> list[float]:
    """TRL GRPOTrainer 的 reward_funcs 入口。数据集的 gold 列是句子样本 JSON（见 gold_json）。"""
    out = []
    for comp, g in zip(completions, gold):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        out.append(total_reward(score_components(text, json.loads(g) if isinstance(g, str) else g)))
    return out


def gold_json(c: dict) -> str:
    return json.dumps({"label": c["label"], "type": c["type"], "source": c["source"]}, ensure_ascii=False)
