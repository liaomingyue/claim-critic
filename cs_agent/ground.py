"""客服回复的事实核对（纯代码）：回复里出现的金额、订单号、商品 / 款式 id、支付方式 id、用户 id，必须在此前的工具返回或用户的话里出现过，
金额也可以是此前 calculate 的结果，或两个已出现金额的和 / 差（客服自己算的差价、合计）。对不上的标为「无依据陈述」候选（E）。只覆盖结构化事实；「可以取消」这类判断性的话交给 9B 判断。
"""
from __future__ import annotations

import re

MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
IDS = [re.compile(p) for p in (r"#W\d{7}", r"\b\d{10}\b", r"\b(?:credit_card|gift_card|paypal)_\d+\b", r"\b[a-z]+_[a-z]+_\d{4}\b")]


def _nums(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text or ""):
        s = m.group(0).replace(",", "")
        try:
            out.add(f"{float(s):.2f}")
        except ValueError:
            pass
    return out


def check_replies(messages: list[dict]) -> list[dict]:
    """返回 [{idx, kind: money|id, value, detail}]：idx 是客服文字消息的位置。"""
    seen_text, seen_nums, out = "", set(), []
    for i, m in enumerate(messages):
        role, content = m.get("role"), m.get("content") or ""
        if role in ("tool", "user"):
            seen_text += "\n" + content
            seen_nums |= _nums(content)
            continue
        if role != "assistant" or m.get("tool_calls") or m.get("greeting") or i == 0:
            continue
        derived = None
        for mm in MONEY.finditer(content):
            v = f"{float(mm.group(1).replace(',', '')):.2f}"
            if v not in seen_nums and derived is None:   # 客服自己算的差价、合计：两两之和 / 差也算有依据（算错了才标）
                xs = sorted({float(x) for x in seen_nums if float(x) < 1e5})
                derived = {f"{abs(a - b):.2f}" for a in xs for b in xs} | {f"{a + b:.2f}" for a in xs for b in xs}
            if v not in seen_nums and v not in derived:
                out.append({"idx": i, "kind": "money", "value": v, "detail": f"${v} not in any earlier tool output or user message"})
        for p in IDS:
            for mm in p.finditer(content):
                v = mm.group(0)
                if re.fullmatch(r"#?W?0+", v) or len(set(v.lstrip("#W"))) == 1:   # 举例用的占位号（#W0000000）
                    continue
                if v not in seen_text and v.lstrip("#") not in seen_text:
                    out.append({"idx": i, "kind": "id", "value": v, "detail": f"{v} not in any earlier tool output or user message"})
    return out
