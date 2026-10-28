"""三项 reward：总分 = w_label × 标签 + w_format × 格式 + w_loc × 定位，各项 0/1。

- 标签：verdict 与真值一致。
- 格式：common.output_format.parse_output 判定合规。
- 定位：标签正确，且 evidence 合规（见 evidence_ok）；真值为 hallucination 时还要求 type 一致。
  定位分以标签正确为前提，否则「判错但贴了一段原文」也能拿 0.3，等于奖励错误答案。

evidence 合规按真值区分：
  citation_error  真值证据为 none（无中生有）时必须输出 none；否则逐字可定位，且必须出自未被引用的文献
                  （内容真正的出处）。要看真值，否则模型一律写 none 就能绕开「找出处」；
  faithful        逐字可定位，claim 有引用时必须出自被引用的文献；
  其余错误类型    逐字可定位。

纯函数，无 IO，verl 和 TRL 都直接调用；单测见 tests/test_reward.py。
"""
from __future__ import annotations

import json

from common.output_format import ParsedOutput, evidence_locatable, parse_output

DEFAULT_WEIGHTS = {"label": 1.0, "format": 0.3, "loc": 0.3}


def evidence_ok(p: ParsedOutput, label: str, type_: str, refs: dict[str, str], cited: list[str] | None,
                min_chars: int = 8, gold_unsourced: bool | None = None) -> bool:
    """gold_unsourced：真值是否为「无中生有」。None 表示不知道真值（推理时检查预测自身），两种写法都接受。"""
    if label == "hallucination" and type_ == "citation_error":
        if gold_unsourced is None and p.evidence_none:
            return True
        if gold_unsourced:
            return p.evidence_none
    if not evidence_locatable(p.evidence_ref, p.evidence_text, refs, min_chars):
        return False
    cited = list(refs) if cited is None else cited
    if label == "hallucination" and type_ == "citation_error":
        return p.evidence_ref not in cited
    if label == "faithful" and cited:
        return p.evidence_ref in cited
    return True


def score_components(output: str, label: str, type_: str, refs: dict[str, str], cited: list[str] | None = None,
                     gold_evidence: str = "", min_evidence_chars: int = 8) -> dict[str, float]:
    p = parse_output(output)
    label_ok = p.verdict == label
    fmt_ok = p.format_ok
    unsourced = gold_evidence.strip().lower() == "none"
    loc_ok = label_ok and evidence_ok(p, label, type_, refs, cited, min_evidence_chars, unsourced)
    if label == "hallucination":
        loc_ok = loc_ok and p.type == type_
    return {
        "label": float(label_ok),
        "format": float(fmt_ok),
        "loc": float(loc_ok),
        # 以下只用于监控（5.6 节），不进总分
        "pred_hallucination": float(p.verdict == "hallucination"),
        "parse_error": p.error,
    }


def total_reward(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return sum(w[k] * components[k] for k in ("label", "format", "loc"))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl 的 custom_reward_function 入口。ground_truth 为 JSON 字符串：{label, type, refs, cited, evidence}。"""
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    comp = score_components(solution_str, gt["label"], gt["type"], gt["refs"], gt.get("cited"), gt.get("evidence", ""))
    score = total_reward(comp)
    # 返回 dict 时 verl 会把额外字段记进 reward_extra_info，用于按类型统计
    return {
        "score": score,
        "label_acc": comp["label"],
        "format_ok": comp["format"],
        "loc_ok": comp["loc"],
        "pred_hallucination": comp["pred_hallucination"],
    }


def trl_reward(completions, label, type, refs, cited=None, evidence=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer 的 reward_funcs 入口。数据集需含 label、type、refs、cited、evidence 列（refs、cited 为 JSON 字符串）。"""
    rewards = []
    cited = cited if cited is not None else [None] * len(completions)
    evidence = evidence if evidence is not None else [""] * len(completions)
    for c, lb, tp, rf, ct, ev in zip(completions, label, type, refs, cited, evidence):
        text = c[-1]["content"] if isinstance(c, list) else c
        rf = json.loads(rf) if isinstance(rf, str) else rf
        ct = json.loads(ct) if isinstance(ct, str) else ct
        rewards.append(total_reward(score_components(text, lb, tp, rf, ct, ev or "")))
    return rewards
