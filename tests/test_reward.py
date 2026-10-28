import json

import pytest

from common.output_format import format_output
from train.reward import compute_score, score_components, total_reward

REFS = {
    "ref-1": "In adults aged 18 to 65, metformin reduced HbA1c by 1.1% over 24 weeks.",
    "ref-2": "The placebo group showed no significant change in fasting glucose.",
}
EV = "metformin reduced HbA1c by 1.1%"


def out(verdict="hallucination", ref="ref-1", ev=EV, type_="missing_condition", reason="丢了年龄限定"):
    return format_output(verdict, ref, ev, type_, reason, think="核对年龄范围")


def test_all_correct_hallucination():
    c = score_components(out(), "hallucination", "missing_condition", REFS)
    assert (c["label"], c["format"], c["loc"]) == (1, 1, 1)
    assert total_reward(c) == pytest.approx(1.6)


def test_all_correct_faithful():
    c = score_components(out("faithful", type_="none", reason=""), "faithful", "none", REFS)
    assert (c["label"], c["format"], c["loc"]) == (1, 1, 1)


def test_wrong_type_loses_loc_only():
    c = score_components(out(type_="wrong_attribution"), "hallucination", "missing_condition", REFS)
    assert (c["label"], c["format"], c["loc"]) == (1, 1, 0)


def test_wrong_label_gets_no_loc_even_with_valid_evidence():
    c = score_components(out(), "faithful", "none", REFS)
    assert (c["label"], c["loc"]) == (0, 0)
    assert total_reward(c) == pytest.approx(0.3)


def test_evidence_not_verbatim():
    c = score_components(out(ev="metformin lowered HbA1c by 1.1%"), "hallucination", "missing_condition", REFS)
    assert c["loc"] == 0


def test_evidence_in_wrong_ref():
    c = score_components(out(ref="ref-2"), "hallucination", "missing_condition", REFS)
    assert c["loc"] == 0


def test_evidence_too_short_is_rejected():
    c = score_components(out(ev="HbA1c"), "hallucination", "missing_condition", REFS)
    assert c["loc"] == 0


def test_evidence_whitespace_and_quotes_normalized():
    c = score_components(out(ev="metformin  reduced\nHbA1c by 1.1%"), "hallucination", "missing_condition", REFS)
    assert c["loc"] == 1


def test_missing_tag_breaks_format_but_label_counts():
    text = "<think>x</think>\n<verdict>hallucination</verdict>\n<type>missing_condition</type>"
    c = score_components(text, "hallucination", "missing_condition", REFS)
    assert (c["label"], c["format"], c["loc"]) == (1, 0, 0)


def test_wrong_order_breaks_format():
    text = ('<verdict>faithful</verdict>\n<think>x</think>\n'
            f'<evidence>ref-1: "{EV}"</evidence>\n<type>none</type>')
    assert score_components(text, "faithful", "none", REFS)["format"] == 0


def test_empty_think_breaks_format():
    text = out().replace("<think>核对年龄范围</think>", "<think></think>")
    assert score_components(text, "hallucination", "missing_condition", REFS)["format"] == 0


def test_hallucination_requires_reason_faithful_does_not():
    no_reason = out().split("\n<reason>")[0]
    assert score_components(no_reason, "hallucination", "missing_condition", REFS)["format"] == 0
    faithful = out("faithful", type_="none", reason="")
    assert "<reason>" not in faithful
    assert score_components(faithful, "faithful", "none", REFS)["format"] == 1


def test_verdict_type_mismatch_breaks_format():
    assert score_components(out(type_="none"), "hallucination", "missing_condition", REFS)["format"] == 0


def test_garbage_output():
    c = score_components("I think it is fine.", "faithful", "none", REFS)
    assert total_reward(c) == 0


def test_verl_entry():
    gt = json.dumps({"label": "hallucination", "type": "missing_condition", "refs": REFS})
    res = compute_score("claim_critic", out(), gt)
    assert res["score"] == pytest.approx(1.6)
    assert res["pred_hallucination"] == 1
