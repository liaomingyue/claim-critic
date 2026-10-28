import random

from data.build_trainset import balance_types, split_by_group, stratified_negatives
from data.synth_negative import edit_distance, parse_synth, tokens, validate
from train.grpo import balanced_order

REFS = {"ref-1": "In adults aged 18 to 65, metformin reduced HbA1c by 1.1% over 24 weeks."}
ORIG = "In adults, metformin reduced HbA1c by 1.1% over 24 weeks."


def synth(claim="In patients, metformin reduced HbA1c by 1.1% over 24 weeks.",
          ev='ref-1: "In adults aged 18 to 65"', type_="missing_condition"):
    return parse_synth(f"<plan>t</plan><claim>{claim}</claim><evidence>{ev}</evidence>"
                       f"<type>{type_}</type><reason>r</reason>")


def test_edit_distance():
    assert edit_distance(tokens("a b c"), tokens("a x c")) == 1
    assert edit_distance(tokens("成年患者"), tokens("所有患者")) == 2


def test_validate_ok():
    assert validate(synth(), ORIG, REFS, "missing_condition") == ""


def test_validate_rejects():
    assert validate(synth(type_="wrong_attribution"), ORIG, REFS, "missing_condition") == "type mismatch"
    assert validate(synth(ev='ref-1: "adults aged 18-65 years"'), ORIG, REFS, "missing_condition") \
        == "evidence not verbatim"
    assert validate(synth(claim=ORIG), ORIG, REFS, "missing_condition").startswith("edit distance")
    long_claim = ORIG + " " + " ".join(["extra"] * 50)
    assert validate(synth(claim=long_claim), ORIG, REFS, "missing_condition").startswith("edit distance")


def _neg(i, t, r):
    return {"id": f"n{i}", "label": "hallucination", "type": t, "round": r, "source_group": f"g{i % 10}"}


def test_stratified_caps_last_round():
    by_round = {1: [_neg(i, "missing_condition", 1) for i in range(1000)],
                2: [_neg(i, "missing_condition", 2) for i in range(1000)],
                3: [_neg(i, "missing_condition", 3) for i in range(1000)]}
    out = stratified_negatives(by_round, 1000, 0.4, random.Random(0))
    assert len(out) == 1000
    assert sum(n["round"] == 3 for n in out) == 400


def test_stratified_gives_slack_when_short():
    by_round = {1: [_neg(i, "missing_condition", 1) for i in range(100)],
                2: [_neg(i, "missing_condition", 2) for i in range(1000)]}
    out = stratified_negatives(by_round, 500, 0.4, random.Random(0))
    assert sum(n["round"] == 2 for n in out) == 200  # 上限不被突破，宁可总数不足


def test_balance_types():
    negs = [_neg(i, t, 1) for i, t in enumerate(["missing_condition"] * 50 + ["unsupported_inference"] * 30
                                                 + ["wrong_attribution"] * 20)]
    out = balance_types(negs, random.Random(0))
    assert len(out) == 60


def test_split_by_group_disjoint():
    samples = [_neg(i, "missing_condition", 1) for i in range(100)]
    train, val = split_by_group(samples, 0.2, random.Random(0))
    assert not {s["source_group"] for s in train} & {s["source_group"] for s in val}


def test_balanced_order_batches():
    pos = [{"id": f"p{i}", "label": "faithful", "type": "none"} for i in range(300)]
    negs = [_neg(i, t, 1) for t in ("missing_condition", "unsupported_inference", "wrong_attribution")
            for i in range(100)]
    order = balanced_order(pos + negs, batch_size=12)
    assert len(order) % 12 == 0 and order
    for b in range(0, len(order), 12):
        batch = order[b:b + 12]
        assert sum(s["label"] == "faithful" for s in batch) == 6
        assert {s["type"] for s in batch if s["label"] == "hallucination"} == {
            "missing_condition", "unsupported_inference", "wrong_attribution"}


def test_topic_query_matches_all_words_not_phrase():
    from data.fetch_pmc import topic_query

    q = topic_query("type 2 diabetes metformin")
    assert "TITLE_ABS:metformin" in q and '"type 2 diabetes metformin"' not in q
    assert topic_query('TITLE_ABS:"heart failure" AND TITLE_ABS:SGLT2').startswith('(TITLE_ABS:"heart failure"')
