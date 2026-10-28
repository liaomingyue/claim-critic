import pytest

from annotation.kappa import agreement, cohen_kappa, disagreements, merge
from common.output_format import format_output
from eval.metrics import aggregate, bootstrap_f1, compare, f1_halluc, per_sample

REFS = {"ref-1": "In adults aged 18 to 65, metformin reduced HbA1c by 1.1% over 24 weeks."}


def gold(i, label, type_="none", ev='ref-1: "metformin reduced HbA1c by 1.1%"'):
    return {"id": f"s{i}", "refs": REFS, "claim": "c", "label": label, "type": type_, "evidence": ev,
            "split": "test", "source_group": "g", "reason": ""}


def pred(verdict, type_="none", ev="metformin reduced HbA1c"):
    return format_output(verdict, "ref-1", ev, type_, "r" if verdict == "hallucination" else "", think="t")


def test_unparseable_counts_as_wrong():
    r = per_sample(gold(0, "hallucination", "missing_condition"), "garbage")
    assert r["pred"] == "faithful" and not r["format_ok"]
    r = per_sample(gold(1, "faithful"), "garbage")
    assert r["pred"] == "hallucination"


def test_f1_and_type_recall():
    rows = [
        per_sample(gold(0, "hallucination", "missing_condition"), pred("hallucination", "missing_condition")),
        per_sample(gold(1, "hallucination", "wrong_attribution"), pred("faithful")),
        per_sample(gold(2, "faithful"), pred("hallucination", "missing_condition")),
        per_sample(gold(3, "faithful"), pred("faithful")),
    ]
    res = aggregate(rows)
    assert res["precision"] == pytest.approx(0.5) and res["recall"] == pytest.approx(0.5)
    assert res["recall_missing_condition"] == 1.0 and res["recall_wrong_attribution"] == 0.0
    assert res["recall_unsupported_inference"] is None
    assert res["loc_acc"] == 1.0  # 两条判对的样本证据都与人工证据重叠
    lo, hi = bootstrap_f1(rows, n_boot=200)
    assert 0 <= lo <= hi <= 1


def test_loc_overlap():
    r = per_sample(gold(0, "faithful"), pred("faithful", ev="In adults aged 18 to 65"))
    assert r["loc_ok"] is False


def test_compare():
    assert compare({"f1": 0.8}, {"f1": 0.5}) == {"abs": pytest.approx(0.3), "rel": pytest.approx(0.6)}


def test_kappa_known_value():
    # 经典例：po = 0.7, pe = 0.5 -> kappa = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    sklearn = pytest.importorskip("sklearn.metrics")
    c = ["y", "n", "n", "y", "y", "n", "y"]
    d = ["y", "y", "n", "y", "n", "n", "n"]
    assert cohen_kappa(c, d) == pytest.approx(sklearn.cohen_kappa_score(c, d))
    assert cohen_kappa(["y", "n"], ["y", "n"]) == 1.0


def test_agreement_disagreement_merge():
    a = {"s0": gold(0, "faithful"), "s1": gold(1, "hallucination", "missing_condition")}
    b = {"s0": gold(0, "faithful"), "s1": gold(1, "hallucination", "wrong_attribution")}
    res = agreement(a, b)
    assert res["kappa_verdict"] == 1.0
    assert [d["id"] for d in disagreements(a, b)] == ["s1"]
    with pytest.raises(KeyError):
        merge(a, b, {})
    final = merge(a, b, {"s1": {"label": "hallucination", "type": "wrong_attribution"}})
    assert final[1]["type"] == "wrong_attribution"


def test_f1_empty_safe():
    assert f1_halluc([])["f1"] == 0.0
