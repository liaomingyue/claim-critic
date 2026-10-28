from data.split_claims import extract_refs, split_claims, split_sentences, strip_citations


def test_basic_split_and_drop_uncited():
    text = ("This review summarizes recent evidence. Metformin lowered HbA1c in adults [ref-1]. "
            "Statins reduced LDL by 40% [ref-2, ref-3].")
    claims = split_claims(text)
    assert [c["ref_ids"] for c in claims] == [["ref-1"], ["ref-2", "ref-3"]]
    assert claims[0]["claim"] == "Metformin lowered HbA1c in adults."


def test_decimal_and_abbreviation_not_split():
    text = "Smith et al. reported a 1.5-fold increase (p<0.05) in risk [ref-1]."
    assert len(split_sentences(text)) == 1


def test_leading_citation_goes_to_previous_sentence():
    text = "Mortality fell by 12% in the treatment arm. [ref-2] Adverse events were rare [ref-1]."
    claims = split_claims(text)
    assert [c["ref_ids"] for c in claims] == [["ref-2"], ["ref-1"]]


def test_short_fragment_merged():
    text = "Blood pressure dropped by 8 mmHg in elderly patients [ref-1]. However, not in all. Renal function was stable [ref-2]."
    claims = split_claims(text)
    assert len(claims) == 2
    assert "not in all" in claims[0]["claim"]


def test_chinese():
    text = "二甲双胍可使成年患者的糖化血红蛋白下降1.1%[ref-1]。本节小结如下。他汀类药物降低了低密度脂蛋白[ref-2]。"
    claims = split_claims(text)
    assert [c["ref_ids"] for c in claims] == [["ref-1"], ["ref-2"]]
    assert "1.1%" in claims[0]["claim"]


def test_extract_and_strip():
    s = "A was associated with B [ref-1，ref-4] in cohort studies [ref-1]."
    assert extract_refs(s) == ["ref-1", "ref-4"]
    assert strip_citations(s) == "A was associated with B in cohort studies."
