from ragtruth.annotate import check, truth_text
from ragtruth.convert import coarse_sentences, label_sentences, split_sentences
from ragtruth.metrics import aggregate, per_sample
from ragtruth.task import target_text as target_text_v
from ragtruth.task import (TIPS, balance, evidence_in_source, expand_claims, format_output, messages, parse_output,
                           reason_consistent, render_input, score_components, sentence_target_ok, total_reward)

SOURCE = "Passage 1: The shop opens at 9 am and closes at 5 pm. It sells fresh bread and cakes every day."


def _texts(t):
    return [t[a:b] for a, b in split_sentences(t)]


def test_split_basic_and_abbreviations():
    t = "Dr. Smith arrived at 3.5 p.m. on Monday. He met Mr. J. Brown! Was it fun? Yes."
    assert _texts(t) == ["Dr. Smith arrived at 3.5 p.m. on Monday.", "He met Mr. J. Brown!", "Was it fun?", "Yes."]


def test_split_newlines_and_list_markers():
    t = "Here is a summary:\n1. The shop opens at 9.\n2. It sells bread.\n\nOverall good."
    assert _texts(t) == ["Here is a summary:", "1. The shop opens at 9.", "2. It sells bread.", "Overall good."]


def test_split_year_end_of_sentence():
    assert _texts("It opened in 2019. Sales grew.") == ["It opened in 2019.", "Sales grew."]


def test_label_sentences_overlap_and_kind():
    t = "A is red. B is blue. C is green."
    spans = split_sentences(t)
    labels = [{"start": t.index("B is"), "end": t.index("B is") + 9, "label_type": "Evident Conflict"},
              {"start": t.index("green"), "end": t.index("green") + 5, "label_type": "Subtle Baseless Info"},
              {"start": t.index(". B"), "end": t.index(". B") + 2, "label_type": "Evident Baseless Info"}]
    assert label_sentences(spans, labels) == {2: "conflict", 3: "baseless"}


def test_evidence_exact_and_ellipsis():
    assert evidence_in_source("opens at 9 am and closes", SOURCE)
    assert evidence_in_source("The shop opens at ... and cakes every day", SOURCE)
    assert evidence_in_source("the shop opens at 9 AM", SOURCE)          # 大小写不敏感
    assert not evidence_in_source("and cakes every day ... The shop opens at", SOURCE)   # 顺序反了
    assert not evidence_in_source("opens at 10 am", SOURCE)
    assert not evidence_in_source("9 am ... cakes", SOURCE)               # 首末段太短


def test_format_roundtrip():
    p = parse_output(format_output("closes at 5 pm", "The source says it closes at 5 pm; the sentence says 6 pm.",
                                   "conflict"))
    assert p.format_ok and p.verdict == "hallucination" and p.type == "conflict" and p.evidence == "closes at 5 pm"
    p = parse_output(format_output(None, "No factual content.", "none"))
    assert p.format_ok and p.verdict == "faithful" and p.evidence_none


def test_parse_rejects_bad_order_and_mismatch():
    assert not parse_output("<reason>x</reason><evidence>none</evidence><type>none</type>"
                            "<verdict>faithful</verdict>").format_ok
    assert not parse_output('<evidence>"abc defgh"</evidence><reason>x</reason><type>none</type>'
                            "<verdict>hallucination</verdict>").format_ok


def _resp(bad, n=3, analysis=True):
    sents = [f"Sentence {i}." for i in range(1, n + 1)]
    s = {"id": "r1", "source": SOURCE, "sentences": sents, "task_type": "QA", "model": "m", "split": "train",
         "bad": {str(k): v for k, v in bad.items()}, "label": "hallucination" if bad else "faithful"}
    if analysis:
        s["analysis"] = [{"evidence": "opens at 9 am", "reason": f"r{i}"} for i in range(1, n + 1)]
        s["analysis_ok"] = True
    return s


def test_expand_and_render():
    claims = expand_claims(_resp({2: "conflict"}))
    assert [c["label"] for c in claims] == ["faithful", "hallucination", "faithful"]
    assert claims[1]["type"] == "conflict" and claims[1]["id"] == "r1-S2" and claims[1]["evidence"] == "opens at 9 am"
    assert render_input(claims[1]).endswith("[sentence to check]\nS2: Sentence 2.")
    assert expand_claims(_resp({}, analysis=False), require_analysis=True) == []


def test_balance_keeps_all_bad():
    claims = expand_claims(_resp({1: "baseless"}, n=10))
    out = balance(claims, 2.0)
    assert sum(c["label"] == "hallucination" for c in out) == 1 and len(out) == 3


def test_reward_components():
    c = expand_claims(_resp({2: "conflict"}))[1]
    perfect = total_reward(score_components(format_output("closes at 5 pm", "x", "conflict"), c))
    assert perfect == 1.8
    no_ev = score_components(format_output(None, "x", "conflict"), c)      # conflict 必须给片段
    assert no_ev["label"] == 1 and no_ev["type"] == 1 and no_ev["evidence"] == 0
    wrong_type = score_components(format_output("closes at 5 pm", "x", "baseless"), c)
    assert wrong_type["label"] == 1 and wrong_type["type"] == 0
    fabricated = score_components(format_output("closes at 6 pm sharp", "x", "conflict"), c)
    assert fabricated["evidence"] == 0


def test_metrics_aggregate_to_response():
    c = expand_claims(_resp({2: "conflict"}))
    outs = [format_output(None, "x", "none"), format_output("closes at 5 pm", "x", "conflict"),
            format_output(None, "x", "none")]
    res = aggregate([per_sample(g, o) for g, o in zip(c, outs)])
    assert res["sentence"]["f1"] == 1 and res["response"]["f1"] == 1 and res["type_accuracy"] == 1
    assert res["conflict_evidence_rate"] == 1
    res = aggregate([per_sample(g, "garbage") for g in c])
    assert res["format_rate"] == 0 and res["sentence"]["accuracy"] == 0


def test_annotate_check():
    s = {**_resp({2: "conflict"}, analysis=False), "response": "Sentence 1. Sentence 2. Sentence 3.",
         "sent_spans": [(0, 11), (12, 23), (24, 35)], "spans": [[12, 23, "Evident Conflict"]],
         "span_notes": [[12, 23, "Source says 5 pm"]]}
    assert "S2 contradicts the source" in truth_text(s) and "Source says 5 pm" in truth_text(s)
    good = [{"id": "S1", "evidence": "opens at 9 am", "reason": "Matches the opening time."},
            {"id": "S2", "evidence": "closes at 5 pm", "reason": "The source says 5 pm, but the sentence says 6 pm."},
            {"id": "S3", "evidence": None, "reason": "No factual content."}]
    assert check(good, s) == []
    bad = [dict(good[0]), {"id": "S2", "evidence": None, "reason": "This is a conflict."}, dict(good[2])]
    problems = check(bad, s)
    assert any("not null" in p for p in problems) and any("must not use" in p for p in problems)
    assert check(good[:2], s) and check(None, s)


def test_coarse_sentences():
    t = "A one. B two. C three. D four."
    spans = split_sentences(t)
    big = {"start": 0, "end": t.index("C three.") + 8, "label_type": "Evident Baseless Info"}
    small = {"start": t.index("D four"), "end": t.index("D four") + 6, "label_type": "Evident Conflict"}
    bad = label_sentences(spans, [big, small])
    assert coarse_sentences(spans, [big, small], bad) == [1, 2, 3]


def test_reason_consistency():
    assert reason_consistent("The source lists the opening hours, which match the sentence.", "none")
    assert not reason_consistent("The source does not mention free WiFi.", "none")
    assert reason_consistent("The source does not mention free WiFi.", "baseless")
    assert reason_consistent("The source says 5 pm, but the sentence says 6 pm.", "conflict")
    assert not reason_consistent("The source confirms the hours.", "conflict")
    assert not sentence_target_ok("opens at 9 am", "The source does not mention cakes.", "none", SOURCE)
    # 寒暄、复述任务的句子：「没有事实内容」不算说原文没有
    assert reason_consistent("This sentence is an introductory statement and does not contain factual claims.", "none")
    assert reason_consistent("The sentence contains no specific factual claims to verify.", "none")
    assert reason_consistent("The sentence accurately reflects these items without adding unsupported details.", "none")
    assert reason_consistent("The source lists iOS but does not explicitly mention the App Store.", "baseless")


def test_expand_skips_coarse_and_disputed():
    s = _resp({1: "baseless", 2: "baseless"}, n=3)
    s["coarse"] = [1]
    s["analysis"][1]["disputed"] = True
    s["analysis"] = [{"evidence": "opens at 9 am", "reason": "The source does not mention it.", "disputed": False},
                     {"evidence": "opens at 9 am", "reason": "The source states the opening time.", "disputed": True},
                     {"evidence": "opens at 9 am", "reason": "The source states the opening time."}]
    claims = expand_claims(s, require_analysis=True)
    assert [c["idx"] for c in claims] == [3]          # S1 粗粒度、S2 存疑且说明与真值相反
    s["analysis"][1]["reason"] = "The source does not mention it."    # 标了存疑但说明与真值一致：照常保留
    assert [c["idx"] for c in expand_claims(s, require_analysis=True)] == [2, 3]
    assert [c["coarse"] for c in expand_claims(s)] == [True, False, False]


def test_tips_only_when_asked():
    c = expand_claims(_resp({}))[0]
    assert TIPS not in messages(c)[0]["content"] and TIPS in messages(c, tips=True)[0]["content"]
    assert messages(c, tips=True)[0]["content"].rstrip().endswith("<verdict>hallucination or faithful</verdict>")


def test_balance_natural_ratio():
    claims = expand_claims(_resp({1: "baseless"}, n=10))
    assert len(balance(claims, None)) == 10


def test_metrics_exclude_coarse_from_sentence_level():
    s = _resp({1: "baseless", 2: "conflict"})
    s["coarse"] = [1]
    c = expand_claims(s)
    outs = [format_output(None, "x", "none"), format_output("closes at 5 pm", "x", "conflict"),
            format_output(None, "x", "none")]
    res = aggregate([per_sample(g, o) for g, o in zip(c, outs)])
    assert res["sentence"]["recall"] == 1 and res["n_coarse_excluded"] == 1 and res["response"]["recall"] == 1


def test_verdict_style_roundtrip():
    c = expand_claims(_resp({2: "conflict"}))[1]
    t = target_text_v(c, "verdict")
    p = parse_output(t, "verdict")
    assert p.format_ok and p.verdict == "hallucination" and p.type == "conflict"
    assert "evidence" not in messages(c, style="verdict")[0]["content"].split("Output exactly")[1]
    assert not parse_output(t, "full").format_ok


def test_vfirst_style_roundtrip():
    c = {**expand_claims(_resp({2: "conflict"}))[1], "evidence": "opens at 9 am", "reason": "The source says 9 am."}
    t = target_text_v(c, "vfirst")
    assert t.startswith("<type>conflict</type>\n<verdict>hallucination</verdict>\n<evidence>")
    p = parse_output(t, "vfirst")
    assert p.format_ok and p.verdict == "hallucination" and p.evidence == "opens at 9 am"
    assert not parse_output(t, "full").format_ok and not parse_output(target_text_v(c, "full"), "vfirst").format_ok
    assert messages(c, style="vfirst")[0]["content"].split("Output exactly")[1].split("\n")[1].startswith("<type>")


def test_resp_styles_roundtrip_and_rows():
    from ragtruth import resp as rr
    s = _resp({2: "conflict"})
    s["analysis"] = [{"evidence": "opens at 9 am"}, {"evidence": "closes at 5 pm"}, {"evidence": None}]
    for style in rr.STYLES:
        out = rr.target_text(s, style)
        p = rr.parse(out, 3, style)
        assert p["format_ok"] and p["verdict"] == "hallucination" and p["bad"] == {2: "conflict"}
        rows = rr.sentence_rows(s, out, style)
        assert [r["pred"] for r in rows] == ["faithful", "hallucination", "faithful"]
    assert not rr.parse(rr.target_text(s, "plain"), 3, "evidence")["format_ok"]   # 缺 analysis
    assert rr.target_text(_resp({}), "plain").endswith("<errors>none</errors>\n<verdict>faithful</verdict>")
