import random

from data.build_groups import intent_core, parse_questions, split_groups
from data.gen_review import task_text
from data.intents import INTENTS, question_prompt


def test_parse_questions_extracts_json_and_drops_bad_items():
    text = 'Here:\n[{"question": "Q1", "keywords": ["metformin", "HbA1c"]}, {"question": "", "keywords": ["x"]},' \
           ' {"question": "Q3", "keywords": []}, {"question": "Q4", "keywords": ["apixaban"], "drug": "apixaban"}]'
    qs = parse_questions(text)
    assert [q["question"] for q in qs] == ["Q1", "Q4"] and qs[1]["drug"] == "apixaban"
    assert parse_questions("no json") == []


def test_intent_core_filters_years_not_types():
    core = intent_core(["metformin", "kidney disease"], INTENTS["frontier"], 2026)
    assert 'TITLE_ABS:metformin AND TITLE_ABS:"kidney disease"' in core
    assert "PUB_TYPE" not in core and "PUB_YEAR:[2024 TO 2026]" in core
    assert "PUB_YEAR" not in intent_core(["x", "y"], INTENTS["overview"], 2026)


def test_medication_prompt_asks_for_drug():
    assert '"drug"' in question_prompt("t", "medication", 3) and '"drug"' not in question_prompt("t", "frontier", 3)


def group(gid, ids):
    return {"group_id": gid, "papers": [{"pmcid": i} for i in ids], "refs": {f"ref-{k + 1}": i for k, i in enumerate(ids)}}


def test_split_groups_removes_test_papers_from_train():
    groups = [group(f"g{i}", ids) for i, ids in enumerate([["A", "B", "C"], ["A", "D", "E", "F"], ["G", "H", "I"]])]
    out = split_groups(groups, 0.34, random.Random(0), min_group=3)
    test = [g for g in out if g["split"] == "test"]
    test_ids = {p["pmcid"] for g in test for p in g["papers"]}
    for g in out:
        if g["split"] == "train":
            assert not test_ids & {p["pmcid"] for p in g["papers"]}
            assert list(g["refs"]) == [f"ref-{k + 1}" for k in range(len(g["papers"]))]
            assert list(g["refs"].values()) == [p["pmcid"] for p in g["papers"]]


def test_task_text_only_for_intent_groups():
    assert task_text({"refs": {}}) == ("", "")
    task, rule = task_text({"question": "Q?", "intent": "overview"})
    assert "Q?" in task and INTENTS["overview"]["writing_rule"] in rule


def test_parse_results_dedupes_and_skips_retracted():
    from data.fetch_pmc import parse_results

    def r(pmcid, title, lic="cc by"):
        return {"pmcid": pmcid, "title": title, "license": lic, "abstractText": "a", "pubYear": "2025"}

    seen = set()
    rows = [r("PMC1", "Metformin trial."), r("PMC1", "Metformin trial."), r("PMC2", "Metformin  Trial"),
            r("PMC3", "RETRACTED: bad"), r("PMC4", "Other", "cc by-nc"), r("PMC5", "New study")]
    assert [p["pmcid"] for p in parse_results(rows, seen)] == ["PMC1", "PMC5"]
    assert parse_results([r("PMC5", "New study")], seen) == []  # 跨页也去重


def test_study_type_rules():
    from data.study_type import classify

    rct = "Patients with type 2 diabetes were randomized to metformin or placebo; HbA1c fell by 1.1%."
    assert classify("Metformin versus placebo in adults with T2D", rct) == "clinical"
    assert classify("Metformin and cancer: a systematic review and meta-analysis", "") == "meta"
    assert classify("Cellular and Molecular Mechanisms of Metformin Action", "This review summarizes",
                    ["review-article"]) == "review"
    assert classify("Metformin attenuates fibrosis in diabetic mice", "Mice were fed a high-fat diet.") == "animal"
    assert classify("Bullous pemphigoid with a DPP-4 inhibitor: a case report", "") == "case_report"
    assert classify("Rationale and design of the XYZ trial", "Patients will be randomized") == "protocol"
    assert classify("Effects of metformin in HepG2 cell lines", "") == "in_vitro"
    assert classify("Editorial: diabetes care in 2026", "") == "other"


def test_clean_text_unescapes_and_strips_tags():
    from data.fetch_pmc import clean_text

    assert clean_text("Oral Sucrosomial&lt;sup&gt;®&lt;/sup&gt; B<sub>12</sub>  levels") == "Oral Sucrosomial® B12 levels"


def test_strip_trailer_removes_appended_notes():
    from data.gen_review import strip_trailer

    body = "二甲双胍是首选药 [ref-1]。"
    assert strip_trailer(body + "\n\n---\n\n说明：大纲第3部分没写。") == body
    assert strip_trailer(body + "\n说明：参考文献未提及。") == body
    assert strip_trailer(body) == body


def test_relevance_prompt_and_selection_parsing():
    from data.build_groups import parse_selection, relevance_prompt, title_hit

    prompt = relevance_prompt("Q?", "frontier", [{"title": "T1", "year": "2025", "abstract": "A" * 900}])
    assert "[1] T1 (2025)" in prompt and "A" * 601 not in prompt and "前沿研究" in prompt
    assert parse_selection("I pick [3, 1, 3, 9]", 5) == [2, 0]
    assert parse_selection("none", 5) == []
    assert title_hit({"title": "Apixaban in Subclinical Atrial Fibrillation"}, ["atrial fibrillation", "x"])
    assert not title_hit({"title": "Hypothyroidism in hemodialysis"}, ["atrial fibrillation", "subclinical"])
