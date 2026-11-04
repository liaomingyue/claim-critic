"""ragtruth.refuse_data 的单元测试（不调 Claude）。"""
from ragtruth.refuse_data import BATCH_SYSTEM, REWRITE_SYSTEM, drop_answer_passage, replace_question, overlap, splice, synth_rows, RM_SYSTEM, batches, build_rows, parse_batch, pick, split_of, strata, valid


def _r(src, k, refuse, task="QA"):
    return {"id": f"{src}-{k}", "source_id": src, "task_type": task, "source": f"Q {src}", "response": "Unable to answer based on given passages." if refuse else "An answer."}


def test_strata_and_pick_keep_all_high_refusal_sources():
    rows = [_r("a", k, k < 3) for k in range(6)] + [_r("b", k, k < 2) for k in range(6)] + [_r(f"c{i}", k, False) for i in range(20) for k in range(6)] + [_r("s", 0, True, "Summary")]
    by = strata(rows, {"c0"})
    assert by["a"]["refused"] == 3 and by["b"]["refused"] == 2 and "c0" not in by and "s" not in by
    p = pick(by, 6, 0)
    ids = [x["id"] for x in p]
    assert "a" in ids and "b" in ids and len(ids) == 6 and ids == sorted(ids)
    assert p == pick(by, 6, 0)


def test_valid():
    assert valid({"answerable": "partial", "evidence": "x"})["answerable"] == "partial"
    assert valid({"answerable": "maybe"}) is None and valid("x") is None


def test_build_rows_labels():
    labs = [{"source_id": "a", "answerable": "no", "refused_models": 3}, {"source_id": "b", "answerable": "partial", "refused_models": 0}]
    rows = build_rows(labs, {"a": "Q a", "b": "Q b"})
    assert [r["label"] for r in rows] == ["B", "A"] and rows[0]["prompt"][0]["content"] == RM_SYSTEM
    assert split_of("a", 0) == split_of("a", 0) and {split_of(str(i), 0) for i in range(200)} == {"train", "calib", "test"}


def test_batches_and_parse():
    items = [{"id": f"s{i}", "source": f"Q{i}", "refused_models": 0} for i in range(23)]
    bs = batches(items, {"s0", "s1"}, 10)
    assert [len(b["items"]) for b in bs] == [10, 10, 1] and bs[0]["items"][0]["id"] == "s2"
    out = '{"results": [{"id": "s2", "answerable": "no"}, {"id": "s3", "answerable": "bad"}, {"id": "zz", "answerable": "yes"}, {"id": "s4", "answerable": "yes", "evidence": "e"}]}'
    rows = parse_batch(out, bs[0]["items"])
    assert [(r["id"], r["answerable"]) for r in rows] == [("s2", "no"), ("s4", "yes")] and "source" not in rows[0]
    assert "[task <id>]" in BATCH_SYSTEM and '"results"' in BATCH_SYSTEM


def _qa(q, passage):
    return (f"Briefly answer the following question:\n{q}\nBear in mind that your response should be strictly based on the following three passages:\npassage 1:{passage}\n\n"
            "In case the passages do not contain the necessary information to answer the question, please reply with: \"Unable to answer based on given passages.\"\noutput:")


def test_splice_and_synth():
    a, b = _qa("how tall is mount everest", "Everest rises 8849 meters."), _qa("best soup recipe", "Carrots and onions simmer slowly.")
    s = splice(a, b)
    assert "how tall is mount everest" in s and "Carrots and onions" in s and "Everest rises" not in s and s.endswith("output:")
    assert overlap(a, a) > 0.3 and overlap(a, b) == 0.0
    labels = [{"source_id": f"s{i}", "answerable": "yes"} for i in range(40)]
    src = {f"s{i}": _qa(f"question topic{i} words{i}", f"passage about subject{i}") for i in range(40)}
    rows = synth_rows(labels, src, 5, 0)
    assert len(rows) == 5 and all(r["label"] == "B" and r["origin"] == "synth_mismatch" for r in rows)


def test_drop_answer_passage():
    s = _qa("how tall is everest", "Everest is in Nepal.\npassage 2:Everest rises 8849 meters above sea level.\npassage 3:Climbers use oxygen.")
    out = drop_answer_passage(s, "Everest rises 8849 meters")
    assert "8849" not in out and "passage 1:Everest is in Nepal." in out and "passage 2:Climbers use oxygen." in out
    assert "following two passages" in out and out.endswith("output:") and "passage 3" not in out
    assert drop_answer_passage(s, "Everest") is None   # 出现在两段里
    assert drop_answer_passage(s, "") is None and drop_answer_passage(s, "not in any") is None


def test_replace_question_and_rewrite_prompt():
    s = _qa("how tall is everest", "Everest rises 8849 meters.")
    out = replace_question(s, "how tall is k2")
    assert "how tall is k2" in out and "how tall is everest" not in out and "Everest rises 8849 meters." in out and out.endswith("output:")
    assert replace_question(s, "") is None and replace_question(s, "a\nb") is None
    assert "{" not in REWRITE_SYSTEM.format(examples="- q — why").split("Reply with")[0]
