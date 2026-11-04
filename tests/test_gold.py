"""ragtruth.gold 的单元测试（不调 Claude）：输出校验、多数裁决、选回答。"""
from ragtruth.gold import coverage, looks_like, pick_pilot, resolve, valid


def s(idx=1, err=True, mech="number_or_date", sev="clear", ty="conflict"):
    return {"idx": idx, "is_error": err, "type": ty if err else "none", "mechanism": mech if err else "none", "severity": sev if err else "none", "evidence": "", "note": ""}


def test_valid_rejects_bad_structure():
    assert valid({"sentences": [s(1), s(2, err=False)]}, 2)
    assert valid({"sentences": [s(1)]}, 2) is None
    assert valid({"sentences": [s(2)]}, 1) is None
    assert valid({"sentences": [{**s(1), "severity": "none"}]}, 1) is None   # 判错却没有严重程度
    assert valid({"sentences": [{**s(1), "mechanism": "made_up"}]}, 1) is None
    assert valid("oops", 1) is None


def test_resolve_majority():
    a, b = [s(1), s(2, err=False)], [s(1, mech="other", sev="minor"), s(2)]
    two = resolve([a, [s(1), s(2, err=False)]])
    assert [x["agreed"] for x in two] == [True, True]
    three = resolve([a, b, [s(1), s(2)]])
    assert three[1]["is_error"] is True and three[1]["agreed"] is False
    assert three[0]["mechanism"] == "number_or_date"   # 一致时用第一遍


def test_pick_pilot_one_response_per_source():
    rows = [{"id": f"r{k}", "task_type": ("QA", "Summary")[k % 2], "model": f"m{k % 3}", "source_id": str(k), "sentences": ["a"]} for k in range(60)]
    picks = pick_pilot(rows, {"0", "1"}, 6, 0)
    assert len({r["source_id"] for r in picks}) == 6 and not {"0", "1"} & {r["source_id"] for r in picks}
    assert picks == pick_pilot(rows, {"0", "1"}, 6, 0)


def test_coverage_and_features():
    cov = coverage([s(1), s(2, mech="vague_evaluation_or_addon"), s(3, err=False)])
    assert cov["number"] == 1 and cov["adds"] == 1 and cov["scope"] == 0
    r = {"sentences": ["It cost 5 dollars in 2020.", "All of them left."], "source": "x", "task_type": "Summary"}
    assert looks_like(r, "number") == 2 and looks_like(r, "scope") == 1


def test_base_eval_pick_random():
    from ragtruth.base_eval import pick_random, to_answer
    src = [{"source_id": str(i), "task_type": "QA", "source": "s"} for i in range(3)]
    rows = [to_answer(s, k, "One sentence. Two sentence.", "stop") for s in src for k in range(4)]
    rows.append(to_answer(src[0], 9, "Cut off", "length"))
    a = pick_random(rows, 5, 0)
    assert len(a) == 5 and {r["source_id"] for r in a} == {"0", "1", "2"} and all(r["finish"] == "stop" for r in a)
    assert a == pick_random(list(reversed(rows)), 5, 0)
    assert a[0]["sentences"] and a[0]["bad"] == {}


def test_pool_sources_balanced_and_exclude():
    from ragtruth.base_eval import pick_pool_sources
    rows = [{"id": f"{t}{i}-{m}", "source_id": f"{t}{i}", "task_type": t} for t in ("QA", "Summary", "Data2txt") for i in range(10) for m in "ab"]
    a = pick_pool_sources(rows, {"QA0"}, 6, 0)
    assert [r["task_type"] for r in a] == ["QA"] * 3 + ["Summary"] * 3
    assert len({r["source_id"] for r in a}) == 6 and "QA0" not in {r["source_id"] for r in a}
    assert a == pick_pool_sources(list(reversed(rows)), {"QA0"}, 6, 0)


def test_holdout_metrics():
    from ragtruth.base_eval import holdout_metrics
    ys = [1, 1, 0, 0]
    m = holdout_metrics(ys, [.9, .1, .2, .05], [.9, .8, .2, .05])
    assert m["auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0 and m["flagged"] == 2
    assert m["auc_delta"] > 0 and m["same_flag_delta"] == 0.5
