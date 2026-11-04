"""ragtruth.ans_accept 的单元测试（不调 Claude、不调模型）。"""
from ragtruth.ans_accept import answer_flags, compare, quality_items


def _resp(prefix, src, k, task="QA"):
    return {"id": f"{prefix}-{src}-{k}", "task_type": task, "response": f"{prefix} answer", "source": f"source {src}"}


def test_quality_items_paired_blind_and_capped():
    base = [_resp("b9", f"s{i}", 0) for i in range(10)]
    arm = [_resp("ag", f"s{i}", 0) for i in range(10)]
    a = quality_items(base, arm, 4, 0)
    assert len(a) == 8 and a == quality_items(base, arm, 4, 0)
    by = {}
    for x in a:
        by.setdefault(x["pair"], {})[x["arm"]] = x
    assert len(by) == 4 and all(set(v) == {"base", "ckpt"} for v in by.values())
    assert all(v["base"]["answer"] == "b9 answer" and v["ckpt"]["answer"] == "ag answer" for v in by.values())
    assert all("b9" not in x["id"] and "ag" not in x["id"] for x in a)
    assert len(quality_items(base, arm, 100, 0)) == 20


def _sent(prefix, src, k, err=False, sev="none", task="QA"):
    return {"resp_id": f"{prefix}-{src}-{k}", "source_id": src, "task_type": task, "is_error": err, "severity": sev}


def test_answer_flags_and_compare_pairs_by_source():
    base = [_sent("b9", "a", 0, True, "clear"), _sent("b9", "a", 0), _sent("b9", "b", 0, True, "minor"), _sent("b9", "b", 0)]
    arm = [_sent("ag", "a", 0), _sent("ag", "a", 0), _sent("ag", "b", 0, True, "minor"), _sent("ag", "b", 0)]
    fb, fa = answer_flags(base), answer_flags(arm)
    assert fb["a-0"]["any_clear"] == 1.0 and fa["a-0"]["any_error"] == 0.0 and fb["b-0"]["err_rate"] == 0.5
    r = compare(fb, fa, {"a-0": "a", "b-0": "b"}, n=200)
    assert r["answers"] == 2 and r["sources"] == 2
    assert r["any_error"]["delta"] == -0.5 and r["any_clear"]["base"] == 0.5 and r["any_clear"]["ckpt"] == 0.0
    assert r["any_clear"]["ci"][1] <= 0.0
