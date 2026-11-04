"""ragtruth.ans_quality 的单元测试（不调 Claude）。"""
from ragtruth.ans_quality import SYSTEM, pick_pairs, summarize, valid


def _dev():
    return [{"id": f"q{i}", "task_type": "QA"} for i in range(10)] + [{"id": f"s{i}", "task_type": "Summary"} for i in range(10)]


def _samples(tag):
    return {r["id"]: {"outputs": [f"{tag}-{r['id']}", "other"]} for r in _dev()}


def test_pick_pairs_paired_blind_and_deterministic():
    a = pick_pairs(_dev(), _samples("b"), _samples("c"), 7, 8, 0)
    assert len(a) == 30 and a == pick_pairs(_dev(), _samples("b"), _samples("c"), 7, 8, 0)
    by = {}
    for x in a:
        by.setdefault(x["source_id"], set()).add(x["arm"])
    assert len(by) == 15 and all(v == {"base", "ckpt"} for v in by.values())
    assert sum(x["task_type"] == "QA" for x in a) == 14
    assert [x["arm"] for x in a[:8]] != ["base", "ckpt"] * 4   # 打乱了，不是成对相邻
    assert all("base" not in x["id"] and "ckpt" not in x["id"] for x in a)
    assert "base" not in SYSTEM and "checkpoint" not in SYSTEM


def test_valid():
    ok = {"answers_task": "yes", "misses_key_info": False, "mostly_copied_fragments": False, "score": 5, "note": "x"}
    assert valid(ok)["score"] == 5
    assert valid({**ok, "score": 6}) is None
    assert valid({**ok, "answers_task": "maybe"}) is None
    assert valid({**ok, "misses_key_info": "no"}) is None
    assert valid("x") is None


def _row(s, t, arm, score, ans="yes"):
    return {"source_id": s, "task_type": t, "arm": arm, "score": score, "answers_task": ans, "misses_key_info": score < 4, "mostly_copied_fragments": False}


def test_summarize_only_counts_complete_pairs():
    rows = [_row("a", "QA", "base", 5), _row("a", "QA", "ckpt", 3, "partly"), _row("b", "Summary", "base", 4), _row("b", "Summary", "ckpt", 4), _row("c", "QA", "base", 2)]
    r = summarize(rows)
    assert r["pairs"] == 2 and r["calls"] == 5
    assert r["all"]["worse"] == 1 and r["all"]["same"] == 1 and r["all"]["better"] == 0 and r["all"]["mean_delta"] == -1
    assert r["QA"]["score"] == {"base": 5, "ckpt": 3} and r["QA"]["answers_yes"]["ckpt"] == 0
