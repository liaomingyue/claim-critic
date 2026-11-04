"""ragtruth.sat_data 的单元测试（不调 Claude）。"""
from ragtruth.sat_data import pick_pairs, repeat_agreement, summarize, to_items, valid


def _a(step, chars=100, copy=0.1, text=None):
    return {"step": step, "text": text or f"answer {step} {chars} {copy}", "chars": chars, "copy": copy, "why": "ok", "p_mean": 0.2}


def _by():
    return {f"s{i}": [_a(5), _a(10, 120), _a(200, 60), _a(250, 180), _a(280, 110, 0.6)] for i in range(30)}


def test_pick_pairs_buckets_and_each_source_once():
    p = pick_pairs(_by(), {"contrast": 5, "cross": 5, "late": 5}, 20, 150, 0)
    assert len(p) == 15 and len({x["prompt_id"] for x in p}) == 15
    assert {x["bucket"] for x in p} == {"contrast", "cross", "late"}
    for x in p:
        if x["bucket"] == "cross":
            assert x["a"]["step"] <= 20 and x["b"]["step"] >= 150
        if x["bucket"] == "contrast":
            assert x["a"]["step"] >= 150 and x["b"]["step"] >= 150 and x["a"]["chars"] * 1.5 <= x["b"]["chars"]
    assert p == pick_pairs(_by(), {"contrast": 5, "cross": 5, "late": 5}, 20, 150, 0)


def test_to_items_adds_swapped_repeats():
    p = pick_pairs(_by(), {"contrast": 5, "cross": 5, "late": 5}, 20, 150, 0)
    it = to_items(p, 3, 0)
    assert len(it) == 18 and sum(x["repeat"] for x in it) == 3
    orig = {x["pair"]: x for x in it if not x["repeat"]}
    for x in it:
        if x["repeat"]:
            assert x["A"] == orig[x["pair"]]["B"] and x["B"] == orig[x["pair"]]["A"] and x["id"].endswith("r")
    assert all("step" not in str(x["id"]) for x in it)


def test_valid():
    ok = {"winner": "tie", "score_a": 4, "score_b": 4, "flaws_a": ["none"], "flaws_b": ["too_short"], "note": "x"}
    assert valid(ok)["winner"] == "tie"
    assert valid({**ok, "winner": "C"}) is None
    assert valid({**ok, "score_a": 0}) is None
    assert valid({**ok, "flaws_a": ["bad"]}) is None


def _row(pair, winner, bucket="cross", repeat=False, a=None, b=None):
    return {"pair": pair, "winner": winner, "bucket": bucket, "repeat": repeat, "A": a or _a(5, 100), "B": b or _a(200, 50), "score_a": 4, "score_b": 2, "flaws_a": ["none"], "flaws_b": ["too_short"], "task_type": "QA"}


def test_repeat_agreement_requires_flip():
    rows = [_row(0, "A"), _row(0, "B", repeat=True), _row(1, "A"), _row(1, "A", repeat=True), _row(2, "tie"), _row(2, "tie", repeat=True)]
    assert repeat_agreement(rows) == {"n": 3, "agree": 0.667}


def test_summarize():
    rows = [_row(0, "A"), _row(1, "B"), _row(2, "tie"), _row(0, "B", repeat=True)]
    r = summarize(rows)
    assert r["pairs"] == 3 and r["calls"] == 4 and r["tie"] == 0.333
    assert r["longer_wins"] == 0.5 and r["by_bucket"]["cross"]["late_wins"] == 0.5
    assert r["flaws"]["too_short"] == 3 and r["low_scores(<=3)"] == 3
