import json

from ragtruth.mine import select
from ragtruth.synth import build_prompt, check, parse_json_obj, pick_seeds, to_claims
from ragtruth.task import messages

SOURCE = "Passage 1: The shop opens at 9 am and closes at 5 pm. It sells fresh bread and cakes every day."
SENTS = ["The shop opens at 9 am and closes at 5 pm.", "It sells fresh bread and cakes every day."]


def _resp(i=0, task="QA"):
    return {"id": f"rt-{i}", "label": "faithful", "bad": {}, "task_type": task, "model": "m", "source": SOURCE,
            "sentences": SENTS}


def _seed(kind="conflict"):
    return {"id": "rt-0-syn", "resp": _resp(), "idx": 1, "type": kind, "tactic": "a number"}


def test_pick_seeds_stratified_and_alternating():
    pool = [_resp(i, t) for i, t in enumerate(["QA", "Summary"] * 5)] + [{**_resp(99), "label": "hallucination"}]
    seeds = pick_seeds(pool, 4)
    assert len(seeds) == 4
    assert sorted(s["resp"]["task_type"] for s in seeds) == ["QA", "QA", "Summary", "Summary"]
    assert {s["type"] for s in seeds} == {"conflict", "baseless"}
    assert all(s["resp"]["id"] != "rt-99" for s in seeds)


def test_check_accepts_minimal_conflict():
    obj = {"sentence": "The shop opens at 8 am and closes at 5 pm.", "evidence": "The shop opens at 9 am",
           "reason": "The source says 9 am, while the sentence says 8 am.",
           "original_evidence": "The shop opens at 9 am and closes at 5 pm.",
           "original_reason": "The source gives the same opening and closing times."}
    res, problems = check(obj, _seed())
    assert problems == [] and res["evidence"] == "The shop opens at 9 am"


def test_check_rejects_bad_rewrites():
    base = {"evidence": "The shop opens at 9 am", "reason": "The source says 9 am, while the sentence says 8 am.",
            "original_evidence": "The shop opens at 9 am", "original_reason": "The source gives the same time."}
    _, p = check({**base, "sentence": SENTS[0]}, _seed())
    assert any("identical" in x for x in p)
    _, p = check({**base, "sentence": "Bananas are a popular fruit enjoyed around the whole wide world today."}, _seed())
    assert any("too much" in x for x in p)
    _, p = check({**base, "sentence": "The shop opens at 8 am and closes at 5 pm.", "evidence": None}, _seed())
    assert any("must quote" in x for x in p)
    _, p = check({**base, "sentence": "The shop opens at 8 am and closes at 5 pm.",
                  "original_reason": "The source does not mention the opening time."}, _seed())
    assert any(x.startswith("original_reason") for x in p)
    assert check(None, _seed())[1]


def test_parse_json_obj_strips_think():
    assert parse_json_obj('<think>{"x": 0}</think>\n{"sentence": "a"}') == {"sentence": "a"}
    assert parse_json_obj("no json") is None


def test_prompt_mentions_target_and_tactic():
    p = build_prompt(_seed("baseless"), [])
    assert "Rewrite sentence S1" in p and "a number" in p and "baseless" in p


def test_to_claims_pair_and_critic_input():
    rec = {"id": "rt-0-syn", "idx": 1, "type": "conflict", "task_type": "QA", "model": "m", "source": SOURCE,
           "sentences": SENTS, "sentence": "The shop opens at 8 am and closes at 5 pm.",
           "evidence": "The shop opens at 9 am", "reason": "r1", "original_evidence": None, "original_reason": "r2"}
    bad, good = to_claims(rec)
    assert bad["label"] == "hallucination" and bad["sentences"][0].startswith("The shop opens at 8")
    assert good["label"] == "faithful" and good["sentences"] == SENTS and good["type"] == "none"
    assert bad["sentences"][1] == SENTS[1] and bad["id"] != good["id"]
    assert "S1: The shop opens at 8 am" in messages(bad)[1]["content"]


def test_select_arms():
    synth = [{"id": f"s{i}"} for i in range(10)]
    mined = {f"s{i}": {"detect_rate": 0.0 if i < 3 else 1.0} for i in range(10)}
    assert len(select(synth, mined, "direct", 0.5, 0)) == 10
    assert [r["id"] for r in select(synth, mined, "critic", 0.5, 0)] == ["s0", "s1", "s2"]
    assert len(select(synth, mined, "random", 0.5, 0)) == 3
    json.dumps(select(synth, mined, "random", 0.5, 0))


def test_iterate_select_critic_balanced_and_random():
    from ragtruth.iterate import select
    pool = [{"id": f"s{i}", "type": "conflict"} for i in range(20)]
    rates = {f"s{i}": (0.0 if i < 3 else 1.0) for i in range(20)}
    crit = select(pool, rates, "critic", 10, 0.5, 0)
    assert len(crit) == 10 and sum(rates[r["id"]] < 0.5 for r in crit) == 3   # 难样本不够 5 条：全取，易样本补足
    assert len(select(pool, rates, "random", 10, 0.5, 0)) == 10
    pool2 = [{"id": f"s{i}", "type": "conflict"} for i in range(20)]
    rates2 = {f"s{i}": (0.0 if i < 12 else 1.0) for i in range(20)}
    assert sum(rates2[r["id"]] < 0.5 for r in select(pool2, rates2, "critic", 10, 0.5, 0)) == 5   # 1:1


def test_pick_seeds_excludes_used_sentences():
    pool = [{**_resp(i), "sentences": SENTS} for i in range(4)]
    used = {(f"rt-{i}", 1) for i in range(4)}
    seeds = pick_seeds(pool, 4, exclude=used)
    assert all(s["idx"] == 2 for s in seeds) and len({s["id"] for s in seeds}) == len(seeds)
