"""ragtruth.sat_build 的单元测试（不需要模型）。"""
import random

from ragtruth.sat_build import SAT_ABS_SYSTEM, SAT_PAIR_SYSTEM, build_labeled, build_synth, degrade, pair_rows, split_of

ANS = "The cafe opens at nine every morning of the week. It serves tea and small cakes all day long. Staff close the doors at five in the evening."
SRC = "Summarize the news:\nThe cafe opens at nine every morning. It serves tea and cakes all day. The doors close at five each evening. Locals say the owner is friendly.\n\noutput:"


def test_degrade_kinds():
    rng = random.Random(0)
    t = degrade("truncated", ANS, SRC, "x", rng)
    assert t and len(t) < len(ANS) and ANS.startswith(t) and t[-1] not in ".!?"
    assert degrade("too_short", ANS, SRC, "x", rng) == "The cafe opens at nine every morning of the week."
    assert degrade("too_short", "Only one sentence here.", SRC, "x", rng) is None
    c = degrade("copied", ANS, SRC, "x", rng)
    assert c and "cafe" in c or "tea" in c or "doors" in c or "owner" in c
    assert c not in ANS
    assert degrade("off_topic", ANS, SRC, "another answer", rng) == "another answer"
    assert degrade("off_topic", ANS, SRC, ANS, rng) is None


def test_pair_rows_labels_follow_swap():
    r = pair_rows("p", SRC, "good", "bad", "x", "claude", {"n": 0})
    assert [x["label"] for x in r] == ["A", "B"]
    assert "[answer A]\ngood" in r[0]["prompt"][1]["content"] and "[answer A]\nbad" in r[1]["prompt"][1]["content"]
    r = pair_rows("p", SRC, "bad", "good", "y", "claude", {"n": 0})
    assert [x["label"] for x in r] == ["B", "A"]
    assert r[0]["prompt"][0]["content"] == SAT_PAIR_SYSTEM and r[0]["group"] == "p"


def test_split_is_by_prompt_and_stable():
    assert split_of("abc", 0) == split_of("abc", 0)
    c = {split_of(str(i), 0) for i in range(200)}
    assert c == {"train", "calib", "test"}


def _a(step, text, why="ok"):
    return {"step": step, "text": text, "chars": len(text), "copy": 0.1, "why": why, "p_mean": 0.2}


def test_build_labeled_drops_ties_for_pair_keeps_for_abs():
    src = {"p": {"source": SRC, "task_type": "QA"}}
    lab = [{"prompt_id": "p", "winner": "tie", "bucket": "late", "task_type": "QA", "score_a": 4, "score_b": 3, "A": _a(200, "x"), "B": _a(210, "y")},
           {"prompt_id": "p", "winner": "B", "bucket": "cross", "task_type": "QA", "score_a": 2, "score_b": 5, "A": _a(5, "m"), "B": _a(200, "n")}]
    pair, abso = build_labeled(lab, src)
    assert len(pair) == 2 and [r["label"] for r in pair] == ["B", "A"]
    assert len(abso) == 4 and [r["label"] for r in abso] == ["A", "B", "B", "A"]
    assert abso[0]["prompt"][0]["content"] == SAT_ABS_SYSTEM


def test_build_synth_counts_and_labels():
    src = {f"p{i}": {"source": SRC, "task_type": "QA"} for i in range(8)}
    by = {f"p{i}": [_a(5, ANS + f" Item {i} is here."), _a(9, ANS, why="length")] for i in range(8)}
    pair, abso, counts = build_synth(by, src, 3, 0)
    assert counts == {"truncated": 3, "too_short": 3, "copied": 3, "off_topic": 3}
    assert len(pair) == 24 and len(abso) == 12 and all(r["label"] == "B" for r in abso)
    assert len({r["id"] for r in pair}) == 24 and {r["origin"] for r in abso} == {"synth_truncated", "synth_too_short", "synth_copied", "synth_off_topic"}
