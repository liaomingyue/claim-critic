from types import SimpleNamespace as NS

from ragtruth.attr_fill import T_TAIL, answer_probs, check_text, clip, has, key_of, to_think


def tok(t, top=()):
    return NS(token=t, top_logprobs=[NS(token=a, logprob=b) for a, b in top])


def test_answer_probs_takes_the_letter_after_the_last_answer_line():
    c = NS(logprobs=NS(content=[tok("Claim"), tok(" subject"), tok(":"), tok(" A"), tok("\n"), tok("Answer"), tok(":"), tok(" "), tok("B", [("B", -0.1), (" A", -2.5), ("C", -4.0)])]))
    lp = answer_probs(c)
    assert lp == {"A": -2.5, "B": -0.1, "C": -4.0}
    assert answer_probs(NS(logprobs=NS(content=[tok("B", [("B", -0.1)])]))) is None


def test_to_think_swaps_the_last_instruction_and_keeps_bare_rows_short():
    sys_ = "You check.\n\nOptions.\n\nAnswer with a single letter: A, B or C."
    base = {"id": "x", "prompt": [{"role": "system", "content": sys_}, {"role": "user", "content": "u"}], "completion": "A", "label": "A"}
    t = to_think(base)
    assert t["completion"] == "Answer: A" and t["prompt"][0]["content"].endswith(T_TAIL) and t["prompt"][1]["content"] == "u" and t["label"] == "A"
    t = to_think({**base, "label": "B", "check": check_text("Smith", "Jones said so.", "Jones", "B")})
    assert t["completion"].splitlines() == ["Claim subject: Smith", 'Source: "Jones said so."', "Source subject: Jones", "Answer: B"] and "check" not in t


def test_keys_and_clip():
    assert key_of("person", "Hillary Clinton") == "clinton" and key_of("org", "White House") == "white house"
    assert has("clinton", "clinton's vote") and not has("lin", "clinton said")
    ev = " ".join(f"w{i}" for i in range(200)) + " Clinton " + " ".join(f"v{i}" for i in range(200))
    c = clip(ev, "clinton", 20)
    assert "Clinton" in c and len(c.split()) <= 22
