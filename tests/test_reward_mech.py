"""机制分析里离线重算的奖励 / 优势，要和训练时 AnswerRewarder.score_group + group_advantages 逐条一致。"""
import random
from concurrent.futures import ThreadPoolExecutor

from ragtruth import reward_mech as rm
from ragtruth.convert import split_sentences
from train.ans_reward import AnswerRewarder, sentence_adv, sentences
from train.gen_reward import group_advantages

SRC = "Briefly answer the following question:\nwhat is x\npassage 1: " + "The quick brown fox jumps over the lazy dog near the river bank. " * 30
REF = "Alpha is one. Beta is two. Gamma is three. Delta is four."


def fake_group(rng, n=16):
    gens = []
    for k in range(n):
        ns = rng.randint(2, 6)
        text = " ".join(f"Item {k} part {j} says something about topic number {rng.randint(0, 99)}." for j in range(ns))
        if k == 3:
            text = "Unable to answer based on given passages."
        if k == 5:
            text = "This one never ends"
        gens.append({"text": text, "finish": "length" if k == 7 else "stop"})
    return gens


def test_offline_reward_matches_training():
    rng = random.Random(0)
    row = {"id": "q", "task_type": "QA", "source": SRC, "ref_text": REF, "ref_chars": 180}
    gens = fake_group(rng)
    table = {g["text"].strip(): [round(rng.random() ** 3, 4) for _ in sentences(g["text"].strip())] for g in gens}
    sats = {g["text"].strip(): rng.random() for g in gens}

    class RW(AnswerRewarder):
        def p_answer(self, source, sents, model=None, temp=None, priority=0):
            return table["".join(sents)] if "".join(sents) in table else table[next(t for t in table if sentences(t) == sents)]

        def p_sat(self, row, answer, priority=0):
            return sats[answer]

        def p_refuse(self, row, priority=0):
            return 0.7

    rw = RW(None, "c", 1.0, ThreadPoolExecutor(2), "sat", "pair", 1.0, rm.SAT_W, "refuse", 1.0, rm.REFUSE_W, "count", count_ratio=rm.COUNT_RATIO, count_min=rm.COUNT_MIN)
    rw.group_pool = ThreadPoolExecutor(2)
    scored = rw.score_group(row, gens)
    adv = group_advantages([s["reward"] for s in scored], [None] * len(scored))
    assert {s["why"] for s in scored} >= {"ok", "refusal"}

    rec = {"id": "q", "count_k": rw.count_k_for(row), "has_ref": True,
           "gens": [{**rw.describe(g["text"], g["finish"], row), "spans": [list(x) for x in split_sentences(g["text"].strip())]} for g in gens]}
    ps = [table[g["text"].strip()] if d["why"] == "ok" else None for g, d in zip(gens, rec["gens"])]
    sat = [sats[g["text"].strip()] if d["why"] == "ok" else None for g, d in zip(gens, rec["gens"])]
    off = rm.group_reward(rec, ps, sat, 0.7)
    for s, a, o in zip(scored, adv, off):
        assert abs(s["reward"] - o["reward"]) < 1e-9
        assert abs(a - o["adv"]) < 1e-9
        if s.get("ps"):
            want = sentence_adv(s["ps"], s["sent_base"], s["sent_sigma"], "neg", rm.GAMMA)
            assert max(abs(x - y) for x, y in zip(want, o["sent_adv"])) < 1e-3   # sent_base / sent_sigma 存盘时四舍五入到 4 位


def test_prompt_stats_direction():
    """金标错句被 critic 打高分时，优势差为正；critic 和金标反着来时为负。"""
    gens = [{"why": "ok", "n_sent": 3, "chars": 100, "copy": 0.1, "sent_tok": [10, 10, 10]} for _ in range(8)]
    rec = {"id": "q", "count_k": 2.0, "has_ref": False, "gens": gens}
    gold = [[0.95 if (i % 2 == 0 and j == 1) else 0.05 for j in range(3)] for i in range(8)]
    good = [[0.8 if g > 0.5 else 0.05 for g in gs] for gs in gold]
    bad = [[0.05 if g > 0.5 else 0.6 for g in gs] for gs in gold]
    sat = [None] * 8
    a = rm.aggregate([rm.prompt_stats(rec, good, sat, None, gold)])
    b = rm.aggregate([rm.prompt_stats(rec, bad, sat, None, gold)])
    assert a["M1_clear"] > 1.0 and b["M1_clear"] < 0
    assert a["M2_prec_clear"] == 1.0 and a["M2_recall_clear"] == 1.0 and a["M3_sel_has_clear"] < 0
    assert a["auc_within"] == 1.0 and b["auc_within"] == 0.0
    assert a["M4_learnable"] == 1.0 and a["M4_noise_groups"] == 0.0
