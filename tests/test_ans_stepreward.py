"""计数 + 去底噪奖励、句子层优势、token 对句子、逐 token 优势进损失（分步奖励，单独优化出错的那一句）。不需要服务。"""
import collections
import types

import pytest

from train.ans_grpo import token_advantages, token_sentence_index
from train.ans_reward import AnswerRewarder, combine, count_reward, group_baseline, sentence_adv
from train.gen_grpo import has_adv

ROW = {"task_type": "Summary", "source": "Hours for the corner coffee shop: doors 9am, shut 5pm daily; tea served.", "ref_chars": 42}


def test_count_reward_counts_errors_and_does_not_dilute():
    one5, two5 = [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]
    one10 = [1] + [0] * 9
    r = lambda ps: count_reward(ps, 0.0)
    assert r([0] * 5) == 1.0
    assert abs(r(one5) - (1 - 0.5 - 0.5 / 3)) < 1e-9
    assert r(two5) < r(one5) - 0.15          # 第二句错多扣约 0.17（旧公式只多扣 0.1）
    assert abs(r(one10) - r(one5)) < 1e-9     # 句子多不稀释
    assert combine(one10) > combine(one5)      # 旧公式会被稀释


def test_count_reward_baseline_removes_length_bias():
    noise = 0.1
    b = noise
    full4, full8 = [noise] * 4, [noise] * 8
    assert count_reward(full4, b) == count_reward(full8, b) == 1.0   # 底噪不随句数累加
    assert count_reward(full8, 0.0) < count_reward([0.2] + [noise] * 3, 0.0)   # 不减底噪：8 句全对反而比 4 句带错低
    assert count_reward([0.6] + [noise] * 3, b) < 1.0


def test_group_baseline_and_floor():
    b, sd = group_baseline([[0.1, 0.1], [0.1]])
    assert abs(b - 0.1) < 1e-9 and sd == 0.1   # 全一样 → 标准差取下限
    b, sd = group_baseline([[0.0, 1.0]])
    assert abs(b - 0.5) < 1e-9 and abs(sd - 0.5) < 1e-9
    assert group_baseline([]) == (0.0, 0.1)


def test_sentence_adv_neg_center_gamma():
    ps = [0.1, 0.6, 0.1]
    a = sentence_adv(ps, 0.1, 0.25)
    assert a[0] == 0.0 and a[2] == 0.0 and abs(a[1] + 2.0) < 1e-9   # 只罚超出底噪的那一句
    c = sentence_adv([0.0, 0.6], 0.1, 0.25, mode="center")
    assert c[0] > 0 and c[1] < 0
    g = sentence_adv(ps, 0.1, 0.25, gamma=0.5)
    assert abs(g[0] + 1.0) < 1e-9 and abs(g[1] + 2.0) < 1e-9 and g[2] == 0.0   # 只往前传一步、只传负值


class Tok:
    """一个字符一个 token 的假分词器。"""
    def __init__(self, text):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text[:len(ids)]


def test_token_sentence_index_maps_chars():
    text = " Ab. Cd."           # 开头空格在 strip 前
    spans = [[0, 3], [4, 7]]    # strip 后 "Ab. Cd."
    idx = token_sentence_index(Tok(text), list(range(len(text))), text, spans)
    assert idx == [-1, 0, 0, 0, -1, 1, 1, 1]


def test_token_sentence_index_rescales_on_mismatch():
    class Half(Tok):
        def decode(self, ids, skip_special_tokens=True):
            return "x" * (2 * len(ids))   # 解码长度是 text 的两倍
    text = "Ab. Cd."
    idx = token_sentence_index(Half(text), list(range(len(text))), text, [[0, 3], [4, 7]])
    assert idx[0] == 0 and idx[-1] == 1


def test_token_advantages():
    assert token_advantages(0.5, [0.0, -2.0], [-1, 0, 1, 1], 1.0) == [0.5, 0.5, -1.5, -1.5]
    assert token_advantages(0.5, [0.0, -2.0], [0, 1], 0.0) == [0.5, 0.5]


def test_has_adv_scalar_and_list():
    assert has_adv({"adv": 0.3}) and not has_adv({"adv": 0.0})
    assert has_adv({"adv": [0.0, -0.2]}) and not has_adv({"adv": [0.0, 0.0]})


def _rw(shape):
    rw = AnswerRewarder(None, "c", 1.0, None, shape=shape)
    rw.group_pool = types.SimpleNamespace(map=lambda f, xs: [f(x) for x in xs])
    scores = {"The cafe opens at nine.": 0.1, "It closes at five.": 0.1, "It closes at six.": 0.9}
    rw.p_answer = lambda source, sents, *a, **k: [scores.get(s.strip(), 0.1) for s in sents]
    rw.p_sat = lambda row, ans, priority=0: None
    return rw


def test_score_group_count_shape_uses_group_baseline():
    gens = [{"text": "The cafe opens at nine. It closes at five.", "finish": "stop"},
            {"text": "The cafe opens at nine. It closes at six.", "finish": "stop"}]
    out = _rw("count").score_group(ROW, gens)
    b = (0.1 + 0.1 + 0.1 + 0.9) / 4
    assert abs(out[0]["sent_base"] - b) < 1e-4 and out[0]["spans"] == [[0, 23], [24, 42]]
    assert out[0]["reward"] == 1.0 and out[1]["reward"] < out[0]["reward"]
    legacy = _rw("legacy").score_group(ROW, gens)
    assert abs(legacy[1]["reward"] - combine([0.1, 0.9])) < 1e-6   # 旧行为不变


def test_producer_builds_token_advantages(monkeypatch):
    """sent_beta > 0：错句的 token 优势比同一条回答里正确句子的低；sent_beta = 0：每条一个标量（旧行为）。"""
    from train import ans_grpo
    text_ok, text_bad = "The cafe opens at nine. It closes at five.", "The cafe opens at nine. It closes at six."

    class Srv:
        def sample(self, name, ids, n, max_tokens, temp):
            return [{"text": t, "ids": list(range(len(t))), "old_lp": [0.0] * len(t), "finish": "stop"} for t in (text_ok, text_bad)]

    class PTok:
        def __call__(self, p):
            return {"input_ids": [1, 2]}

        def decode(self, ids, skip_special_tokens=True):
            return (text_ok if len(ids) <= len(text_ok) else text_bad)[:len(ids)]

    monkeypatch.setattr(ans_grpo, "policy_prompt", lambda tok, r, sys_qa: "p")
    for beta in (1.0, 0.0):
        a = types.SimpleNamespace(prompts_per_step=1, n=2, max_tokens=64, temperature=1.0, sys_qa=0, version_prefix="", pipeline=0,
                                  sent_beta=beta, sent_mode="neg", sent_gamma=0.0, max_stale=2)
        prod = ans_grpo.Producer.__new__(ans_grpo.Producer)
        prod.a, prod.tok, prod.servers, prod.rw = a, PTok(), [Srv()], _rw("count")
        prod.rows, prod.pos, prod.version = [{"id": "r", "task_type": "Summary", **ROW}], 0, 0
        prod.gen_pool = prod.score_pool = types.SimpleNamespace(map=lambda f, xs: [f(x) for x in xs])
        ss = prod.make_batch()["samples"]
        bad = ss[1]["adv"]
        if beta:
            assert isinstance(bad, list) and len(bad) == len(text_bad)
            first, second = bad[5], bad[30]   # 第一句（对）、第二句（错）里的 token
            assert second < first
        else:
            assert all(isinstance(s["adv"], float) for s in ss)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_chunk_loss 在 cuda 上建张量")
def test_chunk_loss_token_adv_equals_scalar_when_constant():
    import torch
    from train.gen_grpo import _chunk_loss
    args = types.SimpleNamespace(clip=0.2, beta=0.05)
    lp = torch.tensor([[-0.5, -1.0, -0.2]], device="cuda")
    mask = torch.tensor([[True, True, False]], device="cuda")
    base = {"old_lp": [-0.6, -1.0], "ids": [1, 2]}
    l1 = _chunk_loss([{**base, "adv": 0.7}], lp, mask, [[-0.5, -1.0]], 2, args, collections.Counter())
    l2 = _chunk_loss([{**base, "adv": [0.7, 0.7]}], lp, mask, [[-0.5, -1.0]], 2, args, collections.Counter())
    assert abs(l1.item() - l2.item()) < 1e-6
    l3 = _chunk_loss([{**base, "adv": [0.7, -0.7]}], lp, mask, [[-0.5, -1.0]], 2, args, collections.Counter())
    assert abs(l3.item() - l1.item()) > 1e-3


def test_count_k_from_reference_ratio():
    rw = AnswerRewarder(None, "c", 1.0, None, shape="count", count_k=3.0, count_ratio=0.5, count_min=2.0)
    ref10 = " ".join(f"Sentence number {i} is here." for i in range(10))
    assert rw.count_k_for({"ref_text": ref10}) == 5.0
    assert rw.count_k_for({"ref_text": "One. Two."}) == 2.0      # 下限
    assert rw.count_k_for({}) == 3.0                              # 没有参考回答
    assert AnswerRewarder(None, "c", 1.0, None).count_k_for({"ref_text": ref10}) == 3.0   # ratio 0 = 固定 K


def _rewarder(ps_by_text, sat_by_text, s0):
    from concurrent.futures import ThreadPoolExecutor
    from train.ans_reward import AnswerRewarder, sentences

    class RW(AnswerRewarder):
        def p_answer(self, source, sents, model=None, temp=None, priority=0):
            return next(v for t, v in ps_by_text.items() if sentences(t) == sents)

        def p_sat(self, row, answer, priority=0):
            return sat_by_text[answer]

    rw = RW(None, "c", 1.0, ThreadPoolExecutor(2), "sat", "pair", 1.0, 0.8, shape="count")
    rw.group_pool = ThreadPoolExecutor(2)
    rw.sat_s0 = s0
    return rw


def test_satisfaction_weight_follows_critic_contrast():
    """critic 在一组里没有区分度时，满意度不再拉开奖励；有区分度时照常扣。"""
    src = "Briefly answer the following question:\nwhat is x\npassage 1: " + "The quick brown fox jumps over the lazy dog near the river bank. " * 30
    row = {"id": "q", "task_type": "QA", "source": src, "ref_text": "Alpha is one. Beta is two.", "ref_chars": 60}
    texts = [f"Item {k} says something about topic {k}. It adds one more remark here." for k in range(4)]
    gens = [{"text": t, "finish": "stop"} for t in texts]
    sat = dict(zip(texts, [0.9, 0.1, 0.9, 0.1]))
    quiet = dict(zip(texts, [[0.05, 0.05]] * 4))
    loud = dict(zip(texts, [[0.05, 0.05], [0.05, 0.05], [0.9, 0.05], [0.9, 0.05]]))
    fixed = _rewarder(quiet, sat, 0.0).score_group(row, gens)
    assert fixed[0]["reward"] - fixed[1]["reward"] > 0.3                       # 固定权重：只靠满意度拉开
    dyn = _rewarder(quiet, sat, 0.1).score_group(row, gens)
    assert dyn[0]["sat_w"] == 0.0 and len({s["reward"] for s in dyn}) == 1     # critic 没话说：全组奖励一样
    dyn = _rewarder(loud, sat, 0.1).score_group(row, gens)
    assert dyn[0]["sat_w"] == 0.8 and dyn[0]["reward"] - dyn[1]["reward"] > 0.3


def test_group_advantage_std_floor():
    from train.gen_reward import group_advantages
    small, big = [0.70, 0.72, 0.70, 0.72], [0.2, 0.8, 0.2, 0.8]
    assert abs(max(group_advantages(small, [None] * 4)) - max(group_advantages(big, [None] * 4))) < 1e-2   # 旧行为：差距再小也放大到一样
    a = group_advantages(small, [None] * 4, std_floor=0.1)
    assert abs(max(a) - 0.01 / (0.1 + 1e-4)) < 1e-6
    assert group_advantages(big, [None] * 4, std_floor=0.1) == group_advantages(big, [None] * 4)


def test_remap_is_piecewise_linear_and_off_by_default():
    from train.ans_reward import remap
    k = ([0.1, 0.5, 0.9], [0.03, 0.2, 0.6])
    assert remap(0.7, None) == 0.7
    assert remap(0.0, k) == 0.03 and remap(1.0, k) == 0.6
    assert abs(remap(0.3, k) - 0.115) < 1e-9 and abs(remap(0.7, k) - 0.4) < 1e-9


def test_reward_uses_remapped_scores_and_keeps_raw():
    src = "Briefly answer the following question:\nwhat is x\npassage 1: " + "The quick brown fox jumps over the lazy dog near the river bank. " * 30
    row = {"id": "q", "task_type": "QA", "source": src, "ref_text": "Alpha is one. Beta is two.", "ref_chars": 60}
    texts = [f"Item {k} says something about topic {k}. It adds one more remark here." for k in range(4)]
    gens = [{"text": t, "finish": "stop"} for t in texts]
    ps = dict(zip(texts, [[0.05, 0.05], [0.05, 0.05], [0.6, 0.05], [0.9, 0.05]]))
    sat = dict(zip(texts, [0.9] * 4))
    raw = _rewarder(ps, sat, 0.0).score_group(row, gens)
    rw = _rewarder(ps, sat, 0.0)
    rw.p_knots = ([0.0, 0.5, 0.7, 1.0], [0.0, 0.2, 0.3, 0.8])
    cal = rw.score_group(row, gens)
    assert "ps_raw" not in raw[2] and cal[2]["ps_raw"] == [0.6, 0.05] and cal[2]["ps"][0] == 0.25
    assert cal[2]["p_max"] == 0.6                                   # 日志里的最高分仍是原始分数
    assert raw[0]["reward"] - raw[2]["reward"] > cal[0]["reward"] - cal[2]["reward"] > 0   # 没把握的句子罚得轻了
    assert cal[2]["reward"] > cal[3]["reward"]
