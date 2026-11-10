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
