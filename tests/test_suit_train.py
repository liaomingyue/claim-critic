"""合适度模型：选项渲染、评测用的固定顺序、温度拟合（不需要 GPU）。"""
import math

from ragtruth.suit_data import MECHS
from train import suit_train as st


def test_render_lists_all_options_in_perm_order():
    perm = list(reversed(MECHS))
    msgs = st.render("SRC", perm)
    sys = msgs[0]["content"]
    for i, m in enumerate(perm):
        assert f"{st.LETTERS[i]}. {st.DEFS[m]}" in sys
    assert msgs[1] == {"role": "user", "content": "SRC"}


def test_perms_reproducible_and_first_is_identity():
    a, b = st.perms_for("x1", 4), st.perms_for("x1", 4)
    assert a == b and a[0] == list(MECHS) and all(sorted(p) == sorted(MECHS) for p in a)
    assert st.perms_for("x2", 4)[1:] != a[1:]


def test_fit_temperature_recovers_scale():
    import random
    rng = random.Random(0)
    logits, labels = [], []
    for _ in range(400):
        z = [rng.gauss(0, 2) for _ in MECHS]
        y = [int(rng.random() < 1 / (1 + math.exp(-v / 2.0))) for v in z]   # 真实温度 2
        logits.append(z); labels.append(y)
    assert 1.6 < st.fit_temperature(logits, labels) < 2.5
