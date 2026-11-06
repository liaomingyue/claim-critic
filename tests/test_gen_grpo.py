"""train.gen_reward / train.gen_grpo 的单元测试（不需要 GPU / 服务）：硬规则、组内优势值、PPO 截断损失的数学。"""
import json
import math

import pytest

from train.gen_reward import RunningStat, group_advantages, parse

ROW = {"source": "The cafe opens at 9 am. It serves tea.", "sentences": ["The cafe opens at 9 am.", "It serves tea."],
       "task_type": "QA"}


def out(**kw):
    o = {"plan": "p", "idx": 1, "type": "conflict", "tactic": "a date, time or duration", "sentence": "The cafe opens at 10 am.",
         "evidence": "opens at 9 am", "reason": "The source says 9 am."}
    o.update(kw)
    return json.dumps(o)


def test_parse_rules():
    assert parse(out(), ROW)[1] == "ok"
    assert parse("not json", ROW)[1] == "unparsable"
    assert parse(out(idx=5), ROW)[1] == "bad_idx"
    assert parse(out(type="none"), ROW)[1] == "bad_type"
    assert parse(out(sentence="The cafe opens at 9 am!"), ROW)[1] == "identical"
    assert parse(out(sentence="It serves tea."), ROW)[1] == "copied_other_sentence"
    assert parse(out(sentence="x" * 400), ROW)[1] == "too_long"
    assert parse(out(sentence="This hallucination says the cafe opens at 10 am."), ROW)[1] == "leaks_label"
    rec, why = parse(out(op="insert", idx=3, sentence="It also serves cake."), ROW)
    assert why == "ok" and rec["op"] == "insert"


def test_group_advantages_same_type_baseline():
    r = [1.0, 0.0, 0.6, 0.2]
    t = ["conflict", "conflict", "baseless", "baseless"]
    a = group_advantages(r, t)
    std = math.sqrt(sum((x - 0.45) ** 2 for x in r) / 4)
    assert abs(a[0] - (1.0 - 0.5) / (std + 1e-4)) < 1e-6 and abs(a[2] - (0.6 - 0.4) / (std + 1e-4)) < 1e-6
    assert group_advantages([0.3] * 4, t) == [0.0] * 4
    assert abs(group_advantages([1.0, 0.0], ["conflict", None])[1] - (0.0 - 0.5) / (0.5 + 1e-4)) < 1e-6


def test_running_stat_bounded():
    s = RunningStat()
    xs = [s.norm("k", v) for v in (0.1, 0.9, 0.5, 0.2, 0.8)]
    assert all(0.0 < x < 1.0 for x in xs)


def test_ppo_clip_math():
    import torch
    lp, old, a, clip = torch.tensor([0.0, -1.0]), torch.tensor([-0.5, -1.0]), 1.0, 0.2
    ratio = torch.exp(lp - old)
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    assert abs(surr[0].item() - 1.2) < 1e-6 and abs(surr[1].item() - 1.0) < 1e-6
    ref = torch.tensor([0.0, -1.0])
    kl = torch.exp(ref - lp) - (ref - lp) - 1
    assert kl.min().item() >= 0 and abs(kl[0].item()) < 1e-6


def test_entry_points_resolve():
    """main() 分发到的入口和训练循环用到的模块级函数都得在（重构时被删掉过，只有开训才报 NameError）。"""
    import ast
    import builtins
    from pathlib import Path
    tree = ast.parse(Path("train/gen_grpo.py").read_text(encoding="utf-8"))
    top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"run", "check", "check_share", "push", "batch_stats", "QuickEval", "train_step", "Producer", "Server"} <= top
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    called = {n.func.id for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    local = {n.id for n in ast.walk(run) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    imported = {a.asname or a.name for n in ast.walk(run) if isinstance(n, ast.ImportFrom) for a in n.names}
    mod = {a.asname or a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert called - top - local - imported - mod - set(dir(builtins)) == set()


def test_invented_tactics_share_one_bucket():
    """编新改法名字拿不到多样性加分（第一版正式训练的 reward hack）。"""
    from train.gen_reward import OTHER_TACTIC, closed_tactic, parse
    from ragtruth.synth import TACTICS
    for t in ("a preheating step", "a closing time that the source never mentions", "", "zzz"):
        assert closed_tactic(t, "baseless") == OTHER_TACTIC
    for kind, cs in TACTICS.items():
        for c in cs:
            assert closed_tactic(c, kind) == c
            assert closed_tactic(c.upper() + ".", kind) == c
    row = {"source": "s", "sentences": ["A b c.", "D e f."], "task_type": "QA"}
    a, _ = parse('{"idx": 1, "type": "baseless", "tactic": "a preheating step", "sentence": "A b d."}', row)
    b, _ = parse('{"idx": 1, "type": "baseless", "tactic": "a freezing step", "sentence": "A b e."}', row)
    assert a["tactic"] == b["tactic"] == OTHER_TACTIC and a["tactic_raw"] != b["tactic_raw"]


def _rewarder(window=8):
    from train.gen_reward import Rewarder
    rw = Rewarder(None, {"rm1": 1.0, "rm2": 1.0, "rm3": 1.0}, window=window, type_floor=0.0)
    rw.option_p = lambda model, system, user, opts, priority=0: {"rm3": {"A": 0.9, "B": 0.1, "C": 0, "D": 0},
                                                     "rm1": {"A": 0.5, "B": 0.25, "C": 0.25}, "rm2": {"A": 0.5, "B": 0.5}}[model]
    return rw


def _out(tactic, sent):
    import json
    return json.dumps({"idx": 1, "type": "baseless", "tactic": tactic, "sentence": sent})


ROW_DIV = {"source": "s", "sentences": ["A b c.", "D e f."], "task_type": "QA"}


def test_diversity_counts_recent_window_only():
    """多样性按最近窗口算：常见改法减分，窗口滑过之后恢复（不再累计衰减到 0）。"""
    import math
    rw = _rewarder(window=8)
    a, b = "one extra item in a list", "a reason or motive the source does not give"
    assert rw.score(_out(a, "x1"), ROW_DIV)["div"] == 1.0
    for i in range(7):
        rw.score(_out(a, f"y{i}"), ROW_DIV)
    assert abs(rw.score(_out(a, "z"), ROW_DIV)["div"] - 1 / math.sqrt(1 + 8)) < 1e-9
    assert rw.score(_out(b, "w"), ROW_DIV)["div"] == 1.0       # 冷门标准改法拿满分
    for i in range(8):
        rw.score(_out(b, f"v{i}"), ROW_DIV)                     # 把 a 挤出窗口
    assert rw.score(_out(a, "u"), ROW_DIV)["div"] == 1.0
    assert sum(rw.cls.values()) == len(rw.recent) == 8


def test_place_diversity_rewards_spreading_within_group():
    from train.gen_reward import OTHER_TACTIC, place_diversity
    mk = lambda i, t="one extra item in a list": {"qualified": True, "op": "rewrite", "idx": i, "type": "baseless", "tactic": t, "hard": 0.5, "real": 0.5, "div": 9.0, "reward": 9.0}
    ss = [mk(1), mk(1), mk(1), mk(1), mk(2), mk(3, OTHER_TACTIC), {"qualified": False, "reward": 0.01, "type": "baseless"}]
    place_diversity(ss, (1.0, 1.0, 1.0))
    assert ss[0]["div"] == 0.5 and ss[4]["div"] == 1.0 and ss[5]["div"] == 0.0 and ss[6]["reward"] == 0.01
    assert ss[4]["reward"] > ss[0]["reward"] > ss[5]["reward"]
    adv = group_advantages([s["reward"] for s in ss], [s.get("type") for s in ss])
    assert adv[4] > adv[0]      # 同类型基线之后，改冷门位置的优势值仍然更高，梯度往分散走


def test_place_diversity_ignores_labels():
    """同一位置换类型、换改法都不算换位置。"""
    from train.gen_reward import place_diversity
    mk = lambda t, kind: {"qualified": True, "op": "rewrite", "idx": 1, "type": kind, "tactic": t, "hard": 0.5, "real": 0.5, "div": 9.0, "reward": 9.0}
    ss = [mk("a date, time or duration", "conflict"), mk("one extra item in a list", "baseless"), mk("a number, quantity or rating", "conflict"), mk("a reason or motive the source does not give", "baseless")]
    place_diversity(ss, (1.0, 1.0, 1.0))
    assert all(s["div"] == 0.5 for s in ss)


def test_invented_tactic_gets_no_diversity():
    rw = _rewarder()
    for i, name in enumerate(("a preheating step", "a freezing step", "a closing time")):
        assert rw.score(_out(name, f"s{i}"), ROW_DIV)["div"] == 0.0


def _clean(**kw):
    return {"other_tactic_share": 0.02, "raw_tactics": 14, "conflict_share": 0.35, "qualified_rate": 0.7, "top_tactic_share": 0.3, "kl": 0.005,
            "reason_chars": 150.0, "evidence_chars": 80.0, "sentence_chars": 120.0, **kw}


def test_tripwire_quiet_on_clean_run_and_fires_on_each_signal():
    from train.gen_grpo import TripWire
    t = TripWire()
    assert all(t.update(_clean()) == [] for _ in range(300))
    for bad, key in (({"other_tactic_share": 0.1}, "other_tactic_share"), ({"raw_tactics": 40}, "raw_tactics"),
                     ({"conflict_share": 0.7}, "conflict_share"), ({"conflict_share": 0.05}, "conflict_share"),
                     ({"qualified_rate": 0.5}, "qualified_rate"), ({"top_tactic_share": 0.7}, "top_tactic_share"), ({"kl": 0.2}, "kl"), ({"reason_chars": 300.0}, "reason_chars"),
                     ({"sentence_chars": 200.0}, "sentence_chars")):
        t = TripWire()
        for _ in range(60):
            t.update(_clean())
        fired = []
        for _ in range(25):
            fired += t.update(_clean(**bad))
        assert any(key in r for r in fired), bad
    t = TripWire()   # 前期不报：基线还没有
    assert all(t.update(_clean(kl=0.5)) == [] for _ in range(44))


def test_save_and_load_state_roundtrip(tmp_path):
    """LoRA、优化器、奖励统计、prompt 位置都恢复；奖励的窗口计数和 recent 一致。"""
    import types
    import torch
    from peft import LoraConfig, get_peft_model
    from train import gen_grpo as G
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    m = get_peft_model(net, LoraConfig(r=2, target_modules=["0", "2"]), adapter_name="policy")
    ps = G.lora_params(m, "policy")
    opt = torch.optim.AdamW(ps, lr=0.1)
    m(torch.ones(1, 4)).sum().backward()
    opt.step()
    rw = _rewarder(window=4)
    rw.score(_out("one extra item in a list", "a"), ROW_DIV)
    d = G.save_state(m, opt, rw, types.SimpleNamespace(pos=48), 25, 25, tmp_path, keep=2)
    want = [p.detach().clone() for p in ps]
    want_opt = opt.state_dict()["state"][0]["exp_avg"].clone()
    want_cls = dict(rw.cls)
    with torch.no_grad():
        for p in ps:
            p.add_(1.0)
    m(torch.ones(1, 4)).sum().backward()
    opt.step()
    rw2, prod2 = _rewarder(window=4), types.SimpleNamespace(pos=0)
    st = G.load_state(m, opt, rw2, prod2, d)
    assert st["step"] == 25 and prod2.pos == 48 and dict(rw2.cls) == want_cls and len(rw2.recent) == 1
    assert all(torch.equal(p, w) for p, w in zip(ps, want))
    assert torch.equal(opt.state_dict()["state"][0]["exp_avg"], want_opt)
    rw3 = _rewarder(window=4)
    G.load_state(m, opt, rw3, prod2, d, reset_reward=True)
    assert len(rw3.recent) == 0
    for s in (50, 75, 100):
        G.save_state(m, opt, rw, prod2, s, s, tmp_path, keep=2)
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == ["step100", "step75"]


def test_batch_stats_and_samples_saved(tmp_path):
    import gzip
    from train import gen_grpo as G
    from train.gen_reward import parse
    rw = _rewarder(window=8)
    ss = []
    for i, t in enumerate(("one extra item in a list", "a preheating step", "not json")):
        text = _out(t, f"s{i}") if t != "not json" else t
        ss.append({"prompt_id": "p", "prompt_ids": [1], "ids": [1, 2], "old_lp": [0.0, 0.0], "text": text, "finish": "stop", "adv": 0.1,
                   **rw.score(text, ROW_DIV)})
    b = {"version": 3, "samples": ss, "t_gen": 1.0, "t_rm": 1.0}
    st = G.batch_stats(b)
    assert st["other_tactic_share"] == 0.5 and st["raw_tactics"] == 2 and st["place_per_prompt"] == 1.0
    G.save_samples(b, 7, tmp_path)
    rows = [json.loads(l) for l in gzip.open(tmp_path / "samples" / "step7.jsonl.gz", "rt", encoding="utf-8")]
    assert len(rows) == 3 and rows[0]["text"] == ss[0]["text"] and rows[0]["gen_version"] == 3 and "ids" not in rows[0]


def test_quick_eval_paces_and_keeps_order(tmp_path):
    """匀速：每条回答隔 gap 才开始，总时长 ≈ eval_minutes；结果顺序和回答对齐。"""
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from train import gen_grpo as G
    q = G.QuickEval.__new__(G.QuickEval)
    q.a = types.SimpleNamespace(eval_minutes=0.01, eval_k=2, max_tokens=8, temperature=1.0, rm_concurrency=4, version_prefix="t_")
    q.rows = [{"id": f"r{i}", "chat": f"r{i}", **ROW_DIV} for i in range(6)]
    q.first, q.out, q.pool, q.futs, q.gate = None, Path(tmp_path), ThreadPoolExecutor(1), [], None
    calls = []
    q.srv = types.SimpleNamespace(sample=lambda n, p, k, m, t, priority=0: (calls.append((time.time(), p)), [{"text": _out("one extra item in a list", p + str(j)), "ids": [1], "finish": "stop"} for j in range(k)])[1])
    q.rw = _rewarder(window=32)
    import difflib
    q.difflib = difflib
    t0 = time.time()
    q._run(0, None)
    assert time.time() - t0 >= 0.55
    assert sorted(c[1] for c in calls) == [f"r{i}" for i in range(6)] and calls[-1][0] - calls[0][0] >= 0.45
    rows = [json.loads(l) for l in open(tmp_path / "quick_samples_t_s0.jsonl", encoding="utf-8")]
    assert [r["id"] for r in rows] == [f"r{i}" for i in range(6)]
    assert rows[3]["outputs"][1].find("r31") > 0 and json.loads(open(tmp_path / "quick_eval.jsonl", encoding="utf-8").readline())["samples"] == 12


def test_flagged_original_gets_no_reward():
    rw = _rewarder(window=8)
    base = rw.option_p
    rw.option_p = lambda model, system, user, opts, priority=0: ({"A": 0.2, "B": 0.4, "C": 0.4} if model == "rm1" else base(model, system, user, opts))
    s = rw.score(_out("one extra item in a list", "x"), ROW_DIV)
    assert s["reward"] == 0.0 and s["qualified"] is False and s["orig_flagged"] and len(rw.recent) == 0
    rw.option_p = base
    s = rw.score(_out("one extra item in a list", "y"), ROW_DIV)
    assert s["qualified"] is True and s["p_orig_halluc"] == 0.5 and "orig_flagged" not in s


def test_quick_eval_uses_idle_first_and_yields_when_busy(tmp_path):
    """生产者空闲时全速评；忙时停下，只按兜底进度走；评不完不会卡死。"""
    import threading
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from train import gen_grpo as G
    import difflib
    def make(gate, minutes):
        q = G.QuickEval.__new__(G.QuickEval)
        q.a = types.SimpleNamespace(eval_minutes=minutes, eval_k=2, max_tokens=8, temperature=1.0, rm_concurrency=4, version_prefix="t_")
        q.rows = [{"id": f"r{i}", "chat": f"r{i}", **ROW_DIV} for i in range(6)]
        q.first, q.out, q.pool, q.futs, q.gate = None, Path(tmp_path), ThreadPoolExecutor(1), [], gate
        q.srv = types.SimpleNamespace(sample=lambda n, p, k, m, t, priority=0: [{"text": _out("one extra item in a list", p + str(j)), "ids": [1], "finish": "stop"} for j in range(k)])
        q.rw, q.difflib = _rewarder(window=32), difflib
        return q
    idle = threading.Event()
    idle.set()
    t0 = time.time()
    make(idle, 1.0)._run(0, None)   # 兜底 10 s 一条，空闲时不等
    assert time.time() - t0 < 2.0
    busy = threading.Event()
    t0 = time.time()
    make(busy, 0.01)._run(1, None)   # 一直忙：按兜底进度
    assert time.time() - t0 >= 0.45


def test_critic_aug_recall_at_precision_and_claim():
    from ragtruth.critic_aug import _point, claim_row, recall_at_precision
    ys, ps = [1, 1, 0, 1, 0, 0], [.9, .8, .7, .6, .2, .1]
    assert recall_at_precision(ys, ps, 1.0) == 2 / 3 and recall_at_precision(ys, ps, 0.75) == 1.0
    assert abs(_point(ys, ps, 1.0)["precision"] - 0.75) < 1e-9
    r = {"resp_id": "rt-1", "source": "s", "sentences": ["A.", "B.", "C."], "task_type": "QA"}
    row = claim_row(r, {"idx": 2, "op": "rewrite", "sentence": "X.", "type": "conflict"}, "v3", "hard", 0)
    assert row["completion"] == "B" and "X." in row["prompt"][1]["content"] and "B." not in row["prompt"][1]["content"]
    ins = claim_row(r, {"idx": 2, "op": "insert", "sentence": "X.", "type": "baseless"}, "v3", "all", 1)
    assert ins["completion"] == "C" and "B." in ins["prompt"][1]["content"] and "X." in ins["prompt"][1]["content"]


def test_critic_aug_claim_waits_or_takes_over(tmp_path, monkeypatch):
    import os
    import time
    import types
    from ragtruth import critic_aug as C
    monkeypatch.setattr(C, "D", tmp_path)
    args = types.SimpleNamespace(shard=[0, 2], tag="v3")
    claim = tmp_path / "claim_v3_s0"
    claim.mkdir()
    (claim / "done").touch()
    assert C.cmd_sample(args) is None          # 别人写完了：直接返回，不碰模型
    (claim / "done").unlink()
    os.utime(claim, (time.time() - 700, time.time() - 700))
    try:
        C.cmd_sample(args)                     # 别人挂了：接管，往下走到要用模型的地方
    except Exception as e:
        assert "base" in str(e) or "n_resps" in str(e) or "AttributeError" in type(e).__name__ or isinstance(e, (OSError, ImportError))


def test_critic_aug_compare_runs_end_to_end(tmp_path, monkeypatch):
    """用假的打分文件把 compare 跑一遍：配对 bootstrap、四个子集、固定精确率都走到。"""
    import types
    from ragtruth import critic_aug as C
    import ragtruth.pp_valid as pv
    monkeypatch.setattr(C, "RM", tmp_path / "rm")
    monkeypatch.setattr(C, "D", tmp_path)
    monkeypatch.setattr(pv, "load", lambda: {})
    labs = ["A", "B", "C", "A", "B", "C", "A", "A"] * 5
    for arm, lp_pos in (("rm1", -1.0), ("rm1__x", -0.5)):
        d = tmp_path / "rm" / arm / "eval"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"temperature": 1.0}))
        with open(d / "test_scores.jsonl", "w") as f:
            for k, lab in enumerate(labs):
                lp = {"A": -0.5, "B": -5.0, "C": -5.0} if lab == "A" and k % 3 else {"A": -3.0, "B": lp_pos, "C": -4.0}
                f.write(json.dumps({"id": f"r{k // 2}-S{k}", "label": lab, "lp": lp}) + chr(10))
    C.cmd_compare(types.SimpleNamespace(seed=0, boot=20, base_arm="rm1", arms=None))
    rep = json.load(open(tmp_path / "compare.json"))
    assert set(rep) == {"valid", "strict", "valid_conflict", "valid_baseless"}
    assert rep["valid"]["rm1__x"]["auc"]["ci95"][0] <= rep["valid"]["rm1__x"]["auc"]["ci95"][1]


def test_pick_replacement_keeps_label_counts():
    from ragtruth.critic_aug import pick_replacement
    cand = [({"label": "B" if k % 2 else "C", "id": f"e{k}"}, k) for k in range(8)]
    orig = lambda c: None if c[1] == 0 else {"prompt": [c[1]], "label": "A", "id": f"o{c[1]}"}   # 第 0 条是插入，没有原句
    rows = pick_replacement(cand, {"B": 2, "C": 2, "A": 5}, orig)
    labs = [r["label"] for r in rows]
    assert (labs.count("B"), labs.count("C"), labs.count("A")) == (2, 2, 5)
    assert "o0" not in {r["id"] for r in rows}
    with pytest.raises(SystemExit):
        pick_replacement(cand, {"B": 9, "A": 1}, orig)
    with pytest.raises(SystemExit):
        pick_replacement(cand, {"B": 1, "A": 9}, orig)


def test_pick_seed_keeps_groups_and_is_deterministic():
    from ragtruth.critic_aug import pick_seed
    rows = [{"group": f"g{k // 3}", "id": k} for k in range(30)]
    a = pick_seed(rows, 7, 0)
    assert 7 <= len(a) <= 9 and a == pick_seed(rows, 7, 0)
    groups = {r["group"] for r in a}
    assert len(a) == 3 * len(groups)   # 同一回答的行都在


def test_gold_rows_labels_and_length():
    from ragtruth.critic_aug import gold_rows
    picks = {"r1": {"source": "s", "sentences": ["a", "b", "c"], "task_type": "QA"}}
    final = [{"resp_id": "r1", "idx": 1, "is_error": False, "type": "none"}, {"resp_id": "r1", "idx": 2, "is_error": True, "type": "conflict"}, {"resp_id": "r1", "idx": 3, "is_error": True, "type": "baseless"}]
    rows, too = gold_rows(picks, final, 3072)
    assert [r["label"] for r in rows] == ["A", "B", "C"] and too == 0
    assert rows[1]["group"] == "r1" and rows[1]["prompt"][1]["content"].endswith("S2: b")
    assert gold_rows(picks, final, 1) == ([], 3)


def test_raw_rows_labels_and_calib():
    from ragtruth.critic_aug import raw_rows
    base = {"source": "s", "sentences": ["a", "b"], "task_type": "QA"}
    claims = [{**base, "id": "r1-S1", "response_id": "r1", "idx": "1", "label": "faithful", "type": "none"},
              {**base, "id": "r1-S2", "response_id": "r1", "idx": "2", "label": "hallucination", "type": "conflict"},
              {**base, "id": "r2-S1", "response_id": "r2", "idx": "1", "label": "hallucination", "type": "other"},
              {**base, "id": "r3-S1", "response_id": "r3", "idx": "1", "label": "faithful", "type": "none"}]
    rows, too = raw_rows(claims, {"r3"}, 3072)
    assert [r["label"] for r in rows] == ["A", "B", "C"] and too == 0
    assert rows[1]["group"] == "r1" and rows[1]["prompt"][1]["content"].endswith("S2: b")
    assert raw_rows(claims, set(), 1) == ([], 4)


def test_gold_rows_origin():
    from ragtruth.critic_aug import gold_rows
    picks = {"r": {"source": "S.", "sentences": ["A."], "task_type": "QA"}}
    final = [{"resp_id": "r", "idx": 1, "is_error": True, "type": "conflict"}]
    rows, too = gold_rows(picks, final, 10000, origin="b9")
    assert too == 0 and rows[0]["origin"] == "b9" and rows[0]["id"] == "r-b9-S1" and rows[0]["label"] == "B"


def _target(p3=0.9, p1=(0.5, 0.25, 0.25), pn=0.5, pb=None, **kw):
    from train.gen_reward import TargetRewarder
    rw = TargetRewarder(None, {"rm1": 1.0, "rm2": 1.0, "rm3": 1.0}, type_floor=0.0, **kw)
    b = 1 - p3 if pb is None else pb
    rw.option_p = lambda model, system, user, opts, priority=0: {"rm3": {"A": p3, "B": b, "C": 1 - p3 - b, "D": 0},
                                                     "rm1": dict(zip("ABC", p1)), "rm2": {"A": pn, "B": 1 - pn}}[model]
    return rw


def test_target_reward_gates_and_penalties():
    """定向版：RM3 低于门槛只给 0.1 × P；原句预算 orig_p 有错给 0；critic 判出越确定扣得越多，判不出不加分；真实性是主力。"""

    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    s = _target(p3=0.6).score(o, ROW_DIV)
    assert s["qualified"] is False and abs(s["reward"] - 0.06) < 1e-9
    s = _target().score(o, {**ROW_DIV, "orig_p": [0.8, 0.1]})
    assert s["reward"] == 0.0 and s.get("orig_flagged")
    s = _target().score(o, {**ROW_DIV, "orig_p": [0.1, 0.1]})
    assert s["qualified"] and s["tactic"] == t and s["p_orig_halluc"] == 0.1
    easy = _target(p1=(0.05, 0.9, 0.05)).score(o, {**ROW_DIV, "orig_p": [0.1, 0.1]})
    miss = _target(p1=(0.9, 0.05, 0.05)).score(o, {**ROW_DIV, "orig_p": [0.1, 0.1]})
    half = _target(p1=(0.5, 0.25, 0.25)).score(o, {**ROW_DIV, "orig_p": [0.1, 0.1]})
    assert easy["reward"] < miss["reward"] and abs(miss["reward"] - half["reward"]) < 1e-9   # 判不出不额外加分
    assert _target(pn=0.9).score(o, ROW_DIV)["real"] >= _target(pn=0.1).score(o, ROW_DIV)["real"]


def test_focus_tactic_closed():
    from train.gen_reward import OTHER_TACTIC, focus_tactic
    from ragtruth.synth import FOCUS_TACTICS
    for k, cs in FOCUS_TACTICS.items():
        for c in cs:
            assert focus_tactic(c, k) == c
    assert focus_tactic("a made-up tactic", "conflict") == OTHER_TACTIC


def test_target_reward_v2_spreads_and_ab():
    """第二版：解释不对（B）但句子错了也算真错；骗过 critic 和真实性都是对数几率绝对分、都加分。"""
    from train.gen_reward import logodds01
    assert logodds01(1e-6) == 0.0 and abs(logodds01(0.5) - 0.5) < 1e-9 and logodds01(0.05) > logodds01(0.003) > logodds01(1e-4)
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    v2 = dict(err_mode="ab", shape="v2")
    assert _target(p3=0.3, pb=0.6, **v2).score(o, row)["qualified"]          # A 0.3 + B 0.6 = 0.9 过门槛
    assert not _target(p3=0.3, pb=0.6).score(o, row)["qualified"]             # 旧版只看 A
    caught = _target(p1=(0.01, 0.98, 0.01), **v2).score(o, row)["reward"]
    hard = _target(p1=(0.2, 0.7, 0.1), **v2).score(o, row)["reward"]
    fooled = _target(p1=(0.9, 0.05, 0.05), **v2).score(o, row)["reward"]
    assert caught < hard < fooled                                           # 骗过 critic 加分，且在 P 很高时也拉得开
    assert _target(pn=0.02, **v2).score(o, row)["reward"] > _target(pn=0.001, **v2).score(o, row)["reward"]   # 自然度在低概率段也拉得开


def test_target_reward_filter():
    """过滤版：奖励只看真实性；P(真错) 不到门槛、critic 判出超过 crit_max 的一律 0。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="filter", crit_max=0.98)
    assert _target(p3=0.3, pb=0.2, **f).score(o, row)["reward"] == 0.0                     # P(真错) 0.5 < 0.7
    s = _target(p1=(0.005, 0.99, 0.005), **f).score(o, row)
    assert s["reward"] == 0.0 and s["crit_filtered"] and s["qualified"]
    lo = _target(p1=(0.1, 0.8, 0.1), pn=0.01, **f).score(o, row)["reward"]
    hi = _target(p1=(0.1, 0.8, 0.1), pn=0.3, **f).score(o, row)["reward"]
    same_crit = _target(p1=(0.9, 0.05, 0.05), pn=0.3, **f).score(o, row)["reward"]
    assert 0 < lo < hi and abs(hi - same_crit) < 1e-9                                    # 过了过滤只看真实性


def test_target_reward_soft():
    """soft：critic 不过滤，乘系数（P < 0.5 记 1，P = 1 记 floor）；骗过 critic 的同等自然度拿得更多，判出的也有自然度梯度。"""
    from train.gen_reward import crit_factor
    assert crit_factor(0.3) == 1.0 and abs(crit_factor(1.0) - 0.4) < 1e-9 and abs(crit_factor(0.75) - 0.7) < 1e-9
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="soft")
    caught_lo = _target(p1=(0.005, 0.99, 0.005), pn=0.01, **f).score(o, row)["reward"]
    caught_hi = _target(p1=(0.005, 0.99, 0.005), pn=0.3, **f).score(o, row)["reward"]
    fooled_hi = _target(p1=(0.9, 0.05, 0.05), pn=0.3, **f).score(o, row)["reward"]
    assert 0 < caught_lo < caught_hi < fooled_hi
    assert _target(p3=0.3, pb=0.2, **f).score(o, row)["reward"] == 0.0


def test_target_reward_mult_and_ens():
    from train.gen_reward import TargetRewarder
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    m = dict(err_mode="ab", shape="mult")
    assert _target(p1=(0.005, 0.99, 0.005), pn=0.3, **m).score(o, row)["reward"] < _target(p1=(0.3, 0.6, 0.1), pn=0.3, **m).score(o, row)["reward"]
    assert _target(pn=0.01, **m).score(o, row)["reward"] < _target(pn=0.2, **m).score(o, row)["reward"]
    rw = TargetRewarder(None, {"rm1": 1.0, "rm2": 1.0, "rm2b": 1.0, "rm3": 1.0}, type_floor=0.0, err_mode="ab", shape="soft", rm2_ens=["rm2b"])
    rw.option_p = lambda model, system, user, opts, priority=0: {"rm3": {"A": 0.9, "B": 0.1, "C": 0, "D": 0}, "rm1": {"A": 0.5, "B": 0.25, "C": 0.25},
                                                                  "rm2": {"A": 0.1, "B": 0.9}, "rm2b": {"A": 0.5, "B": 0.5}}[model]
    assert abs(rw.score(o, row)["p_natural"] - 0.3) < 1e-9


def test_target_reward_soft_gate():
    """连续正确性系数：P(真错) < gate_lo 记 0，到 gate 线性升到 1；门槛以下不算合格但有奖励。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="soft", gate=0.7, gate_lo=0.3)
    assert _target(p3=0.2, pb=0.05, **f).score(o, row)["reward"] == 0.0
    mid = _target(p3=0.3, pb=0.2, **f).score(o, row)
    full = _target(p3=0.9, pb=0.05, **f).score(o, row)
    assert mid["qualified"] is False and full["qualified"] and abs(mid["reward"] - 0.5 * full["reward"]) < 1e-9


def test_target_reward_min_sim():
    from train.gen_reward import edit_sim
    assert edit_sim("Bob paid the fine today.", "Ann paid the fine today.") > 0.7 and edit_sim("Bob paid the fine.", "Some oral infections cause bad breath.") < 0.4
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="soft", min_sim=0.4)
    far = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "Zebras quietly roam far northern plains."})
    near = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    assert _target(**f).score(far, row)["reward"] == 0.0 and _target(**f).score(far, row)["why_zero"] == "rewrote_whole"
    assert _target(**f).score(near, row)["reward"] > 0


def test_target_reward_boundary():
    """边界奖励：critic P(判出) 越接近 0.5 系数越高，两头都低（下限 crit_floor）。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary", crit_floor=0.05)
    r = lambda a, b, c: _target(p1=(a, b, c), pn=0.3, **f).score(o, row)["reward"]   # p_halluc = b + c
    mid, low, high, top = r(0.5, 0.3, 0.2), r(0.95, 0.03, 0.02), r(0.05, 0.9, 0.05), r(0.002, 0.996, 0.002)
    assert mid > low and mid > high and high > top and top > 0


def test_target_reward_boundary2():
    """boundary2：没过门槛给基线；明显的错（P ≥ 0.95）低于基线；边界样本高于基线。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary2", base_fail=0.1)
    fail = _target(p3=0.3, pb=0.2, **f).score(o, row)["reward"]
    r = lambda a, b, c, pn=0.05: _target(p1=(a, b, c), pn=pn, **f).score(o, row)["reward"]
    assert fail == 0.1 and r(0.5, 0.3, 0.2) > fail > r(0.02, 0.96, 0.02) and r(0.01, 0.98, 0.01, pn=0.9) == 0.0
    assert _target(**f).score("not json", row)["reward"] == 0.0


def test_target_reward_boundary3():
    from train.gen_reward import tent
    assert tent(0.01) == 0 and tent(0.95) == 0 and tent(0.3) == 1 and 0 < tent(0.1) < 1 and 0 < tent(0.6) < 1
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary3", base_fail=0.1)
    r = lambda a, b, c: _target(p1=(a, b, c), pn=0.3, **f).score(o, row)["reward"]
    assert r(0.7, 0.2, 0.1) > r(0.5, 0.3, 0.2) > 0.1 > r(0.05, 0.9, 0.05) == 0.0
    assert _target(p3=0.3, pb=0.2, **f).score(o, row)["reward"] == 0.1


def test_target_reward_boundary4():
    """a9：不造错 ≈ 0 < 明显的错 < 差一点过门槛 < 边界真错。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary4", base_fail=0.1, gate=0.7)
    no_err = _target(p3=0.01, pb=0.01, **f).score(o, row)["reward"]
    near = _target(p3=0.4, pb=0.25, **f).score(o, row)["reward"]
    obvious = _target(p1=(0.04, 0.9, 0.06), **f).score(o, row)["reward"]
    blatant = _target(p1=(0.005, 0.99, 0.005), **f).score(o, row)["reward"]
    border = _target(p1=(0.7, 0.2, 0.1), pn=0.3, **f).score(o, row)["reward"]
    assert no_err < obvious < near < border and blatant == 0.0 and abs(obvious - 0.03) < 1e-9


def test_target_reward_boundary5():
    """a10：峰值 0.5；两头（< 0.02、≥ 0.98）记 0；保底按离真错多近给。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary5", base_fail=0.1, gate=0.7)
    r = lambda a, b, c: _target(p1=(a, b, c), pn=0.3, **f).score(o, row)["reward"]
    assert r(0.5, 0.3, 0.2) > r(0.7, 0.2, 0.1) > 0 and r(0.99, 0.005, 0.005) == 0.0 and r(0.005, 0.99, 0.005) == 0.0
    assert _target(p3=0.01, pb=0.01, **f).score(o, row)["reward"] < _target(p3=0.4, pb=0.25, **f).score(o, row)["reward"] < r(0.5, 0.3, 0.2)


def test_target_reward_boundary6():
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary6", base_fail=0.1, gate=0.7)
    r = lambda a, b, c: _target(p1=(a, b, c), pn=0.3, **f).score(o, row)["reward"]
    assert r(0.5, 0.3, 0.2) > r(0.85, 0.1, 0.05) > 0 and r(0.06, 0.9, 0.04) == 0.0 and r(0.95, 0.03, 0.02) == 0.0
    assert _target(p3=0.01, pb=0.01, **f).score(o, row)["reward"] < 0.01


def test_target_reward_boundary7():
    """a12：明显的错 < 不造错 ≈ 0 < 差一点过门槛 < 边界真错。"""
    t = "attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)"
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": t, "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.1, 0.1]}
    f = dict(err_mode="ab", shape="boundary7", base_fail=0.1, gate=0.7, obvious_pen=0.1)
    obvious = _target(p1=(0.04, 0.9, 0.06), **f).score(o, row)["reward"]
    no_err = _target(p3=0.01, pb=0.01, **f).score(o, row)["reward"]
    near = _target(p3=0.4, pb=0.25, **f).score(o, row)["reward"]
    border = _target(p1=(0.5, 0.3, 0.2), pn=0.3, **f).score(o, row)["reward"]
    assert obvious == -0.1 and obvious < no_err < near < border


def _target_sev(ps, p1=(0.5, 0.25, 0.25), pn=0.5, **kw):
    from train.gen_reward import TargetRewarder
    rw = TargetRewarder(None, {"rm1": 1.0, "rm2": 1.0, "sev": 1.0}, type_floor=0.0, shape="sev", gate=0.5, **kw)
    seen = {}

    def op(model, system, user, opts, priority=0):
        seen.setdefault(model, user)
        return {"sev": dict(zip("ABCDEF", ps)), "rm1": dict(zip("ABC", p1)), "rm2": {"A": pn, "B": 1 - pn}}[model]
    rw.option_p = op
    rw.seen = seen
    return rw


def test_target_reward_sev_gate_and_shape():
    """sev：P(显然) = A + B 过门槛才合格；critic 越判不出越高；原句有错（F）记 0；不用 RM3、不用 9B 的 orig_p；sev 输入是改后回答。"""
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": "reverse a relation", "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.99, 0.99]}   # 9B 的 orig_p 即使说原句有错也不该影响 sev 版
    low = _target_sev((0.1, 0.1, 0.3, 0.3, 0.2, 0.0)).score(o, row)
    assert low["qualified"] is False and abs(low["reward"] - 0.1 * 0.2 / 0.5) < 1e-9
    orig_bad = _target_sev((0.3, 0.3, 0.0, 0.0, 0.0, 0.6)).score(o, row)
    assert orig_bad["reward"] == 0.0 and orig_bad.get("orig_flagged")
    miss = _target_sev((0.5, 0.3, 0.1, 0.1, 0.0, 0.0), p1=(0.9, 0.05, 0.05)).score(o, row)
    easy = _target_sev((0.5, 0.3, 0.1, 0.1, 0.0, 0.0), p1=(0.05, 0.9, 0.05)).score(o, row)
    assert miss["qualified"] and miss["reward"] > easy["reward"] > 0
    nat = _target_sev((0.5, 0.3, 0.1, 0.1, 0.0, 0.0), p1=(0.9, 0.05, 0.05), pn=0.9).score(o, row)
    assert nat["reward"] > miss["reward"]
    rw = _target_sev((0.5, 0.3, 0.1, 0.1, 0.0, 0.0))
    rw.score(o, row)
    assert "rm3" not in rw.seen and "[S1] A b d." in rw.seen["sev"]   # sev 看的是已经换成改写句的回答


def test_target_reward_sev_blind_bonus():
    """方案 A：critic 判不出（P < 0.5）的合格样本固定加 blind_bonus；判出的不加；没过门槛的保底按 base_fail。"""
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": "reverse a relation", "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.0, 0.0]}
    sev_ok = (0.5, 0.3, 0.1, 0.1, 0.0, 0.0)
    base = _target_sev(sev_ok, p1=(0.9, 0.05, 0.05)).score(o, row)["reward"]
    blind = _target_sev(sev_ok, p1=(0.9, 0.05, 0.05), blind_bonus=0.5).score(o, row)["reward"]
    seen = _target_sev(sev_ok, p1=(0.05, 0.9, 0.05), blind_bonus=0.5).score(o, row)
    assert abs(blind - base - 0.5) < 1e-9 and seen["reward"] < 0.1
    low = _target_sev((0.1, 0.1, 0.3, 0.3, 0.2, 0.0), base_fail=0.02, blind_bonus=0.5).score(o, row)
    assert abs(low["reward"] - 0.02 * 0.2 / 0.5) < 1e-9


def test_target_reward_sev_band_bonus():
    """加分只给 critic P 在 [0.3, 0.7] 的边界样本。"""
    o = json.dumps({"idx": 1, "type": "conflict", "tactic": "reverse a relation", "sentence": "A b d."})
    row = {**ROW_DIV, "orig_p": [0.0, 0.0]}
    sev_ok = (0.5, 0.3, 0.1, 0.1, 0.0, 0.0)
    kw = dict(blind_bonus=0.5, blind_lo=0.3, blind_thr=0.7)
    mid = _target_sev(sev_ok, p1=(0.5, 0.25, 0.25), **kw).score(o, row)["reward"]
    far = _target_sev(sev_ok, p1=(0.9, 0.05, 0.05), **kw).score(o, row)["reward"]
    hi = _target_sev(sev_ok, p1=(0.1, 0.45, 0.45), **kw).score(o, row)["reward"]
    assert mid > 0.5 and far < 0.2 and hi < 0.2   # P 0.1 / 0.9 都远离边界，不加分、难度项也小
