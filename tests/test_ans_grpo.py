"""train.ans_reward / train.ans_grpo / ragtruth.ans_data 的单元测试（不需要 GPU / 服务）。"""
from ragtruth.ans_data import filter_ref
from train.ans_grpo import Producer, TripWire, batch_stats, by_source, eval_metrics, first_from_samples, paired_delta
from train.ans_reward import AnswerRewarder, check, combine, copy_frac, guarded, over_limit, sentences

SRC = "Summarize the following news within 10 words:\nThe cafe opens at nine in the morning every single day of the week.\n\noutput:"
ROW = {"task_type": "Summary", "source": SRC, "ref_chars": 40}


def test_check_rules():
    assert check("The cafe serves customers from nine each day.", "stop", ROW) == "ok"
    assert check("", "stop", ROW) == "empty"
    assert check("The cafe serves customers from nine each day.", "length", ROW) == "truncated"
    assert check("Unable to answer based on given passages.", "stop", ROW) == "refusal"
    assert check("Short one.", "stop", ROW) == "length"
    assert check("x " * 60, "stop", ROW) == "length"
    assert check("The cafe opens at nine in the morning every single day.", "stop", {**ROW, "ref_chars": 50}) == "copy"


def test_copy_frac():
    assert copy_frac("the cafe opens at nine in", SRC) == 1.0
    assert copy_frac("a different sentence with no overlap at all", SRC) == 0.0
    assert copy_frac("short", SRC) == 0.0


def test_combine_mean_and_max():
    assert abs(combine([0.0, 0.0]) - 1.0) < 1e-9
    assert abs(combine([1.0, 0.0]) - 0.25) < 1e-9
    # 多写安全句稀释平均 P，但最大 P 不变：不能靠加句子拿满分
    assert combine([0.8] + [0.0] * 9) < 0.6


def test_over_limit():
    assert over_limit("one two three four five six seven eight nine ten eleven", SRC)
    assert not over_limit("one two three", SRC)
    assert not over_limit("one " * 50, "Briefly answer the following question")


def test_filter_ref_drops_prompts_that_usually_refuse():
    rows = [{"id": "a", "task_type": "QA"}, {"id": "b", "task_type": "QA"}]
    ok = {"text": "An answer.", "finish": "stop"}
    no = {"text": "Unable to answer based on given passages.", "finish": "stop"}
    kept, meta = filter_ref(rows, [[ok, ok, ok, no], [ok, no, no, no]], 4)
    assert [r["id"] for r in kept] == ["a"] and kept[0]["ref_chars"] == len("An answer.")
    assert meta["dropped"] == {"QA": 1}


def _sample(why="ok", p=0.2, chars=100, task="QA"):
    return {"why": why, "reward": 0.5 if why == "ok" else 0.0, "p_mean": p, "p_max": p + 0.1, "chars": chars, "n_sent": 3, "copy": 0.2, "refusal": why == "refusal", "over_limit": False, "finish": "stop", "ids": [1, 2], "task_type": task}


def test_batch_stats():
    st = batch_stats({"samples": [_sample(), _sample(why="refusal", chars=40), _sample(task="Summary", p=0.6)], "t_gen": 1.0, "t_rm": 2.0})
    assert st["ok_rate"] == round(2 / 3, 4) and st["answer_flag"] == 0.5 and st["why"] == {"ok": 2, "refusal": 1}
    assert st["p_mean_qa"] == 0.2 and st["p_mean_summary"] == 0.6


def _rec(**kw):
    return {"chars_qa": 100, "chars_summary": 200, "copy": 0.2, "refusal": 0.1, "ok_rate": 0.9, "kl": 0.001, **kw}


def test_tripwire():
    t = TripWire()
    assert not any(t.update(_rec()) for _ in range(60))
    t = TripWire()
    res = [t.update(_rec()) for _ in range(30)] + [t.update(_rec(chars_qa=50, refusal=0.3)) for _ in range(30)]
    assert any("chars_qa" in x for r in res for x in r) and any("refusal" in x for r in res for x in r)


def test_eval_metrics_and_paired_delta():
    rows = [{"task_type": "QA"}, {"task_type": "Summary"}]
    d = {"why": "ok", "refusal": False, "over_limit": False, "copy": 0.1, "chars": 50, "n_sent": 2}
    g = {"finish": "stop"}
    rec = eval_metrics(rows, [[g], [g]], [[{**d, "ps": [0.8, 0.0]}], [{**d, "ps": [0.1, 0.1]}]], [[[0.4, 0.0]], [[0.2, 0.2]]])
    assert rec["answer_flag"] == 0.5 and abs(rec["p_mean"] - 0.25) < 1e-4 and rec["shadow_p_mean"] == 0.2
    assert paired_delta({"a": 0.2, "b": 0.2}, {"a": 0.5, "b": 0.5}) == [-0.3, -0.3]


def test_sentences():
    assert sentences("One here. Two there.") == ["One here.", "Two there."]


def test_producer_failure_reaches_main_thread(monkeypatch):
    import queue
    import threading

    import pytest
    p = Producer.__new__(Producer)
    p.stop, p.dead, p.q = False, None, queue.Queue(maxsize=1)
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("vllm down")

    p.make_batch = boom
    monkeypatch.setattr("train.ans_grpo.time.sleep", lambda s: None)
    monkeypatch.setattr(p.q, "get", lambda timeout: (_ for _ in ()).throw(queue.Empty()))
    t = threading.Thread(target=p.run)
    t.start()
    t.join(5)
    assert len(calls) == 5 and isinstance(p.dead, ValueError)
    with pytest.raises(RuntimeError, match="producer died"):
        p.get()


def test_first_from_samples_uses_first_k(tmp_path):
    import json
    f = tmp_path / "s0.jsonl"
    f.write_text(json.dumps({"id": "a", "scores": [{"ps": [0.2]}, {"ps": [0.4]}, {"ps": [1.0]}], "shadow": [[0.1], [0.3], [1.0]]}) + "\n")
    by = first_from_samples(f, 2)
    assert abs(by["p_mean"]["a"] - 0.3) < 1e-9 and by["answer_flag"]["a"] == 0.0 and abs(by["shadow_p_mean"]["a"] - 0.2) < 1e-9
    assert by == by_source([{"source_id": "a"}], [[{"ps": [0.2]}, {"ps": [0.4]}]], [[[0.1], [0.3]]])


def test_new_rules_unfinished_and_language():
    r = {**ROW, "ref_chars": 45}
    assert check("The cafe serves customers from nine each day.", "stop", r) == "ok"
    assert check("The cafe serves customers from nine each day and", "stop", r) == "unfinished"
    assert check("The cafe serves customers 从九点开始 each day.", "stop", r) == "language"
    assert check("The cafe serves customers 从九点开始 each day.", "stop", {**r, "source": SRC + "中文"}) == "ok"


def test_guard_only_penalizes():
    assert guarded(0.6, None, 0.5) == 0.6
    assert guarded(0.6, 0.9, 0.5) == 0.6   # 高分不奖
    assert guarded(0.6, 0.5, 0.5) == 0.6
    assert abs(guarded(0.6, 0.1, 0.5) - 0.4) < 1e-9
    assert guarded(0.1, 0.0, 0.5) == 0.0   # 不能比硬规则的 0 还低


def test_p_sat_needs_model_and_reference():
    rw = AnswerRewarder(None, "c", 1.0, None)
    assert rw.p_sat({"source": "s", "ref_text": "r"}, "a") is None
    rw = AnswerRewarder(None, "c", 1.0, None, "sat", "pair", 1.0, 0.5)
    assert rw.p_sat({"source": "s", "ref_text": None}, "a") is None


def test_policy_prompt_adds_system_only_for_qa():
    from ragtruth.ans_data import SYSTEM_QA, policy_prompt

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            assert kw["enable_thinking"] is False
            return " | ".join(f"{m['role']}:{m['content']}" for m in msgs)

    assert policy_prompt(Tok(), {"source": "q", "task_type": "QA"}, True) == f"system:{SYSTEM_QA} | user:q"
    assert policy_prompt(Tok(), {"source": "s", "task_type": "Summary"}, True) == "user:s"
    assert policy_prompt(Tok(), {"source": "q", "task_type": "QA"}, False) == "user:q"


def test_eval_metrics_without_shadow():
    rows = [{"task_type": "QA", "source_id": "a"}]
    d = {"why": "ok", "refusal": False, "over_limit": False, "copy": 0.1, "chars": 50, "n_sent": 2, "ps": [0.8, 0.0]}
    rec = eval_metrics(rows, [[{"finish": "stop"}]], [[d]], [[None]])
    assert "shadow_p_mean" not in rec and rec["answer_flag"] == 1.0
    assert set(by_source(rows, [[d]], [[None]])) == {"p_mean", "answer_flag"}


def test_filter_ref_keep_refusal():
    rows = [{"id": "b", "task_type": "QA"}]
    no = {"text": "Unable to answer based on given passages.", "finish": "stop"}
    kept, meta = filter_ref(rows, [[no, no, no, no]], 4, keep_refusal=True)
    assert len(kept) == 1 and kept[0]["ref_chars"] is None and meta["dropped"] == {}


def test_refusal_reward_uses_refuse_model():
    rw = AnswerRewarder(None, "c", 1.0, None, refuse_model="refuse", refuse_temp=1.0, refuse_w=0.4)
    rw.p_refuse = lambda row, priority=0: 0.5
    rw.group_pool = None
    row = {"task_type": "QA", "source": "q", "ref_chars": None}
    out = rw.score_group(row, [{"text": "Unable to answer based on given passages.", "finish": "stop"}])
    assert out[0]["why"] == "refusal" and abs(out[0]["reward"] - 0.2) < 1e-9 and out[0]["p_refuse"] == 0.5
    rw0 = AnswerRewarder(None, "c", 1.0, None)
    assert rw0.score_group(row, [{"text": "Unable to answer based on given passages.", "finish": "stop"}])[0]["reward"] == 0.0
