"""ragtruth.accept_more label_batch：同一原文多条回答一次标，按 id 对齐；单条不合格只丢这一条，再单条重标补齐（假 LLM）。"""
import json
import types

from ragtruth import accept_more as M


def _ans(rid, sents):
    return {"id": rid, "source_id": "s1", "task_type": "QA", "model": "m", "source": "SRC", "response": " ".join(sents), "sentences": sents, "finish": "stop", "bad": {}}


def _lab(n, err_idx=()):
    return [{"idx": k, "is_error": k in err_idx, "type": "conflict" if k in err_idx else "none", "mechanism": "wrong_attribution" if k in err_idx else "none",
             "severity": "clear" if k in err_idx else "none", "evidence": "", "note": ""} for k in range(1, n + 1)]


def test_label_batch_aligns_and_retries(tmp_path, monkeypatch):
    rows = [_ans("a", ["x.", "y."]), _ans("b", ["z."]), _ans("c", ["u.", "v."])]
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    calls = []

    class FakeLLM:
        def __init__(self, cfg):
            pass

        def chat_one(self, msgs):
            calls.append(msgs[1]["content"])
            if '<response id="a">' in msgs[1]["content"]:   # 批量：a 对、b 句数不对、c 漏回
                return json.dumps({"responses": [{"id": "a", "sentences": _lab(2, (2,))}, {"id": "b", "sentences": _lab(2)}]})
            return json.dumps({"sentences": _lab(1) if "[S2]" not in msgs[1]["content"] else _lab(2)})   # 单条重标

    monkeypatch.setattr(M, "LLM", FakeLLM)
    a = types.SimpleNamespace(dir=str(tmp_path), max_per_call=3, max_calls=5, max_single=5, claude="x", effort="high", concurrency=1)
    M.cmd_label_batch(a)
    fin = [json.loads(l) for l in open(tmp_path / "final.jsonl", encoding="utf-8")]
    assert {x["resp_id"] for x in fin} == {"a", "b", "c"} and len(fin) == 5
    assert [x["severity"] for x in fin if x["resp_id"] == "a"] == ["none", "clear"]
    assert sum('<response id=' in c for c in calls) == 1 and len(calls) == 3   # 1 次批量 + b、c 各 1 次单条
    assert "SRC" in calls[0] and calls[0].count("<source>") == 1               # 原文只发一遍
