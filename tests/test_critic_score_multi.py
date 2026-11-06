"""ragtruth.critic_score_multi：多任务一起打分，输出格式 / 顺序同 critic_score；每条回答先打第 1 句（假异步客户端）。"""
import asyncio
import json
import types

from ragtruth import critic_score_multi as M


def test_score_all_order_first_sentence_and_format(monkeypatch):
    log = []

    class Comp:
        async def create(self, model, messages, **kw):
            u = messages[1]["content"]
            log.append((model, u))
            await asyncio.sleep(0)
            return types.SimpleNamespace(choices=[{"model": model, "u": u}])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Comp()))
    monkeypatch.setattr(M, "option_logprobs", lambda c, opts: {"model": c["model"]})
    monkeypatch.setattr(M, "probs", lambda lp, t: {"A": 0.5, "B": 0.3 if lp["model"] == "c0" else 0.1, "C": 0.2 * t})
    monkeypatch.setattr(M, "render_input", lambda d: f"{d['source']}|{d['idx']}")
    rows = [{"id": "a", "source": "S1", "sentences": ["x", "y", "z"]}, {"id": "b", "source": "S2", "sentences": ["w"]}]
    jobs = [{"model": "c0", "temp": 1.0, "rows": rows, "out": "o0"}, {"model": "c1", "temp": 0.5, "rows": rows, "out": "o1"}]
    res = asyncio.run(M.score_all(client, jobs, concurrency=4, log_every=999))
    assert [x["id"] for x in res["o0"]] == ["a-S1", "a-S2", "a-S3", "b-S1"]
    assert abs(res["o0"][0]["p"] - 0.5) < 1e-9 and abs(res["o1"][0]["p"] - 0.2) < 1e-9 and abs(res["o1"][0]["p_baseless"] - 0.1) < 1e-9
    for m in ("c0", "c1"):   # 同一条回答：第 1 句的请求在其余句之前发出
        seq = [u for mm, u in log if mm == m and u.startswith("S1|")]
        assert seq[0] == "S1|1" and sorted(seq[1:]) == ["S1|2", "S1|3"]
    assert len(log) == 8


def test_main_skips_existing(tmp_path, monkeypatch):
    ans = tmp_path / "ans.jsonl"
    ans.write_text(json.dumps({"id": "a", "source": "S", "sentences": ["x"], "finish": "stop"}) + "\n", encoding="utf-8")
    (tmp_path / "done.json").write_text("[]", encoding="utf-8")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"model": "c0", "temp": 1.0, "answers": str(ans), "out": str(tmp_path / "done.json")}]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", "--url", "http://x", "--jobs", str(jobs)])
    M.main()   # 输出已存在：什么都不做、不连服务
