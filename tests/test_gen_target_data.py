"""ragtruth.gen_target_data：build 的过滤 / 切分，rm2data 的配平、标签和原文不跨集合（假数据，不需要服务）。"""
import json
import types

from ragtruth import gen_target_data as G


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_and_rm2data(tmp_path, monkeypatch):
    import ragtruth.rm_data as R
    monkeypatch.setattr(G, "D", tmp_path / "gt")
    monkeypatch.setattr(R, "OUT", tmp_path / "rm")
    answers, scores = [], []
    for s in range(30):
        for k in range(2):
            rid = f"gt-{s}-{k}"
            answers.append({"id": rid, "source_id": str(s), "task_type": "Summary", "model": "m", "source": f"src {s}",
                            "response": "A b. C d. E f.", "sentences": ["A b.", "C d.", "E f."], "finish": "stop"})
            for i, p in enumerate((0.1, 0.9, 0.2), 1):
                scores.append({"id": f"{rid}-S{i}", "p": p, "p_conflict": p * (0.8 if s % 2 else 0.2), "p_baseless": p * (0.2 if s % 2 else 0.8)})
    answers.append({"id": "gt-x-0", "source_id": "x", "task_type": "QA", "model": "m", "source": "s", "response": "I am unable to answer.",
                    "sentences": ["I am unable to answer."], "finish": "stop"})
    _write(G.D / "answers.jsonl", answers)
    (G.D / "critic9b_scores.json").write_text(json.dumps(scores))
    monkeypatch.setattr(G, "example_pool", lambda rows: {})
    monkeypatch.setattr(G, "read_jsonl", lambda p: [] if "train_sub35" in str(p) else [json.loads(l) for l in open(p, encoding="utf-8")])
    G.cmd_build(types.SimpleNamespace(seed=0, eval_sources=5, n_variants=2, max_chars=100000, free_size=0, ctx=0, type_split=0, example_pool=""))
    meta = json.loads((G.D / "meta.json").read_text(encoding="utf-8"))
    assert meta["answers_kept"] == 60 and meta["eval_sources"] == 5 and meta["prompts"] == 25 * 2 * 2 and meta["eval_prompts"] == 10
    row = json.loads(open(G.D / "prompts.jsonl", encoding="utf-8").readline())
    assert row["orig_p"] == [0.1, 0.9, 0.2] and "S2" in row["prompt"]
    syn = [{"id": f"gt-{s}-0#v0", "source_id": str(s), "task_type": "Summary", "type": "conflict" if s % 2 else "baseless", "idx": 1,
            "context": ["A x.", "C d.", "E f."], "source": f"src {s}", "keep": s < 20} for s in range(30)]
    _write(G.D / "synth.jsonl", syn)
    sysd = tmp_path / "sys"
    monkeypatch.setattr(G, "Path", lambda p: sysd if p == "outputs/rt/minicpm/sys" else __import__("pathlib").Path(p))
    _write(sysd / "answers.jsonl", [{"id": "mc-99-0", "source": "s99", "sentences": ["Q r.", "S t."]}])
    _write(sysd / "final.jsonl", [{"id": "mc-99-0-S2", "resp_id": "mc-99-0", "source_id": "99", "idx": 2, "task_type": "QA", "type": "conflict", "is_error": True}])
    G.cmd_rm2data(types.SimpleNamespace(inp="synth.jsonl", name="rm2v2__mcgen", seed_errors=0))
    d = tmp_path / "rm" / "rm2v2__mcgen"
    rows = {sp: [json.loads(l) for l in open(d / f"{sp}.jsonl", encoding="utf-8")] for sp in ("train", "calib", "test")}
    allr = [r for v in rows.values() for r in v]
    assert sum(r["label"] == "A" for r in allr) == sum(r["label"] == "B" for r in allr) == 20
    nat = [r for r in allr if r["label"] == "A"]
    assert {r["from"] for r in nat} == {"grpo300"}   # Claude 那句是 QA，合成里没有 QA，按任务配平时不进
    sents = {r["prompt"][1]["content"].split("[sentence]\n")[1].split("\n")[0] for r in nat}
    assert len(sents & {"A b.", "E f."}) > 0   # 自然一侧不看对错：9B 判对的句子（S1、S3）也在
    assert "planted an error" in allr[0]["prompt"][0]["content"] and "known to be unfaithful" not in allr[0]["prompt"][0]["content"]
    groups = {sp: {r["group"] for r in v} for sp, v in rows.items()}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["calib"])
    assert all(r["label"] == ("B" if r["from"] == "gen_seed_mc" else "A") for r in allr)
