"""ragtruth.rm3_v2 build：Claude 标签 → A / B / C；按原文切留出；新样本重复 dup 次；校准 / 测试集原样拷贝（假数据）。"""
import json
import sys

from ragtruth import rm3_v2 as M


def _w(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_build(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "OUT", tmp_path / "rm")
    base = tmp_path / "rm" / "base"
    _w(base / "train.jsonl", [{"id": "t0", "label": "A"}])
    _w(base / "calib.jsonl", [{"id": "c0", "label": "A"}])
    _w(base / "test.jsonl", [{"id": "x0", "label": "B"}])
    (base / "meta.json").write_text(json.dumps({"max_tokens": 4096}), encoding="utf-8")
    hv, labs = [], []
    for k in range(8):
        hv.append({"id": f"h{k}", "source_id": f"s{k}", "task_type": "QA", "type": "conflict", "idx": 1, "sentence": "bad", "original": "good",
                   "context": ["bad"], "source": "src", "evidence": "e", "reason": "r", "p_correct": 0.8, "p_halluc": 0.4})
        labs.append({"hid": f"h{k}", "is_error": k % 3 != 0, "claude_type": "conflict" if k % 2 else "baseless", "severity": "minor"})
    hv.append({"id": "other", "source_id": "s9"})
    _w(tmp_path / "hv.jsonl", hv)
    _w(tmp_path / "lab.jsonl", labs)
    monkeypatch.setattr(sys, "argv", ["x", "build", "--harvest", str(tmp_path / "hv.jsonl"), "--labels", str(tmp_path / "lab.jsonl"), "--base", "base", "--name", "new", "--hold", "0.25"])
    M.main()
    d = tmp_path / "rm" / "new"
    tr = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8")]
    te = [json.loads(l) for l in open(d / "test_synth.jsonl", encoding="utf-8")]
    assert len(te) == 2 and len(tr) == 1 + 6 * 2
    labels = {r["id"]: r["label"] for r in tr + te}
    assert labels["h1c/h0"] == "C" and labels["h1c/h1"] == "A" and labels["h1c/h2"] == "B"
    assert (d / "test.jsonl").read_text(encoding="utf-8").count("x0") == 1
