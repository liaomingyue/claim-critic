"""ragtruth.critic_harvest：只取合格、2B P(判出) 在区间内的；每条错句配一条原句；同一 prompt 有上限；校准 / 测试集原样拷贝（假数据）。"""
import json
import sys

from ragtruth import critic_harvest as H


def _w(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_build(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "OUT", tmp_path / "rm")
    base = tmp_path / "rm" / "base"
    _w(base / "train.jsonl", [{"id": "t0", "label": "A"}])
    _w(base / "calib.jsonl", [{"id": "c0", "label": "A"}])
    _w(base / "test.jsonl", [{"id": "x0", "label": "B"}])
    (base / "meta.json").write_text(json.dumps({"max_tokens": 3072}), encoding="utf-8")
    hv = []
    for k, (ph, pc, op) in enumerate([(0.2, 0.9, 0.1), (0.5, 0.8, 0.1), (0.05, 0.9, 0.1), (0.95, 0.9, 0.1), (0.4, 0.5, 0.1), (0.4, 0.9, 0.8), (0.3, 0.9, 0.1)]):
        hv.append({"id": f"h{k}", "prompt_id": "p0" if k < 6 else "p1", "source_id": "s", "task_type": "QA", "type": "conflict", "idx": 1,
                   "sentence": f"bad {k}", "context": [f"bad {k}", "ok"], "sentences_orig": ["good", "ok"], "source": "src",
                   "p_correct": pc, "orig_p": op, "p_halluc": ph, "p_natural": 0.5})
    _w(tmp_path / "hv.jsonl", hv)
    monkeypatch.setattr(sys, "argv", ["x", "--harvest", str(tmp_path / "hv.jsonl"), "--base", "base", "--name", "new", "--per_prompt", "1"])
    H.main()
    d = tmp_path / "rm" / "new"
    tr = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8")]
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["added_errors"] == 2 and len(tr) == 1 + 4      # p0 只取 1 条（h0 或 h1），p1 的 h6；h2/h3 区间外，h4 不合格，h5 原句有错
    assert sorted(r["label"] for r in tr if r.get("origin", "").startswith("harvest")) == ["A", "A", "B", "B"]
    assert meta["max_tokens"] == 3072 and (d / "test.jsonl").read_text(encoding="utf-8").count("x0") == 1
