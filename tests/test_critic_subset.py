"""ragtruth.critic_subset：按来源 / 来源:标签剔除训练样本，校准 / 测试集原样拷贝（假数据）。"""
import json
import sys

from ragtruth import critic_subset as S


def test_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path)
    b = tmp_path / "base"
    b.mkdir()
    rows = [{"id": "1", "origin": "human", "label": "A"}, {"id": "2", "origin": "human", "label": "B"}, {"id": "3", "origin": "synth", "label": "A"}, {"id": "4", "origin": "gold", "label": "C"}]
    (b / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for sp in ("calib.jsonl", "test.jsonl"):
        (b / sp).write_text('{"id": "x"}\n', encoding="utf-8")
    (b / "meta.json").write_text('{"max_tokens": 3072}', encoding="utf-8")
    for name, drop, want in (("a", ["human:A"], ["2", "3", "4"]), ("b", ["human", "synth"], ["4"])):
        monkeypatch.setattr(sys, "argv", ["x", "--base", "base", "--name", name, "--drop", *drop])
        S.main()
        got = [json.loads(l)["id"] for l in open(tmp_path / name / "train.jsonl", encoding="utf-8")]
        assert got == want and (tmp_path / name / "test.jsonl").exists()
        assert json.loads((tmp_path / name / "meta.json").read_text(encoding="utf-8"))["max_tokens"] == 3072


def test_keep_and_cap_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path)
    b = tmp_path / "base"
    b.mkdir()
    rows = [{"id": "h", "origin": "human", "label": "A", "group": "g0"},
            {"id": "s", "origin": "synth", "label": "B", "group": "g1"},
            {"id": "e1-err", "origin": "harvest_error", "label": "C", "group": "p1"}, {"id": "e1-orig", "origin": "harvest_original", "label": "A", "group": "p1"},
            {"id": "e2-err", "origin": "harvest_error", "label": "B", "group": "p1"}, {"id": "e2-orig", "origin": "harvest_original", "label": "A", "group": "p1"},
            {"id": "r", "origin": "refuse_yes", "label": "B", "group": "g3"}]
    (b / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for sp in ("calib.jsonl", "test.jsonl"):
        (b / sp).write_text('{"id": "x"}\n', encoding="utf-8")
    (b / "meta.json").write_text('{"max_tokens": 3072}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--base", "base", "--name", "syn", "--keep", "synth", "harvest_", "refuse_", "--cap_groups", "harvest_", "1"])
    S.main()
    got = [json.loads(l)["id"] for l in open(tmp_path / "syn" / "train.jsonl", encoding="utf-8")]
    assert got == ["s", "e1-err", "e1-orig", "r"]   # 同一个 group（prompt）下的两对只留一对
