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


def test_ids_type_override(tmp_path, monkeypatch):
    """--ids 白名单可带类型（id<TAB>baseless）覆盖生成器给的类型；不带类型的保持原样；白名单外的不取。"""
    monkeypatch.setattr(H, "OUT", tmp_path / "rm")
    base = tmp_path / "rm" / "base"
    _w(base / "train.jsonl", [])
    _w(base / "calib.jsonl", [])
    _w(base / "test.jsonl", [])
    (base / "meta.json").write_text("{}", encoding="utf-8")
    hv = [{"id": f"h{k}", "prompt_id": f"p{k}", "source_id": "s", "task_type": "QA", "type": "conflict", "idx": 1, "sentence": f"bad {k}",
           "context": [f"bad {k}"], "sentences_orig": ["good"], "source": "src", "p_correct": 0.9, "orig_p": 0.1, "p_halluc": 0.9, "p_natural": 0.5} for k in range(3)]
    _w(tmp_path / "hv.jsonl", hv)
    (tmp_path / "ids.txt").write_text("h0\tbaseless\nh1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--harvest", str(tmp_path / "hv.jsonl"), "--base", "base", "--name", "new", "--ids", str(tmp_path / "ids.txt")])
    H.main()
    tr = {r["id"]: r["label"] for r in map(json.loads, open(tmp_path / "rm" / "new" / "train.jsonl", encoding="utf-8"))}
    assert tr == {"h0-err": "C", "h0-orig": "A", "h1-err": "B", "h1-orig": "A"}


def test_flip_gate(tmp_path, monkeypatch):
    """9B 翻转门（--scores9b）：只留 原句 9B≤lo_clean 且 错句 9B≥hi_err（改写前无幻觉、改写后有幻觉）；原句脏 / 错句不够错 / 缺 9B 分的都丢。"""
    monkeypatch.setattr(H, "OUT", tmp_path / "rm")
    base = tmp_path / "rm" / "base"
    for sp in ("train", "calib", "test"):
        _w(base / f"{sp}.jsonl", [])
    (base / "meta.json").write_text("{}", encoding="utf-8")
    # 五条都过现有门（p_correct/orig_p/区间/p_natural），只有 9B 翻转门区分
    hv = [{"id": f"h{k}", "prompt_id": f"p{k}", "source_id": "s", "task_type": "QA", "type": "conflict", "idx": 1,
           "sentence": f"bad {k}", "context": [f"bad {k}"], "sentences_orig": ["good"], "source": "src",
           "p_correct": 0.9, "orig_p": 0.1, "p_halluc": 0.4, "p_natural": 0.9} for k in range(5)]
    _w(tmp_path / "hv.jsonl", hv)
    sc = [{"id": "h0", "orig_9b": 0.05, "err_9b": 0.9},   # 翻转 ✓
          {"id": "h1", "orig_9b": 0.6, "err_9b": 0.9},    # 原句本来就脏 ✗
          {"id": "h2", "orig_9b": 0.05, "err_9b": 0.3},   # 改写后没变错 ✗
          {"id": "h3", "orig_9b": 0.02, "err_9b": 0.8}]   # 翻转 ✓；h4 缺 9B 分 → 丢
    _w(tmp_path / "sc9b.jsonl", sc)
    monkeypatch.setattr(sys, "argv", ["x", "--harvest", str(tmp_path / "hv.jsonl"), "--base", "base", "--name", "new",
                                      "--lo", "0", "--hi", "1.01", "--per_prompt", "1",
                                      "--scores9b", str(tmp_path / "sc9b.jsonl"), "--lo_clean", "0.1", "--hi_err", "0.7"])
    H.main()
    tr = {r["id"]: r["label"] for r in map(json.loads, open(tmp_path / "rm" / "new" / "train.jsonl", encoding="utf-8"))}
    assert tr == {"h0-err": "B", "h0-orig": "A", "h3-err": "B", "h3-orig": "A"}
    m = json.loads((tmp_path / "rm" / "new" / "meta.json").read_text(encoding="utf-8"))
    assert m["added_errors"] == 2 and m["scores9b"] and m["lo_clean"] == 0.1 and m["hi_err"] == 0.7


def test_per_source_and_top_sources(tmp_path, monkeypatch):
    """多样性对照：per_source 限每篇原文条数；top_sources 只用合格样本最多的前 M 篇。"""
    monkeypatch.setattr(H, "OUT", tmp_path / "rm")
    base = tmp_path / "rm" / "base"
    _w(base / "train.jsonl", [])
    _w(base / "calib.jsonl", [])
    _w(base / "test.jsonl", [])
    (base / "meta.json").write_text("{}", encoding="utf-8")
    hv = [{"id": f"h{s}{k}", "prompt_id": f"p{s}{k}", "source_id": f"s{s}", "task_type": "QA", "type": "conflict", "idx": 1, "sentence": f"bad {s}{k}",
           "context": ["x"], "sentences_orig": ["y"], "source": "src", "p_correct": 0.9, "orig_p": 0.1, "p_halluc": 0.5, "p_natural": 0.5}
          for s, n in ((0, 5), (1, 3), (2, 1)) for k in range(n)]
    _w(tmp_path / "hv.jsonl", hv)
    for name, extra, want in (("a", ["--per_source", "1"], (3, 3)), ("b", ["--top_sources", "1"], (5, 1))):
        monkeypatch.setattr(sys, "argv", ["x", "--harvest", str(tmp_path / "hv.jsonl"), "--base", "base", "--name", name, "--lo", "0", "--hi", "1.01", *extra])
        H.main()
        m = json.loads((tmp_path / "rm" / name / "meta.json").read_text(encoding="utf-8"))
        assert (m["added_errors"], m["sources"]) == want
