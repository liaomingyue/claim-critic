"""严重程度标注 / RM：字段合并成 6 选 1、旧标签近似、按原文切分的数据构建。"""
import json

from ragtruth import sev_label, sev_rm


def test_option_mapping_and_consistency():
    o = sev_label.option
    assert o({"original_ok": False, "severity": "clear", "explanation_ok": True}) == "F"
    assert o({"original_ok": True, "severity": "clear", "explanation_ok": True}) == "A"
    assert o({"original_ok": True, "severity": "clear", "explanation_ok": False}) == "B"
    assert o({"original_ok": True, "severity": "minor", "explanation_ok": True}) == "C"
    assert o({"original_ok": True, "severity": "minor", "explanation_ok": False}) == "D"
    assert o({"original_ok": True, "severity": "none", "explanation_ok": "n/a"}) == "E"
    assert o({"original_ok": True, "severity": "none", "explanation_ok": True}) is None      # 没错却说解释对：自相矛盾
    assert o({"original_ok": True, "severity": "clear", "explanation_ok": "n/a"}) is None


def test_family_and_old_option():
    assert sev_label.family("attribute a statement to the wrong person") == "attribution"
    assert sev_label.family("reverse a relation the source states") == "contradiction"
    assert sev_label.family("generalize what one part says") == "other"
    assert sev_rm.old_option({"is_error": False}, "conflict") == "E"
    assert sev_rm.old_option({"is_error": True, "severity": "clear", "claude_type": "conflict"}, "conflict") == "A"
    assert sev_rm.old_option({"is_error": True, "severity": "minor", "claude_type": "baseless"}, "conflict") == "D"


def _hv(i: int, src: int) -> dict:
    return {"id": f"h{i}", "source_id": str(src), "task_type": "QA", "source": f"source {src}", "context": ["a.", "edited b."], "idx": 2,
            "original": "b.", "sentence": "edited b.", "type": "conflict", "evidence": "b", "reason": "r", "p_correct": "0.8", "p_halluc": 0.3}


def test_build_splits_by_source_and_mixes_rm3(tmp_path, monkeypatch):
    out = tmp_path / "rm"
    (out / "rm3__mc").mkdir(parents=True)
    (out / "rm3__mc" / "meta.json").write_text(json.dumps({"max_tokens": 4096}))
    (out / "rm3__mc" / "train.jsonl").write_text(json.dumps({"id": "old", "label": "A"}) + "\n")
    monkeypatch.setattr(sev_rm, "OUT", out)
    hv = [_hv(i, i // 2) for i in range(40)]
    (tmp_path / "hv.jsonl").write_text("".join(json.dumps(x) + "\n" for x in hv))
    labs = [{"hid": f"h{i}", "option": "ABCDEF"[i % 6], "cell": "gm_mid", "fam": "other"} for i in range(40)]
    (tmp_path / "lab.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labs))
    (tmp_path / "old.jsonl").write_text("")

    class A:
        labels, old_labels, harvest, base, name, frac, dup, seed = (str(tmp_path / "lab.jsonl"), str(tmp_path / "old.jsonl"), str(tmp_path / "hv.jsonl"),
                                                                    "rm3__mc", "sev__t", 1.0, 2, 0)
    sev_rm.cmd_build(A)
    d = out / "sev__t"
    sp = {s: [json.loads(l) for l in open(d / f"{s}.jsonl")] for s in ("train", "calib", "test")}
    groups = {s: {r["group"] for r in rs if r["id"] != "old"} for s, rs in sp.items()}
    assert not (groups["test"] & groups["train"]) and not (groups["calib"] & groups["train"])
    assert sum(r["id"] == "old" for r in sp["train"]) == 1                      # RM3 旧数据原样进训练
    new_train = [r for r in sp["train"] if r["id"] != "old"]
    assert len(new_train) == 2 * len({r["id"] for r in new_train})              # 新样本 ×dup
    assert all(r["prompt"][0]["content"] == sev_rm.SEV_SYSTEM for r in new_train)
