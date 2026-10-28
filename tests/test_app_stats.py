import json

import pytest

from app import stats
from common.output_format import format_output
from common.schema import write_jsonl

REFS = {"ref-1": "In adults aged 18 to 65, metformin reduced HbA1c by 1.1% over 24 weeks."}


def make_cfg(tmp_path):
    p = lambda name: str(tmp_path / name)  # noqa: E731
    return {
        "split_claims": {"out": p("claims.jsonl")},
        "label_positive": {"out": p("positives.jsonl"), "audit_out": p("pos_audit.jsonl"), "max_audit_error": 0.03},
        "synth_negative": {"out": p("synth{batch}.jsonl")},
        "mine_hard": {"out": p("hard{batch}.jsonl"), "audit_out": p("hard{batch}_audit.jsonl"),
                      "max_not_error_ratio": 0.10},
        "build_trainset": {"coldstart_out": p("coldstart.jsonl")},
        "review": {"dir": p("review")},
    }


def hard(i, t, rnd, rate, rewrites):
    return {"id": f"h{i}", "label": "hallucination", "type": t, "round": rnd, "claim": "c", "refs": REFS,
            "evidence": 'ref-1: "x"', "reason": "r", "meta": {"detect_rate": rate, "rewrites": rewrites}}


def test_counts_handle_missing_and_dropped(tmp_path):
    cfg = make_cfg(tmp_path)
    write_jsonl(cfg["split_claims"]["out"], [{"id": "a"}, {"id": "b"}, {"id": "c", "_dropped": True}])
    rows = {r["stage"]: r["n"] for r in stats.pipeline_counts(cfg, max_batch=1)}
    assert rows["claims"] == 2
    assert rows["positives"] == stats.NOT_READY
    assert rows["hard_batch1"] == stats.NOT_READY


def test_hard_distribution_and_meta(tmp_path):
    cfg = make_cfg(tmp_path)
    write_jsonl(cfg["mine_hard"]["out"].format(batch=1),
                [hard(0, "missing_condition", 1, 0.0, 0), hard(1, "missing_condition", 1, 0.25, 1),
                 hard(2, "wrong_attribution", 1, 1.0, 2)])
    dist = stats.hard_distribution(cfg, max_batch=2)
    assert {(d["type"], d["n"]) for d in dist} == {("missing_condition", 2), ("wrong_attribution", 1)}
    meta = stats.hard_meta_distribution(stats.load_kept(cfg["mine_hard"]["out"].format(batch=1)))
    assert {d["bucket"]: d["n"] for d in meta["detect_rate"]} == {"0.00-0.25": 1, "0.25-0.50": 1, "0.75-1.00": 1}
    assert [d["n"] for d in meta["rewrites"]] == [1, 1, 1]


def test_audit_summary_thresholds(tmp_path):
    from review import store

    cfg = make_cfg(tmp_path)
    rdir = cfg["review"]["dir"]
    pos = [{"id": f"p{i}", "label": "faithful", "type": "none", "claim": "c", "evidence": "", "reason": ""}
           for i in range(15)]
    write_jsonl(cfg["label_positive"]["audit_out"], pos)
    for i, s in enumerate(pos[:10]):  # 审了 10 条，其中 1 条强模型判错
        store.save_review(rdir, store.make_review(s, "positives", "reject" if i == 9 else "accept",
                                                  ["judge_wrong"] if i == 9 else [], {}))
    neg = [{"id": f"h{i}", "label": "hallucination", "type": "missing_condition", "claim": "c",
            "evidence": "", "reason": "r"} for i in range(10)]
    write_jsonl(cfg["mine_hard"]["audit_out"].format(batch=1), neg)
    for i, s in enumerate(neg):
        store.save_review(rdir, store.make_review(s, "hard_batch1", "reject" if i < 2 else "accept",
                                                  ["not_an_error"] if i < 2 else [], {}))
    rows = {(r["file"], r["metric"]): r for r in stats.audit_summary(cfg, max_batch=2)}
    pa, ha = cfg["label_positive"]["audit_out"], cfg["mine_hard"]["audit_out"].format(batch=1)
    jw = rows[(pa, "judge_wrong")]
    assert jw["n"] == 15 and jw["checked"] == 10 and jw["ratio"] == pytest.approx(0.1) and jw["over"]
    assert rows[(ha, "not_an_error")]["ratio"] == pytest.approx(0.2) and rows[(ha, "not_an_error")]["over"]
    assert rows[(ha, "type_wrong")]["ratio"] == 0 and not rows[(ha, "type_wrong")]["over"]
    assert rows[(ha, "type_wrong")]["threshold"] is None


def test_filter_samples():
    rows = [hard(0, "missing_condition", 1, 0, 0), hard(1, "wrong_attribution", 2, 0, 0)]
    assert [r["id"] for r in stats.filter_samples(rows, "wrong_attribution")] == ["h1"]
    assert [r["id"] for r in stats.filter_samples(rows, round_=1)] == ["h0"]
    assert "(ref-1)" in stats.filter_samples(rows)[0]["refs"]


def test_scan_metrics_and_table(tmp_path):
    for name, f1 in (("critic-3", 0.8), ("sft-full", 0.5)):
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"model": name, "f1": f1, "f1_ci95": [f1 - 0.02, f1 + 0.02]}),
                                        encoding="utf-8")
    (tmp_path / "nomodel").mkdir()
    (tmp_path / "nomodel" / "metrics.json").write_text(json.dumps({"f1": 0.4}), encoding="utf-8")
    results = stats.scan_metrics(tmp_path)
    assert [r["model"] for r in results] == ["critic-3", "nomodel", "sft-full"]
    table = {r["model"]: r for r in stats.metrics_table(results, baseline="sft-full")}
    assert table["critic-3"]["f1_abs_vs_base"] == pytest.approx(0.3)
    assert table["critic-3"]["f1_rel_vs_base"] == pytest.approx(0.6)
    assert table["critic-3"]["f1_ci95"] == "[0.780, 0.820]"
    assert "f1_abs_vs_base" not in stats.metrics_table(results)[0]
    assert stats.scan_metrics(tmp_path / "missing") == []


def test_parse_refs():
    assert stats.parse_refs('{"ref-1": "a"}') == {"ref-1": "a"}
    assert stats.parse_refs("ref-1: a: b\n(ref-2)：中文\nnoise") == {"ref-1": "a: b", "ref-2": "中文"}
    assert stats.parse_refs("") == {}


def test_highlight_uses_normalized_text():
    refs = {"ref-1": "In  adults\n aged 18 to 65, <b>x</b> metformin works.", "ref-2": "other"}
    out = stats.highlight_refs(refs, "ref-1", "adults aged 18 to 65")
    assert "<mark>adults aged 18 to 65</mark>" in out
    assert "&lt;b&gt;" in out  # 原文被转义
    assert "<mark>" not in stats.highlight_refs(refs, "ref-2", "adults aged 18 to 65")


def test_render_claims_html():
    results = [{"claim": "A <b>", "verdict": "hallucination", "type": "missing_condition", "reason": "r",
                "evidence_ok": True, "refs": REFS, "evidence_ref": "ref-1", "evidence_text": "In adults aged 18 to 65"},
               {"claim": "B", "verdict": "faithful", "type": "none", "refs": REFS}]
    html = stats.render_claims_html(results)
    assert "#c0392b" in html and "#1e8449" in html
    assert "A &lt;b&gt;" in html and "<mark>In adults aged 18 to 65</mark>" in html
    assert stats.judge_table(results)[1]["evidence_ok"] is False
    assert "没有" in stats.render_claims_html([])


def test_reward_breakdown():
    out = format_output("hallucination", "ref-1", "metformin reduced HbA1c by 1.1%", "missing_condition", "r", "t")
    res = stats.reward_breakdown(out, "hallucination", "missing_condition", REFS)
    assert res["total"] == pytest.approx(1.6) and res["parse_error"] == "(无)"
    res = stats.reward_breakdown("garbage", "faithful", "missing_condition", REFS)  # faithful 时 type 强制 none
    assert res["total"] == 0 and res["parse_error"] != "(无)"
