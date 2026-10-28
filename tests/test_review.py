import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from common.schema import write_jsonl
from review import store
from review.server import Data, make_handler

REFS = {"ref-1": "In adults aged 18 to 65 with type 2 diabetes, metformin reduced HbA1c by 1.1% over 24 weeks."}
POS = {"id": "p1", "split": "train", "source_group": "g1", "refs": REFS, "label": "faithful", "type": "none",
       "claim": "In adults aged 18 to 65, metformin reduced HbA1c by 1.1% over 24 weeks.",
       "evidence": 'ref-1: "metformin reduced HbA1c by 1.1% over 24 weeks"', "reason": "", "origin": "positive"}
NEG = {"id": "p1-b1-missing_condition", "split": "train", "source_group": "g1", "refs": REFS,
       "label": "hallucination", "type": "missing_condition",
       "claim": "In patients, metformin reduced HbA1c by 1.1% over 24 weeks.",
       "evidence": 'ref-1: "In adults aged 18 to 65"', "reason": "丢了年龄限定", "origin": "synthetic",
       "round": 1, "parent_id": "p1",
       "meta": {"detect_rate": 0.25, "rewrites": 1, "critic_outputs": ["<verdict>faithful</verdict>"] * 4}}


def make_cfg(tmp_path):
    p = lambda name: str(tmp_path / name)  # noqa: E731
    return {
        "label_positive": {"out": p("positives.jsonl"), "audit_out": p("pos_audit.jsonl"), "max_audit_error": 0.03},
        "synth_negative": {"out": p("synth{batch}.jsonl")},
        "mine_hard": {"out": p("hard{batch}.jsonl"), "audit_out": p("hard{batch}_audit.jsonl"),
                      "max_not_error_ratio": 0.10},
        "build_trainset": {"coldstart_out": p("coldstart.jsonl")},
        "review": {"dir": p("review")},
    }


# ---------- store ----------

def test_latest_review_wins_and_history_kept(tmp_path):
    store.save_review(tmp_path, store.make_review(NEG, "hard_batch1", "accept", [], {}))
    store.save_review(tmp_path, store.make_review(NEG, "hard_batch1", "reject", ["not_an_error"], {}))
    assert store.load_reviews(tmp_path, "hard_batch1")[NEG["id"]]["decision"] == "reject"
    assert len(store.history(tmp_path, "hard_batch1", NEG["id"])) == 2


def test_make_review_keeps_only_real_changes():
    rv = store.make_review(NEG, "hard_batch1", "accept", [], {"claim": NEG["claim"], "type": "wrong_attribution",
                                                              "round": 9})
    assert rv["edits"] == {"type": "wrong_attribution"}
    with pytest.raises(ValueError):
        store.make_review(NEG, "hard_batch1", "maybe", [], {})


def test_apply_reviews_reject_edit_stale():
    other = dict(NEG, id="x2")
    third = dict(NEG, id="x3")
    reviews = {
        NEG["id"]: store.make_review(NEG, "d", "accept", [], {"type": "wrong_attribution"}),
        "x2": store.make_review(other, "d", "reject", ["not_an_error"], {}),
        "x3": store.make_review(dict(third, claim="审核时的旧内容"), "d", "accept", [], {"reason": "新原因"}),
    }
    out, stats = store.apply_reviews([NEG, other, third, dict(NEG, id="x4")], reviews)
    assert [r["id"] for r in out] == [NEG["id"], "x3", "x4"]
    assert out[0]["type"] == "wrong_attribution" and out[0]["meta"]["human_edited"]
    assert out[1]["reason"] == NEG["reason"]  # 原样本变了，旧修改不套用
    assert stats == {"edited": 1, "rejected": 1, "stale_edits_skipped": 1, "unreviewed": 1}


def test_apply_relabel_to_faithful_clears_type():
    reviews = {NEG["id"]: store.make_review(NEG, "d", "accept", [], {"label": "faithful", "type": "none"})}
    out, _ = store.apply_reviews([NEG], reviews)
    assert (out[0]["label"], out[0]["type"]) == ("faithful", "none")


def test_validate():
    assert store.validate(NEG, POS)["errors"] == []
    assert store.validate(NEG, POS)["edit_distance"] >= 1
    assert store.validate(NEG, POS)["evidence_span"][0] == "ref-1"
    bad = store.validate(dict(NEG, evidence='ref-1: "adults between 18 and 65"', reason=""), POS)["errors"]
    assert any("逐字" in e for e in bad) and any("原因" in e for e in bad)
    assert any("编辑距离" in w for w in store.validate(dict(NEG, claim=POS["claim"]), POS)["warnings"])
    assert store.validate(dict(NEG, type="none"), POS)["errors"]  # label/type 不一致


def test_summary_threshold_on_audit_subset():
    rows = [dict(NEG, id=f"h{i}") for i in range(20)]
    reviews = {f"h{i}": store.make_review(rows[i], "d", "reject" if i < 3 else "accept",
                                          ["not_an_error"] if i < 3 else [], {}) for i in range(10)}
    s = store.summary(rows, reviews, {f"h{i}" for i in range(5)}, ("not_an_error", 0.1))
    assert (s["total"], s["reviewed"], s["rejected"]) == (20, 10, 3)
    assert s["threshold"]["ratio"] == pytest.approx(0.6) and s["threshold"]["over"] and s["threshold"]["final"]


# ---------- server ----------

@pytest.fixture()
def server(tmp_path):
    cfg = make_cfg(tmp_path)
    write_jsonl(cfg["label_positive"]["out"], [POS])
    write_jsonl(cfg["mine_hard"]["out"].format(batch=1), [NEG, {"id": "gone", "_dropped": True}])
    write_jsonl(cfg["mine_hard"]["audit_out"].format(batch=1), [NEG])
    data = Data(cfg, cfg["review"]["dir"], max_batch=1)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, "tester"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", cfg
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_flow(server):
    base, cfg = server
    with urllib.request.urlopen(base + "/") as r:
        assert b"Claim Critic" in r.read()

    _, d = call(base, "/api/datasets")
    by = {x["name"]: x for x in d["datasets"]}
    assert by["hard_batch1"]["exists"] and not by["synth_batch1"]["exists"]

    _, lst = call(base, "/api/list?dataset=hard_batch1&view=audit")
    assert [i["id"] for i in lst["items"]] == [NEG["id"]] and lst["items"][0]["status"] == "todo"

    _, det = call(base, f"/api/sample?dataset=hard_batch1&id={NEG['id']}")
    assert det["parent"]["id"] == "p1" and det["lineage"][0]["round"] == 1
    assert det["validation"]["errors"] == []

    # 改坏证据：校验报错，accept 被拒绝保存
    code, v = call(base, "/api/validate", {"dataset": "hard_batch1", "id": NEG["id"],
                                           "edits": {"evidence": 'ref-1: "not in text at all"'}})
    assert code == 200 and v["errors"]
    code, _ = call(base, "/api/review", {"dataset": "hard_batch1", "id": NEG["id"], "decision": "accept",
                                         "edits": {"evidence": 'ref-1: "not in text at all"'}})
    assert code == 422

    # 合法修改 + 标记，保存后回流
    code, res = call(base, "/api/review", {"dataset": "hard_batch1", "id": NEG["id"], "decision": "accept",
                                           "flags": ["type_wrong"], "edits": {"type": "wrong_attribution"}})
    assert code == 200 and res["status"] == "edited" and res["review"]["reviewer"] == "tester"
    _, lst = call(base, "/api/list?dataset=hard_batch1&view=edited")
    assert lst["summary"]["reviewed"] == 1 and lst["items"][0]["flags"] == ["type_wrong"]
    out = store.reviewed(cfg["review"]["dir"], "hard_batch1", [NEG])
    assert out[0]["type"] == "wrong_attribution"

    # 改成 faithful 时 type 自动置 none
    code, res = call(base, "/api/review", {"dataset": "hard_batch1", "id": NEG["id"], "decision": "accept",
                                           "edits": {"label": "faithful", "evidence": NEG["evidence"]}})
    assert code == 200 and res["review"]["edits"]["type"] == "none"

    code, _ = call(base, "/api/sample?dataset=hard_batch1&id=nope")
    assert code == 404
