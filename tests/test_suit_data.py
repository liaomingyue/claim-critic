"""合适度数据：多标签构造（核心集 0/1；测试集按原文汇总成出现比例，没有错的回答也进分母）。"""
from ragtruth import suit_data as sd


def test_from_core_multilabel():
    core = [{"split": "train", "source_id": "1", "task_type": "QA", "source": "q",
             "labels": {"errors": [{"mechanism": "number_or_date"}, {"mechanism": "wrong_attribution"}, {"mechanism": "other"}]}},
            {"split": "holdout", "source_id": "2", "task_type": "QA", "source": "q", "labels": {"errors": []}}]
    tr = sd.from_core(core, "train")
    assert len(tr) == 1 and tr[0]["labels"]["number_or_date"] == 1 and tr[0]["labels"]["wrong_attribution"] == 1
    assert sum(tr[0]["labels"].values()) == 2 and "other" not in tr[0]["labels"]
    assert sum(sd.from_core(core, "holdout")[0]["labels"].values()) == 0


def test_sentence_labels_soft_rate_counts_clean_responses():
    rows = [{"source_id": "9", "resp_id": "a", "task_type": "Summary", "is_error": True, "mechanism": "number_or_date"},
            {"source_id": "9", "resp_id": "a", "task_type": "Summary", "is_error": True, "mechanism": "number_or_date"},
            {"source_id": "9", "resp_id": "b", "task_type": "Summary", "is_error": False, "mechanism": "none"},
            {"source_id": "9", "resp_id": "c", "task_type": "Summary", "is_error": True, "mechanism": "contradicts_source"},
            {"source_id": "8", "resp_id": "d", "task_type": "QA", "is_error": True, "mechanism": "number_or_date"}]
    out = sd.from_sentence_labels(rows, {"9": "src9"}, "t")   # 原文 8 没有提示文本，跳过
    assert len(out) == 1 and out[0]["n_resp"] == 3
    assert out[0]["soft"]["number_or_date"] == round(1 / 3, 4) and out[0]["soft"]["contradicts_source"] == round(1 / 3, 4)
    assert out[0]["labels"]["number_or_date"] == 1 and out[0]["labels"]["wrong_attribution"] == 0


def test_merged_mechanisms():
    core = [{"split": "train", "source_id": "1", "task_type": "QA", "source": "q",
             "labels": {"errors": [{"mechanism": "added_specific_fact"}, {"mechanism": "unsupported_inference"}]}}]
    lab = sd.from_core(core, "train")[0]["labels"]
    assert lab["unsupported_addition"] == 1 and sum(lab.values()) == 1 and len(sd.MECHS) == 8


def test_topn_lift_uses_soft_labels():
    rows = [{"id": f"r{i}", "labels": {m: int(i < 10) for m in sd.MECHS}, "soft": {m: (0.9 if i < 10 else 0.1) for m in sd.MECHS}} for i in range(40)]
    pred = {m: [1.0 - i / 40 for i in range(40)] for m in sd.MECHS}   # 完美排序
    ev = sd.evaluate(pred, rows)
    m0 = ev["by_mech"][sd.MECHS[0]]
    assert m0["top10"] == 0.9 and m0["base"] == 0.3 and m0["lift10"] == 3.0 and ev["mean_lift10"] == 3.0
