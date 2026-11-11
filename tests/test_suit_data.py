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
