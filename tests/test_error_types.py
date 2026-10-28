"""五类错误相关的新逻辑：术语词典、对照文献选取、引用错误（挂载 / 无中生有）的证据与定位规则、规则合成。"""
import random

import pytest

from common.context import select_context
from common.glossary import Glossary, Term
from common.output_format import format_output, parse_output
from common.prompts import TYPE_PRIORITY, critic_messages
from common.schema import ERROR_TYPES, Sample, is_unsourced
from data.build_trainset import balance_types, drop_citations
from data.synth_negative import (applicable_recipes, applicable_types, assign_types, sentence_around, synthesize_one,
                                 validate)
from eval.metrics import per_sample
from review import store
from train.reward import score_components

G = Glossary([Term("neoadjuvant chemotherapy", "新辅助化疗", ("辅助化疗",)),
              Term("adjuvant chemotherapy", "辅助化疗", ("新辅助化疗",)),
              Term("HbA1c", "糖化血红蛋白", ("血红蛋白", "空腹血糖")),
              Term("type 2 diabetes", "2型糖尿病", ("1型糖尿病",))])

GROUP = {
    "ref-1": "In 412 adults aged 18 to 65 with type 2 diabetes, metformin reduced HbA1c by 1.1% over 24 weeks.",
    "ref-2": "Gastrointestinal adverse events occurred in 21% of the metformin group.",
    "ref-3": "Neoadjuvant chemotherapy achieved a pathological complete response in 32% of women.",
}
POS = Sample(id="p1", split="train", source_group="g1", refs=select_context(
    "在 2型糖尿病成人中，二甲双胍使糖化血红蛋白下降 1.1%。", ["ref-1"], GROUP, k=2), cited=["ref-1"],
    claim="在 2型糖尿病成人中，二甲双胍使糖化血红蛋白下降 1.1%。", label="faithful", type="none",
    evidence='ref-1: "metformin reduced HbA1c by 1.1% over 24 weeks"').to_dict()


# ---------- 词典 ----------

def test_longer_term_masks_shorter():
    assert G.zh_in("新辅助化疗", "患者接受新辅助化疗后")
    assert not G.zh_in("辅助化疗", "患者接受新辅助化疗后")  # 落在更长的词典词里，不算出现
    assert G.zh_in("2型糖尿病", "在 2 型糖尿病患者中")       # 字间空白不影响


def test_mistranslation_check():
    refs = {"ref-1": "Neoadjuvant chemotherapy was given."}
    hits = G.mistranslations(refs, "患者接受了辅助化疗。")
    assert [(t.en, w) for t, w in hits] == [("neoadjuvant chemotherapy", "辅助化疗")]
    assert G.mistranslations(refs, "患者接受了新辅助化疗。") == []


def test_glossary_hits_do_not_leak_label():
    """正样本和 term_error 负样本拿到同样的词条，术语表不能暴露答案。"""
    pos = [t.en for t in G.hits(GROUP, "二甲双胍使糖化血红蛋白下降")]
    neg = [t.en for t in G.hits(GROUP, "二甲双胍使空腹血糖下降")]
    assert pos == neg and pos[0] == "HbA1c"


# ---------- 上下文 ----------

def test_select_context_cited_first_then_relevant():
    refs = select_context("二甲双胍组胃肠道不良事件为 21%。", ["ref-3"], GROUP, k=1, glossary=G)
    assert list(refs) == ["ref-3", "ref-2"]  # 21 这个数字把 ref-2 排到前面


def test_prompt_marks_cited_and_glossary():
    text = critic_messages(POS)[1]["content"]
    assert "(ref-1) [被引用]" in text and "[对照]" in text
    assert "HbA1c = 糖化血红蛋白" in text and text.rstrip().endswith("ref-1")
    assert set(TYPE_PRIORITY) == set(ERROR_TYPES) and len(ERROR_TYPES) == 5


# ---------- 输出格式与 reward ----------

def unsourced_out(evidence_none=True, type_="citation_error"):
    return format_output("hallucination", None if evidence_none else "ref-2",
                         None if evidence_none else "Gastrointestinal adverse events occurred", type_, "编造了死亡率",
                         think="所有文献都没有死亡率")


def test_evidence_none_only_for_citation_error():
    assert parse_output(unsourced_out()).format_ok
    assert parse_output(unsourced_out(type_="missing_condition")).error == "evidence none only for citation_error"
    assert parse_output(unsourced_out(False)).format_ok  # 引用错误也可以引原文（挂载错误）


def test_citation_error_loc_depends_on_gold_evidence():
    """真值是无中生有时必须写 none；真值是挂载错误时必须引未被引用的那篇。不能一律写 none 拿分。"""
    c = score_components(unsourced_out(), "hallucination", "citation_error", GROUP, ["ref-1"], gold_evidence="none")
    assert (c["label"], c["format"], c["loc"]) == (1, 1, 1)
    swap_gold = 'ref-2: "Gastrointestinal adverse events occurred in 21%"'
    assert score_components(unsourced_out(), "hallucination", "citation_error", GROUP, ["ref-1"],
                            gold_evidence=swap_gold)["loc"] == 0
    assert score_components(unsourced_out(False), "hallucination", "citation_error", GROUP, ["ref-1"],
                            gold_evidence=swap_gold)["loc"] == 1
    assert score_components(unsourced_out(False), "hallucination", "citation_error", GROUP, ["ref-1"],
                            gold_evidence="none")["loc"] == 0


def test_verl_ground_truth_carries_evidence():
    import json
    from train.grpo import to_verl_row
    from train.reward import compute_score
    gold = dict(POS, label="hallucination", type="citation_error", evidence="none", reason="x")
    row = to_verl_row(gold)
    assert json.loads(row["reward_model"]["ground_truth"])["evidence"] == "none"
    assert compute_score("claim_critic", unsourced_out(), row["reward_model"]["ground_truth"])["loc_ok"] == 1


def test_citation_error_evidence_must_come_from_uncited_ref():
    ok = format_output("hallucination", "ref-1", "metformin reduced HbA1c by 1.1%", "citation_error", "挂错", think="t")
    c = score_components(ok, "hallucination", "citation_error", GROUP, cited=["ref-2"])
    assert c["loc"] == 1
    c = score_components(ok, "hallucination", "citation_error", GROUP, cited=["ref-1"])
    assert c["loc"] == 0  # 证据引的就是被引用的那篇，说明没找到真正的出处


def test_faithful_evidence_must_come_from_cited_ref():
    out = format_output("faithful", "ref-1", "metformin reduced HbA1c by 1.1%", "none", think="t")
    assert score_components(out, "faithful", "none", GROUP, cited=["ref-1"])["loc"] == 1
    assert score_components(out, "faithful", "none", GROUP, cited=["ref-2"])["loc"] == 0
    assert score_components(out, "faithful", "none", GROUP, cited=[])["loc"] == 1  # 没标引用，任一文献都行


def test_metrics_loc_for_unsourced():
    gold = dict(POS, label="hallucination", type="citation_error", evidence="none")
    assert per_sample(gold, unsourced_out())["loc_ok"] is True
    assert is_unsourced(gold) and not is_unsourced(dict(gold, evidence=POS["evidence"]))


# ---------- 合成 ----------

def test_validate_unsourced():
    parsed = {"claim": POS["claim"] + "死亡率降低 18%。", "evidence": "none", "type": "citation_error", "reason": "编造"}
    assert validate(parsed, POS["claim"], POS["refs"], "citation_error", recipe="unsourced") == ""
    assert validate(dict(parsed, evidence='ref-1: "metformin reduced HbA1c"'), POS["claim"], POS["refs"],
                    "citation_error", recipe="unsourced") == "unsourced evidence must be none"


def test_rule_citation_error_swap():
    s = synthesize_one(None, POS, "citation_error", {}, random.Random(0), {}, recipe="swap")
    assert s["claim"] == POS["claim"] and s["cited"] != ["ref-1"]
    assert list(s["refs"])[0] == s["cited"][0]  # 被引用的排第一，和正样本一致，不留位置捷径
    assert s["evidence"] == POS["evidence"] and s["parent_id"] == "p1" and s["meta"]["recipe"] == "swap"


def test_rule_term_error():
    s = synthesize_one(None, POS, "term_error", {}, random.Random(0), {})  # term_error 缺省就是规则方法
    assert s["claim"] != POS["claim"] and s["type"] == "term_error"
    ref_id, _ = s["evidence"].split(":", 1)
    assert ref_id == "ref-1"
    assert store.validate(s, POS)["errors"] == []


def test_sentence_around():
    text = "First sentence. In adults, HbA1c fell by 1.1%. Last one."
    i = text.index("HbA1c")
    assert sentence_around(text, i, i + 5) == "In adults, HbA1c fell by 1.1%."


def test_applicable_and_assign_types():
    assert applicable_recipes(POS)["citation_error"] == ["swap", "unsourced"] and "term_error" in applicable_types(POS)
    single = dict(POS, refs={"ref-1": GROUP["ref-1"]})
    assert applicable_recipes(single)["citation_error"] == ["unsourced"]  # 没有对照文献只能无中生有
    plain = dict(POS, claim="二甲双胍有效。")
    assert "term_error" not in applicable_types(plain)
    triples = assign_types([POS] * 10, per_positive=2)
    assert len(triples) == 20 and {t for _, t, _ in triples} == set(ERROR_TYPES)
    cite = [r for _, t, r in triples if t == "citation_error"]
    assert sorted(cite) == ["swap", "swap", "unsourced", "unsourced"]  # 两种情况轮着来


# ---------- 训练集 ----------

def test_balance_types_tolerates_missing_type():
    negs = [{"type": "missing_condition"}] * 3 + [{"type": "term_error"}] * 5
    out = balance_types(negs, random.Random(0))
    assert sorted(n["type"] for n in out) == ["missing_condition"] * 3 + ["term_error"] * 3


def test_drop_citations_only_positive_and_unsourced():
    rows = [dict(POS), dict(POS, label="hallucination", type="citation_error", evidence="none"),
            dict(POS, label="hallucination", type="citation_error")]
    out = drop_citations(rows, 1.0, random.Random(0))
    assert [r["cited"] for r in out] == [[], [], ["ref-1"]]


# ---------- 审核台校验 ----------

def test_store_validate_new_types():
    other_none = dict(POS, label="hallucination", type="missing_condition", evidence="none", reason="x")
    assert any("none" in e for e in store.validate(other_none)["errors"])  # none 只能用于引用错误
    unsourced = dict(other_none, type="citation_error", claim=POS["claim"] + "死亡率降低 18%。")
    assert store.validate(unsourced, POS) == {**store.validate(unsourced, POS), "errors": [], "warnings": []}
    assert store.validate(dict(unsourced, cited=[]))["errors"] == []  # 没标引用的无中生有也合法
    cite = dict(POS, label="hallucination", type="citation_error", reason="x")
    assert any("真正的出处" in e for e in store.validate(cite)["errors"])  # 证据就在被引用的 ref-1
    other = next(r for r in POS["refs"] if r != "ref-1")
    v = store.validate(dict(cite, cited=[other]), POS)
    assert v["errors"] == [] and v["warnings"] == []


@pytest.mark.parametrize("t", ERROR_TYPES)
def test_every_type_has_definition_and_label(t):
    from common.prompts import TYPE_DEFINITIONS
    from review.server import TYPE_LABELS
    assert TYPE_DEFINITIONS[t] and TYPE_LABELS[t]


# ---------- 推理 ----------

class FakeLLM:
    def __init__(self):
        self.inputs = []

    def chat_one(self, messages, **kw):
        self.inputs.append(messages[1]["content"])
        return format_output("faithful", "ref-1", "metformin reduced HbA1c by 1.1%", "none", think="t")


def test_judge_rules_and_context():
    from infer.judge import judge_review, to_markdown
    llm = FakeLLM()
    text = ("在 2型糖尿病成人中，二甲双胍使空腹血糖下降 1.1% [ref-1]。"
            "二甲双胍组胃肠道不良事件为 21% [ref-9]。"
            "这些结果提示二甲双胍对年轻患者的长期获益仍需更多研究来确认。")
    res = judge_review(llm, text, GROUP, concurrency=1, check_uncited=True)
    assert len(res) == 3
    assert res[0]["term_check"] and "糖化血红蛋白" in res[0]["term_check"][0]  # 词典预检抓到误译
    assert res[1]["type"] == "citation_error" and res[1]["by_rule"]          # 引用了不存在的 ref-9
    assert res[2]["cited"] == [] and "[cited]\n无" in llm.inputs[-1]          # 没标引用的句子也送检
    assert len(llm.inputs) == 2 and "[对照]" in llm.inputs[0]
    assert "术语预检" in to_markdown(res)


def test_mistranslation_ignores_other_terms_standard_name():
    """文献里同时有 cohort study 和 RCT，claim 写「队列研究」说的是 cohort study，不是 RCT 的误译。"""
    from common.glossary import default_glossary
    refs = {"ref-1": "A randomized controlled trial and a cohort study were pooled."}
    assert default_glossary().mistranslations(refs, "该队列研究纳入 3,100 人。") == []
    assert default_glossary().mistranslations({"ref-1": "A randomized controlled trial."}, "该队列研究纳入 3,100 人。")


def test_mistranslation_uses_context_for_legit_names():
    """claim 挂错到 RCT 那篇，但对照文献里有 cohort study：这是引用问题，不是术语误译。"""
    from common.glossary import default_glossary
    cited = {"ref-1": "A randomized controlled trial."}
    context = {**cited, "ref-3": "A cohort study of older adults."}
    assert default_glossary().mistranslations(cited, "该队列研究纳入 3,100 人。", context) == []
