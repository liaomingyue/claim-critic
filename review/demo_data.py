"""生成一小份演示数据，不跑模型也能体验审核台。数据是手写的示例，不是流水线产出；
引用错误中的挂载错误和 term_error 直接调用流水线里的规则合成函数生成，和真实数据走同一段代码。

用法：
  python -m review.demo_data            # 写到 outputs/demo/，并生成 configs/demo.yaml
  python -m review.server --config configs/demo.yaml
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml

from common.context import select_context
from common.schema import Sample, write_jsonl
from data.synth_negative import synthesize_one

OUT = Path("outputs/demo")

GROUPS = {
    "g-demo-1": {
        "ref-1": "In a randomized controlled trial of 412 adults aged 18 to 65 with type 2 diabetes, metformin "
                 "reduced HbA1c by 1.1% over 24 weeks compared with placebo.",
        "ref-2": "In the placebo group, 9% of participants reported gastrointestinal adverse events, versus 21% in "
                 "the metformin group.",
        "ref-3": "In a cohort study of 3,100 older adults, hypoglycemia occurred in 4% of patients treated with "
                 "sulfonylureas.",
    },
    "g-demo-2": {
        "ref-1": "Higher sodium intake was associated with elevated systolic blood pressure in a cohort of 5,200 "
                 "community-dwelling adults followed for 6 years.",
        "ref-2": "Among participants with eGFR above 45, the intervention lowered systolic blood pressure by 8 mmHg "
                 "at 12 months.",
        "ref-3": "Hypertension was diagnosed in 31% of participants at baseline.",
    },
}

# (id, group, cited, claim, evidence)
POSITIVES = [
    ("p-001", "g-demo-1", "ref-1", "在 18 至 65 岁的 2型糖尿病成人患者中，二甲双胍在 24 周内使糖化血红蛋白下降 1.1%。",
     'ref-1: "metformin reduced HbA1c by 1.1% over 24 weeks"'),
    ("p-002", "g-demo-1", "ref-2", "二甲双胍组胃肠道不良事件的发生率（21%）高于安慰剂组（9%）。",
     'ref-2: "9% of participants reported gastrointestinal adverse events, versus 21% in the metformin group"'),
    ("p-003", "g-demo-1", "ref-3", "在一项纳入 3,100 名老年人的队列研究中，使用磺脲类药物的患者有 4% 发生低血糖。",
     'ref-3: "hypoglycemia occurred in 4% of patients treated with sulfonylureas"'),
    ("p-004", "g-demo-2", "ref-1", "在随访 6 年的社区成人队列中，钠摄入量较高与收缩压升高相关。",
     'ref-1: "Higher sodium intake was associated with elevated systolic blood pressure"'),
    ("p-005", "g-demo-2", "ref-2", "在 eGFR 高于 45 的参与者中，干预使 12 个月时的收缩压降低 8 mmHg。",
     'ref-2: "Among participants with eGFR above 45, the intervention lowered systolic blood pressure by 8 mmHg"'),
]

# LLM 类型的负样本手写：(parent, type, claim, evidence, reason, round, detect_rate, rewrites, synth_think)
# 规则类型的负样本：(parent, type, None, ..., round, detect_rate, rewrites, None)，内容由规则函数生成
NEGATIVES = [
    ("p-001", "missing_condition", "在 2型糖尿病患者中，二甲双胍在 24 周内使糖化血红蛋白下降 1.1%。",
     'ref-1: "412 adults aged 18 to 65 with type 2 diabetes"', "原文只覆盖 18 到 65 岁成年人，claim 推广到所有患者。",
     1, 0.25, 0, "去掉「18 至 65 岁」「成人」两个限定。"),
    ("p-001", "citation_error", "在 18 至 65 岁的 2型糖尿病成人患者中，二甲双胍在 24 周内使糖化血红蛋白下降 1.1%，并使全因死亡率降低 18%。",
     "none", "参考文献没有报告死亡率，18% 是编出来的。", 1, 0.25, 1, "无中生有：在句末追加一个参考文献里没有的死亡率结局。"),
    ("p-002", "wrong_attribution", "安慰剂组胃肠道不良事件的发生率（21%）高于二甲双胍组（9%）。",
     'ref-2: "9% of participants reported gastrointestinal adverse events, versus 21% in the metformin group"',
     "把两组的不良事件比例对调了。", 1, 0.25, 2, "交换安慰剂组和二甲双胍组。"),
    ("p-003", "citation_error", None, None, None, 2, 0.0, 0, None),
    ("p-001", "term_error", None, None, None, 2, 0.25, 0, None),
    ("p-004", "unsupported_inference", "在随访 6 年的社区成人队列中，钠摄入量较高导致收缩压升高。",
     'ref-1: "Higher sodium intake was associated with elevated systolic blood pressure"', "原文是相关，claim 写成因果。",
     2, 0.25, 0, "把「与……相关」改成「导致」。"),
    ("p-005", "missing_condition", "干预使 12 个月时的收缩压降低 8 mmHg。",
     'ref-2: "Among participants with eGFR above 45"', "丢掉了 eGFR 高于 45 的人群限定。", 2, 0.0, 1, "删掉 eGFR 条件。"),
    # 一条「不是真错误」的漏检样本，演示人工拒绝
    ("p-005", "missing_condition", "在 eGFR 超过 45 的参与者中，干预使 12 个月时的收缩压降低 8 mmHg。",
     'ref-2: "Among participants with eGFR above 45"', "改写了措辞。", 2, 0.0, 2, "把「高于」换成「超过」。"),
]

FAITHFUL_OUT = ('<think>claim 与被引用文献一致。</think>\n<verdict>faithful</verdict>\n'
                '<evidence>ref-1: "metformin reduced HbA1c by 1.1% over 24 weeks"</evidence>\n<type>none</type>')
HALLUC_OUT = ('<think>claim 去掉了原文的限定条件。</think>\n<verdict>hallucination</verdict>\n'
              '<evidence>ref-1: "adults aged 18 to 65"</evidence>\n<type>missing_condition</type>\n<reason>丢了限定。</reason>')


def main() -> None:
    positives = []
    for pid, g, cited, claim, ev in POSITIVES:
        positives.append(Sample(id=pid, split="train", source_group=g,
                                refs=select_context(claim, [cited], GROUPS[g], k=2), cited=[cited], claim=claim,
                                label="faithful", type="none", evidence=ev, origin="positive",
                                meta={"judge_think": "claim 的对象、条件和数字都能在被引用的文献中找到。"}).to_dict())
    pos_by_id = {p["id"]: p for p in positives}

    synth, hard = [], []
    for k, (parent, t, claim, ev, reason, rnd, rate, rewrites, think) in enumerate(NEGATIVES):
        p = pos_by_id[parent]
        if claim is None:  # 规则类型：走流水线的合成函数
            base = synthesize_one(None, p, t, {}, random.Random(k), {}, recipe={"citation_error": "swap", "term_error": "term"}[t])
            base["meta"] = dict(base["meta"])
        else:
            base = Sample(id="", split="train", source_group=p["source_group"], refs=p["refs"], cited=p["cited"],
                          claim=claim, label="hallucination", type=t, evidence=ev, reason=reason,
                          origin="synthetic", parent_id=parent,
                          meta={"recipe": "unsourced" if ev == "none" else "llm", "synth_think": think}).to_dict()
        base["id"] = f"{parent}-b{rnd}-{t}-{k}"
        synth.append(dict(base, round=0))
        n_hit = round(rate * 4)
        outs = [HALLUC_OUT] * n_hit + [FAITHFUL_OUT] * (4 - n_hit)
        hard.append(dict(base, round=rnd, meta={**base["meta"], "detect_rate": rate, "rewrites": rewrites,
                                                "critic_outputs": outs, "batch": rnd}))

    files = {
        "positives": OUT / "positives.jsonl", "pos_audit": OUT / "positives_audit.jsonl",
        "synth": OUT / "synth_batch{batch}.jsonl", "hard": OUT / "hard_batch{batch}.jsonl",
        "hard_audit": OUT / "hard_batch{batch}_audit.jsonl",
    }
    write_jsonl(files["positives"], positives)
    write_jsonl(files["pos_audit"], positives[:2])
    for b in (1, 2):
        write_jsonl(str(files["synth"]).format(batch=b), [s for s, h in zip(synth, hard) if h["round"] == b])
        batch = [h for h in hard if h["round"] == b]
        write_jsonl(str(files["hard"]).format(batch=b), batch)
        write_jsonl(str(files["hard_audit"]).format(batch=b), batch)

    cfg = {
        "label_positive": {"out": str(files["positives"]), "audit_out": str(files["pos_audit"]), "max_audit_error": 0.03},
        "synth_negative": {"out": str(files["synth"])},
        "mine_hard": {"out": str(files["hard"]), "audit_out": str(files["hard_audit"]), "max_not_error_ratio": 0.10},
        "build_trainset": {"coldstart_out": str(OUT / "coldstart.jsonl")},
        "review": {"dir": str(OUT / "review")},
    }
    Path("configs/demo.yaml").write_text("# 演示数据配置，由 python -m review.demo_data 生成\n"
                                         + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"demo data -> {OUT}; run: python -m review.server --config configs/demo.yaml")


if __name__ == "__main__":
    main()
