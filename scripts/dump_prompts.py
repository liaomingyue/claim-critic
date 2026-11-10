"""把当前循环用到的全部提示词原文导出到 docs/prompts.md（README 引用；改了提示词就重跑一遍，保证文档和代码一致）。

  python -m scripts.dump_prompts            # 写 docs/prompts.md
"""
from __future__ import annotations

import importlib
from pathlib import Path

# (分组, 名称, 模块, 常量, 谁用、用在哪)
ITEMS = [
    ("RAG 生成", "QA 系统提示", "ragtruth.ans_data", "SYSTEM_QA", "2B 回答模型，QA 任务的 system；Summary 不加 system，user = RAGTruth 原 prompt（含原文）"),
    ("核心集标注（Claude）", "逐句改对", "ragtruth.policy_sft", "FIX_SYSTEM", "Claude 改对 2B 回答，得到核心集的改对版本"),
    ("核心集标注（Claude）", "错误类型和满意度", "ragtruth.core_label", "LABEL_SYSTEM", "Claude 给每个改动标严重程度 / 类型 / 机制 / 证据 / 理由，给原回答和改对回答标满意度"),
    ("评测金标（Claude）", "金标逐句标注", "ragtruth.gold", "SYSTEM", "Claude 金标 v1：验收回答逐句标 显然 / 轻微 / 无错 + 类型"),
    ("critic / RM（2B 单字母）", "critic（RM1）", "ragtruth.rm_data", "RM1_SYSTEM", "critic：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = B + C，进 RAG 奖励和生成器奖励"),
    ("critic / RM（2B 单字母）", "sev v3：改动是不是要求的那种错", "ragtruth.sev_gen", "V3_SYSTEM", "sev__v3c0 / v3d：A 要求的那种·显然 / B 要求的那种·细微 / C 别的错 / D 没错；生成器门槛（P(有错) = A + B + C ≥ 0.9）"),
    ("critic / RM（2B 单字母）", "RM2 v3 自然度", "ragtruth.rm2_v3", "RM2V3_SYSTEM", "rm2__v3b / v3d：A 自然 / B 不自然；生成器门槛（P(自然) ≥ 0.5）"),
    ("critic / RM（2B 单字母）", "满意度（成对）", "ragtruth.sat_build", "SAT_PAIR_SYSTEM", "sat_pair：A 参考回答更好 / B 新回答更好；RAG 奖励里只罚不奖"),
    ("critic / RM（2B 单字母）", "拒答 RM", "ragtruth.refuse_data", "RM_SYSTEM", "refuse：A 原文能回答 / B 原文答不了；拒答的奖励 = 0.4 × P(B)"),
    ("错误合成", "错误生成器 v2 指令", "ragtruth.gen_v2", "INSTR", "生成器 v2：原文 + 回答 + 要求的错误类型（9 选 1），输出改哪句、改写句和改动"),
    ("错误合成", "改动描述（9B）", "ragtruth.gen_v2", "DESC_SYSTEM", "9B 给生成器样本写改动描述（sev 训练数据用；训练中不调用）"),
    ("单独标注（Claude）", "合成错误核实", "ragtruth.gen_v2_label", "SYSTEM", "Claude 核实生成器样本：是否真错、显然 / 细微、实际机制、是否等于要求的类型、是否自然；sev / RM2 的训练数据和两轮盲区补样本的核实都用它"),
    ("单独标注（Claude）", "拒答判定", "ragtruth.refuse_data", "SYSTEM", "Claude 判题目能不能从原文回答"),
    ("单独标注（Claude）", "拒答改题", "ragtruth.refuse_data", "REWRITE_SYSTEM", "Claude 最小改动问题，造原文答不了的题"),
    ("旧版（已弃用，留作对照）", "RM2 自然度（旧）", "ragtruth.rm_data", "RM2G_SYSTEM", "第一代 RM2：A 回答模型自己写的 / B 事后改出来的；已由 RM2 v3 取代"),
    ("旧版（已弃用，留作对照）", "RM3 改写是否合格（旧）", "ragtruth.rm_data", "RM3_SYSTEM", "RM3 4 选 1；已由 sev 取代"),
    ("旧版（已弃用，留作对照）", "sev 严重程度（旧）", "ragtruth.sev_rm", "SEV_SYSTEM", "第一代 sev 6 选 1；已由 sev v3 取代"),
    ("旧版（已弃用，留作对照）", "满意度（单条）", "ragtruth.sat_build", "SAT_ABS_SYSTEM", "sat_abs，备用，没进奖励"),
    ("旧版（已弃用，留作对照）", "sev 标注（旧）", "ragtruth.sev_label", "SYSTEM", "第一代 Claude 合成错误标注；已由「合成错误核实」取代"),
    ("旧版（已弃用，留作对照）", "第一代错误改写模型 prompt 模板", "ragtruth.synth", "FREE_PROMPT", "第一代生成器的 user prompt 模板；已由生成器 v2 取代"),
]


def main() -> None:
    out = ["# 提示词原文", "", "由 `python -m scripts.dump_prompts` 从代码导出，不要手改。每段注明谁用、用在哪。", ""]
    group = None
    for g, name, mod, const, use in ITEMS:
        if g != group:
            out += [f"## {g}", ""]
            group = g
        text = getattr(importlib.import_module(mod), const)
        out += [f"### {name}（`{mod}.{const}`）", "", use, "", "```text", text.strip(), "```", ""]
    from ragtruth.synth import FOCUS_TACTICS
    out += ["### 第一代生成器的改法清单（`ragtruth.synth.FOCUS_TACTICS`）", "", "```text"]
    for k, v in FOCUS_TACTICS.items():
        out.append(f"[{k}]")
        out += [f"- {t}" for t in (v if isinstance(v, (list, tuple)) else [v])]
    out += ["```", ""]
    Path("docs/prompts.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/prompts.md", len(out), "lines")


if __name__ == "__main__":
    main()
