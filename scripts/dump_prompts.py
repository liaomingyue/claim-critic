"""把当前循环用到的全部提示词原文导出到 docs/prompts.md（README 引用；改了提示词就重跑一遍，保证文档和代码一致）。

  python -m scripts.dump_prompts            # 写 docs/prompts.md
"""
from __future__ import annotations

import importlib
from pathlib import Path

# (分组, 名称, 模块, 常量, 谁用、用在哪)
ITEMS = [
    ("RAG 生成", "QA 系统提示", "ragtruth.ans_data", "SYSTEM_QA", "2B 回答模型，QA 任务的 system；Summary 不加 system，user = RAGTruth 原 prompt（含原文）"),
    ("核心集标注（Claude）", "逐句改对", "ragtruth.policy_sft", "FIX_SYSTEM", "Claude 改对 2B 回答（核心集 S1，已跑）"),
    ("核心集标注（Claude）", "错误类型和满意度（新增，设计稿）", "ragtruth.core_label", "LABEL_SYSTEM", "Claude 给每个改动标严重程度 / 类型 / 机制 / 证据 / 理由，给原回答和改对回答标满意度（还没跑）"),
    ("评测金标（Claude）", "金标逐句标注", "ragtruth.gold", "SYSTEM", "Claude 金标 v1：验收回答逐句标 显然 / 轻微 / 无错 + 类型"),
    ("critic / RM（2B 单字母）", "RM1 critic", "ragtruth.rm_data", "RM1_SYSTEM", "critic：A 忠实 / B 与原文矛盾 / C 无依据"),
    ("critic / RM（2B 单字母）", "RM2 自然度", "ragtruth.rm_data", "RM2G_SYSTEM", "RM2：A 回答模型自己写的 / B 事后改出来的"),
    ("critic / RM（2B 单字母）", "RM3 改写是否合格（旧）", "ragtruth.rm_data", "RM3_SYSTEM", "RM3 4 选 1（已被 sev 取代，留作对照）"),
    ("critic / RM（2B 单字母）", "sev 严重程度", "ragtruth.sev_rm", "SEV_SYSTEM", "sev 6 选 1：生成器奖励的门槛"),
    ("critic / RM（2B 单字母）", "满意度（成对）", "ragtruth.sat_build", "SAT_PAIR_SYSTEM", "sat_pair：新回答 vs 参考回答，只罚不奖"),
    ("critic / RM（2B 单字母）", "满意度（单条）", "ragtruth.sat_build", "SAT_ABS_SYSTEM", "sat_abs（备用）"),
    ("critic / RM（2B 单字母）", "拒答 RM", "ragtruth.refuse_data", "RM_SYSTEM", "refuse：A 能答 / B 原文答不了"),
    ("单独标注（Claude）", "sev 标注", "ragtruth.sev_label", "SYSTEM", "Claude 标合成错误：显然 / 非显然 / 没有错误 + 解释对不对 + 原句是否已错"),
    ("单独标注（Claude）", "拒答判定", "ragtruth.refuse_data", "SYSTEM", "Claude 判题目能不能从原文回答"),
    ("单独标注（Claude）", "拒答改题", "ragtruth.refuse_data", "REWRITE_SYSTEM", "Claude 最小改动问题，造原文答不了的题"),
    ("错误合成", "错误改写模型 prompt 模板", "ragtruth.synth", "FREE_PROMPT", "生成器的 user prompt 模板（build_prompt_free 填充）"),
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
    out += ["## 错误合成：改法清单（`ragtruth.synth.FOCUS_TACTICS`）", "", "```text"]
    for k, v in FOCUS_TACTICS.items():
        out.append(f"[{k}]")
        out += [f"- {t}" for t in (v if isinstance(v, (list, tuple)) else [v])]
    out += ["```", ""]
    Path("docs/prompts.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/prompts.md", len(out), "lines")


if __name__ == "__main__":
    main()
