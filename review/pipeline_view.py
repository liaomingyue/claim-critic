"""审核台的「调用流程」与「Prompt 模板」：把一条样本从用户问题到最终样本的每一步、每次模型调用串起来。

调用记录来自 common.trace（配置里的 trace_dir），按阶段文件存：
  questions      id = 「主题|意图」       生成用户问题（build_groups）
  reviews        id = 综述 id              大纲 + 正文两次调用（gen_review）
  positives      id = claim id             强模型判定（label_positive）
  synth_batch{k} id = 负样本 id            合成改写（synth_negative；规则类型没有模型调用）
  hard_batch{k}  id = 负样本 id            critic 采样 + 改写（mine_hard）
样本 → claim id（正样本是自己，负样本是 parent_id）→ 综述 id（去掉 -cNNN）→ 文献组 id（去掉 -vN）。
"""
from __future__ import annotations

import os
from pathlib import Path

from common import trace
from common.prompts import CRITIC_SYSTEM, JUDGE_SYSTEM
from common.schema import read_jsonl
from data import build_groups, gen_review, synth_negative
from data.intents import INTENTS, QUESTION_PROMPT, question_prompt
from train.sft import RATIONALE_PROMPT


def prompt_catalog() -> list[dict]:
    """各阶段的 prompt 模板，按流水线顺序。{...} 是运行时填入的字段。"""
    intents = "\n".join(f"- {k}（{v['label']}）：文献类型 {v['kinds']}；近 {v['recent_years'] or '不限'} 年；"
                        f"{'加 FDA 说明书；' if v['drug_label'] else ''}写法：{v['writing_rule']}"
                        for k, v in INTENTS.items())
    return [
        {"stage": "questions", "title": "1. 生成用户问题", "where": "data/intents.py · data/build_groups.py",
         "model": "generator", "calls": "每个「主题 × 意图」1 次",
         "templates": {"QUESTION_PROMPT": QUESTION_PROMPT, "各意图的检索与写作要求": intents}},
        {"stage": "retrieval", "title": "2. 检索文献、判相关性", "where": "data/build_groups.py · data/study_type.py",
         "model": "generator", "calls": "规则：检索词全部命中、相关度前 60 篇 → 文献类型 → 标题含检索词；"
                                        "模型：每个问题 1 次，从最多 15 篇候选里选出相关的；用药另取 FDA 说明书",
         "templates": {"RELEVANCE_PROMPT": build_groups.RELEVANCE_PROMPT}},
        {"stage": "reviews", "title": "3. 生成综述", "where": "data/gen_review.py", "model": "generator",
         "calls": "每组 2 次：大纲、正文",
         "templates": {"OUTLINE_PROMPT": gen_review.OUTLINE_PROMPT, "BODY_PROMPT": gen_review.BODY_PROMPT,
                       "LANGUAGE_RULES[zh]": gen_review.LANGUAGE_RULES["zh"]}},
        {"stage": "claims", "title": "4. 切分 claim（无模型调用）", "where": "data/split_claims.py · common/context.py",
         "model": "—", "calls": "按句切分，附被引用文献 + 2 篇对照文献", "templates": {}},
        {"stage": "positives", "title": "5. 判定正样本", "where": "data/label_positive.py · common/prompts.py",
         "model": "judge", "calls": "每条 claim 1 次；输入格式同 critic，<think> 换成 <analysis>（Claude 会拦截要求输出推理过程的请求）",
         "templates": {"JUDGE_SYSTEM（判定用）": JUDGE_SYSTEM, "CRITIC_SYSTEM（critic 训练与推理用）": CRITIC_SYSTEM}},
        {"stage": "synth_batch", "title": "6. 合成负样本", "where": "data/synth_negative.py", "model": "generator",
         "calls": "LLM 类型每条 1 次；挂载错误（swap）和术语错误（term）走规则，不调用模型",
         "templates": {"SYNTH_PROMPT": synth_negative.SYNTH_PROMPT,
                       **{f"RECIPE_RULES[{k}]": v for k, v in synth_negative.RECIPE_RULES.items()}}},
        {"stage": "hard_batch", "title": "7. 挖难样本", "where": "data/mine_hard.py", "model": "critic + generator",
         "calls": "critic 采样 4 次；被识破的带着 critic 的分析重写，最多 2 次",
         "templates": {"REWRITE_SUFFIX（接在 SYNTH_PROMPT 后）": synth_negative.REWRITE_SUFFIX}},
        {"stage": "rationale", "title": "8. 全量 SFT 的推理过程（基线用）", "where": "train/sft.py", "model": "rationale_model",
         "calls": "每条 1 次", "templates": {"RATIONALE_PROMPT": RATIONALE_PROMPT}},
    ]


def _groups_path(cfg: dict) -> str | None:
    sec = cfg.get("build_groups") or cfg.get("fetch") or {}
    return sec.get("out")


def call_chain(cfg: dict, sample: dict) -> dict:
    """一条样本的完整调用流程。没有 trace_dir 或某阶段没记录时，对应步骤只给数据、不给调用。"""
    tdir = Path(cfg.get("trace_dir") or os.environ.get("CRITIC_TRACE_DIR") or "")
    has_trace = bool(cfg.get("trace_dir") or os.environ.get("CRITIC_TRACE_DIR"))

    def calls(stage_file: str, key: str | None) -> list[dict]:
        if not has_trace or not key:
            return []
        return trace.load(tdir / f"{stage_file}.jsonl").get(key, [])

    claim_id = sample["id"] if sample.get("label") == "faithful" else sample.get("parent_id")
    review_id = claim_id.rsplit("-c", 1)[0] if claim_id else None
    group_id = review_id.rsplit("-v", 1)[0] if review_id else None
    group = next((g for g in read_jsonl(_groups_path(cfg)) if g["group_id"] == group_id), None) \
        if group_id and _groups_path(cfg) and Path(_groups_path(cfg)).exists() else None
    review_path = cfg.get("gen_review", {}).get("out")
    review = next((r for r in read_jsonl(review_path) if r.get("id") == review_id), None) \
        if review_id and review_path and Path(review_path).exists() else None

    def question_calls(g: dict) -> list[dict]:
        """优先用调用记录；问题生成早于调用记录时，用模板重建 prompt、配上 questions.jsonl 里缓存的原始输出。"""
        key = f"{g['topic']}|{g['intent']}"
        got = calls("questions", key)
        qpath = (cfg.get("build_groups") or {}).get("questions_out")
        if got or not qpath or not Path(qpath).exists():
            return got
        row = next((r for r in read_jsonl(qpath) if r.get("id") == key), None)
        if not row or "raw" not in row:
            return []
        n = cfg["build_groups"].get("questions_per_intent", 7)
        return [{"model": cfg.get("generator", {}).get("model", "?"), "provider": "重建（无调用记录）",
                 "system": "", "user": question_prompt(g["topic"], g["intent"], n), "outputs": [row["raw"]],
                 "secs": "?", "ts": "prompt 按当前模板重建，输出为当时缓存的原文", "call": 0}]

    steps = []
    if group and group.get("question"):
        steps.append({"stage": "questions", "title": "生成用户问题",
                      "data": {"意图": INTENTS[group["intent"]]["label"], "主题": group["topic"],
                               "问题": group["question"], "检索词": ", ".join(group.get("keywords", [])),
                               "药名": group.get("drug") or "—"},
                      "calls": question_calls(group)})
    if group:
        cands = group.get("candidates") or []
        steps.append({"stage": "retrieval", "title": "检索文献（规则检索 → 模型判相关性）",
                      "data": {"候选 / 入选": f"{len(cands)} 篇候选，{sum(c['selected'] for c in cands)} 篇判为相关"}
                      if cands else None,
                      "papers": [{"ref": f"ref-{i + 1}", "title": p["title"], "year": p.get("year"),
                                  "kind": p.get("kind", "?"), "pmcid": p["pmcid"]} for i, p in enumerate(group["papers"])],
                      "rejected": [c for c in cands if not c["selected"]],
                      "calls": calls("relevance", group.get("qid"))})
    if review_id:
        steps.append({"stage": "reviews", "title": "生成综述（大纲 → 正文）",
                      "data": {"综述 id": review_id, "正文": (review or {}).get("text", "（未找到综述文件）")},
                      "calls": calls("reviews", review_id)})
    if claim_id:
        steps.append({"stage": "claims", "title": "切分 claim、附对照文献",
                      "data": {"claim id": claim_id}, "calls": []})
        steps.append({"stage": "positives", "title": "强模型判定正样本", "calls": calls("positives", claim_id)})
    if sample.get("label") == "hallucination":
        recipe = sample.get("meta", {}).get("recipe", "llm")
        batch = sample.get("meta", {}).get("batch") or sample.get("round") or 1
        synth_calls = []
        for k in range(1, 4):
            synth_calls += calls(f"synth_batch{k}", sample["id"])
        steps.append({"stage": "synth_batch", "title": f"合成负样本（{recipe}）",
                      "data": {"类型": sample["type"], "合成方式": recipe,
                               "说明": "规则合成，无模型调用" if recipe in ("swap", "term") else "LLM 改写"},
                      "calls": synth_calls})
        steps.append({"stage": "hard_batch", "title": "挖难样本（critic 采样 + 改写）",
                      "calls": calls(f"hard_batch{batch}", sample["id"])})
    return {"has_trace": has_trace, "steps": steps}
