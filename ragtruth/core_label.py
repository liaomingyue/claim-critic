"""核心集第二轮 Claude 标注（2026-11-06 设计稿，还没跑；跑之前要用户同意，约 81 次调用）。

输入：原文、2B 原回答的编号句子、第一轮改对（policy_sft.FIX_SYSTEM）给出的改动。Claude 不重新判哪句错，
只给每个改动标严重程度 / 类型 / 机制 / 证据 / 理由，给原回答和改对后的回答各标一次满意度。
这些字段直接用作：错误改写模型种子 SFT 的 type / evidence / reason / original_evidence / original_reason，
critic 标签 B / C，sev 训练数据 A–D，满意度成对数据。见 README「第二步」。
"""
from __future__ import annotations

LABEL_SYSTEM = """You annotate corrections that were already made to responses written from a source. Each item gives the source (the exact input an assistant received), the assistant's ORIGINAL response split into numbered sentences, and the corrections: for some sentences S<k>, a corrected version that is supported by the source. Every corrected sentence was wrong in the original; do not re-judge whether it was wrong.

For each correction, describe the error in the ORIGINAL sentence:
- severity: "clear" if a careful reader would call it a factual error (it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make); "minor" if it is technically unsupported with small effect, where people could disagree (a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little).
- type: "conflict" if the source states something incompatible with the original sentence; "baseless" if the source says nothing about what it adds.
- mechanism: one of wrong_attribution, near_synonym_shift, certainty_or_scope_shift, number_or_date, merged_across_passages, added_specific_fact, vague_evaluation_or_addon, unsupported_inference, contradicts_source, other.
- evidence: copy, word for word, the part of the source the error is about (for conflict, what the source says instead; for baseless, the most closely related part, or empty if there is none). One contiguous stretch, as short as possible.
- reason: one English sentence of at most 30 words, concrete: "The source says X, while the sentence says Y" or "The source does not mention Z".
- fixed_evidence: copy, word for word, the part of the source that supports the corrected sentence.
- fixed_reason: one English sentence of at most 30 words describing how the source supports the corrected sentence.

Then judge the ORIGINAL response and the CORRECTED response (the original with all corrections applied) separately, as the person who gave the task would. Do NOT judge faithfulness here; that is handled by the corrections.
- answers_task: "yes" if it does what the task asks; "partial" if it only partly does; "no" if it does not.
- completeness: "complete" if it covers what the task needs; "missing_some"; "missing_much".
- style: "ok", or "problem" if it does not fit the task (wrong format or register, mostly copied source text, filler).
- sentence_form: "ok", or "problem" if sentences are not fluent, repeat each other, or the response is unfinished.
- overall: "good" if it does the job, at most with small gaps; otherwise "not_good".
- note: one short sentence.

Each item is in its own <item id="..."> block.
Reply with one JSON object only: {"items": [{"id": "...", "errors": [{"idx": <sentence number>, "severity": "...", "type": "...", "mechanism": "...", "evidence": "...", "reason": "...", "fixed_evidence": "...", "fixed_reason": "..."}], "original": {"answers_task": "...", "completeness": "...", "style": "...", "sentence_form": "...", "overall": "...", "note": "..."}, "corrected": {...same fields...}}, ...]} with one entry per item and one error entry per correction, in order."""
