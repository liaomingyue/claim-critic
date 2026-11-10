# 提示词原文

由 `python -m scripts.dump_prompts` 从代码导出，不要手改。每段注明谁用、用在哪。

## RAG 生成

### QA 系统提示（`ragtruth.ans_data.SYSTEM_QA`）

2B 回答模型，QA 任务的 system；Summary 不加 system，user = RAGTruth 原 prompt（含原文）

```text
Answer the question using the passages. If the passages contain information relevant to the question, give the best answer they support, even if it is partial. Reply that you are unable to answer only when the passages contain nothing relevant to the question.
```

## 核心集标注（Claude）

### 逐句改对（`ragtruth.policy_sft.FIX_SYSTEM`）

Claude 改对 2B 回答，得到核心集的改对版本

```text
You correct hallucinations in responses written from a source. Compare every sentence of each response with its source yourself, using the strict standard below, and return corrections ONLY for the sentences that are wrong.

A sentence is wrong whenever it says anything the source does not state and that cannot be directly inferred from it; when unsure, treat it as wrong. Wrong: changed or invented numbers, dates, names, quantities; facts tied to the wrong person, entity or item; changed certainty or scope; added concrete details, evaluations, purposes, consequences or recommendations the source does not make, including vague add-ons; near-synonyms that shift the meaning; details from different passages joined into a claim the source does not make. Not wrong: true synonyms, faithful summaries and reorderings, leaving information out, what follows directly from the source, saying truthfully that the source does not give some information.

For each wrong sentence give the smallest rewrite that makes it fully supported by the source, keeping its role, length and style in the response; if nothing supported can be said in its place, give an empty string to delete it. Never touch correct sentences.
If a response refuses ("unable to answer") but the source does contain information that answers the question, give a "rewrite": a complete faithful answer in the same style and length as a normal response. If the refusal is justified, leave it alone.

Each response is in its own <item id="..."> block with its own source and numbered sentences.
Reply with one JSON object only: {"items": [{"id": "...", "fixes": [{"idx": <sentence number>, "new": "<rewritten sentence or empty>"}], "rewrite": null or "<full answer>"}, ...]} with one entry per item (fixes empty when nothing is wrong).
```

### 错误类型和满意度（`ragtruth.core_label.LABEL_SYSTEM`）

Claude 给每个改动标严重程度 / 类型 / 机制 / 证据 / 理由，给原回答和改对回答标满意度

```text
You annotate corrections that were already made to responses written from a source. Each item gives the source (the exact input an assistant received), the assistant's ORIGINAL response split into numbered sentences, and the corrections: for some sentences S<k>, a corrected version that is supported by the source. Every corrected sentence was wrong in the original; do not re-judge whether it was wrong.

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
Reply with one JSON object only: {"items": [{"id": "...", "errors": [{"idx": <sentence number>, "severity": "...", "type": "...", "mechanism": "...", "evidence": "...", "reason": "...", "fixed_evidence": "...", "fixed_reason": "..."}], "original": {"answers_task": "...", "completeness": "...", "style": "...", "sentence_form": "...", "overall": "...", "note": "..."}, "corrected": {...same fields...}}, ...]} with one entry per item and one error entry per correction, in order.
```

## 评测金标（Claude）

### 金标逐句标注（`ragtruth.gold.SYSTEM`）

Claude 金标 v1：验收回答逐句标 显然 / 轻微 / 无错 + 类型

```text
You label hallucinations in responses written from a source, for a gold-standard dataset. Compare every sentence of the response with the source yourself. Use earlier sentences only to resolve pronouns and references; judge each sentence's own claims against the source.

A sentence is an error whenever it says anything the source does not state and that cannot be directly inferred from it. When you are unsure whether the source supports it, call it an error.

These are errors:
- a number, date, name or quantity that is changed or invented; a fact tied to the wrong person, entity or item;
- changed certainty or scope ("proposed a fine" becoming "fined", "one producer" becoming "producers", "may" becoming "likely");
- an added concrete detail; an added evaluation, purpose, consequence or recommendation the source does not make, including vague add-ons ("making it a great choice");
- a near-synonym that shifts the meaning ("provided" becoming "promised");
- details from different passages joined into a claim the source does not make;
- a checkably false statement the response makes about itself, such as a wrong word count (mechanism self_description, severity minor).

These are not errors:
- true synonyms, faithful summaries and reorderings; equivalents any reader knows (abbreviations, well-known alternative names);
- boilerplate with no factual content ("Sure, here is a summary"); restating the question;
- leaving information out, unless what remains becomes false;
- what follows directly from the source, such as simple arithmetic;
- saying truthfully that the source does not give some information.

For each sentence give:
- is_error: true or false.
- type: "conflict" if the source states something incompatible; "baseless" if the source says nothing about it; "none" if not an error.
- mechanism: one of wrong_attribution, number_or_date, contradicts_source, merged_across_passages, added_specific_fact, vague_evaluation_or_addon, certainty_or_scope_shift, near_synonym_shift, unsupported_inference, self_description, other, none (none only if not an error).
- severity: "clear" if a careful reader would call it a factual error; "minor" if technically unsupported with small effect, where people could disagree; "none" if not an error.
- evidence: the relevant words copied from the source; empty if the source says nothing about it.
- note: one short sentence.

Reply with one JSON object only: {"sentences": [{"idx": 1, "is_error": true|false, "type": "...", "mechanism": "...", "severity": "...", "evidence": "...", "note": "..."}, ...]} with exactly one entry per sentence, in order.
```

## critic / RM（2B 单字母）

### critic（RM1）（`ragtruth.rm_data.RM1_SYSTEM`）

critic：A 有依据 / B 与原文矛盾 / C 无依据；P(幻觉) = B + C，进 RAG 奖励和生成器奖励

```text
You check whether one sentence of an assistant's response is faithful to the source material the assistant was given.

The source is the exact input the assistant received (instructions plus material). The full response is shown for context, split into numbered sentences; you check only the sentence given under [sentence to check]. Judge only against the source, not against world knowledge.

Options:
A. faithful: the sentence is supported by the source, or has no factual content.
B. conflict: the sentence contradicts the source.
C. baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.

Tips:
- Structured data (JSON): True/False means the feature is offered or not; times such as 17:30-21:0 mean 5:30 PM to 9:00 PM, and a correct conversion is supported; 0:0-0:0 means no listed hours.
- Paraphrases, faithful summaries and numbers that follow directly from the source (for example by simple arithmetic) are supported.
- Check every item of a list or every detail separately: one added item or detail that the source does not contain makes the sentence baseless.
- The type must follow from the reason: if the reason says the source does not contain or contradicts something in the sentence, the type cannot be none.

Answer with a single letter: A, B or C.
```

### sev v3：改动是不是要求的那种错（`ragtruth.sev_gen.V3_SYSTEM`）

sev__v3c0 / v3d：A 要求的那种·显然 / B 要求的那种·细微 / C 别的错 / D 没错；生成器门槛（P(有错) = A + B + C ≥ 0.9）

```text
You check whether an edit to one sentence of a response introduced the REQUESTED kind of error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL (faithful) and EDITED versions of that sentence, the requested kind of error, a description of the edit and the exact word-level changes.

Judge the changes against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source or is stated elsewhere in it.

Options:
A. a clear error of the requested kind.
B. a minor error of the requested kind.
C. an error, but not of the requested kind.
D. not an error.

Answer with a single letter: A, B, C or D.
```

### RM2 v3 自然度（`ragtruth.rm2_v3.RM2V3_SYSTEM`）

rm2__v3b / v3d：A 自然 / B 不自然；生成器门槛（P(自然) ≥ 0.5）

```text
You see a source (the exact input an assistant received) and one sentence from the assistant's response, with the sentence before and after it for context.

Could a real assistant plausibly have written this sentence in this response: fluent, fitting the context and the style of the rest of the response?
A. natural: yes, a real assistant could have written it like this.
B. unnatural: it reads garbled, out of place or obviously artificial.

Answer with a single letter: A or B.
```

### 满意度（成对）（`ragtruth.sat_build.SAT_PAIR_SYSTEM`）

sat_pair：A 参考回答更好 / B 新回答更好；RAG 奖励里只罚不奖

```text
You compare two answers to the same task the way the person who gave the task would.

Judge only whether an answer is well formed, addresses what was asked and actually solves the task. Do NOT judge whether its claims are supported by the source; that is checked elsewhere.

Bad signs: stops abruptly or is unfinished, mostly stitched source text instead of an answer written for the task, so short that the person does not get what they need, does not address what was asked, leaves out information the task needs, filler or repetition. If the source really does not contain the answer, saying so is the useful answer.

Options:
A. answer A is better.
B. answer B is better.

Reply with one letter.
```

### 拒答 RM（`ragtruth.refuse_data.RM_SYSTEM`）

refuse：A 原文能回答 / B 原文答不了；拒答的奖励 = 0.4 × P(B)

```text
You see a question-answering task an assistant received: the question, the passages and instructions. Decide whether the passages let the assistant answer the question.

Options:
A. answer: the passages contain the answer, or relevant information for a useful partial answer.
B. refuse: the passages contain nothing that answers or meaningfully addresses the question.

Reply with one letter.
```

## 错误合成

### 错误生成器 v2 指令（`ragtruth.gen_v2.INSTR`）

生成器 v2：原文 + 回答 + 要求的错误类型（9 选 1），输出改哪句、改写句和改动

```text
You are creating training data for a model that detects unsupported or contradicted sentences in responses written from a source.
Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence is faithful to the source.

Rewrite exactly ONE sentence so that it contains an error of this kind:
{mech_name}: {mech_desc}

Choose the sentence where a real assistant could most plausibly make this kind of mistake. The rewritten sentence must still read naturally in the response.

<source>
{source}
</source>

<response>
{sentences}
</response>

Output only one JSON object, nothing else:
{{"idx": <number of the rewritten sentence>, "sentence": "<the full rewritten sentence>", "edit": "<one sentence describing exactly what you changed>"}}
```

### 改动描述（9B）（`ragtruth.gen_v2.DESC_SYSTEM`）

9B 给生成器样本写改动描述（sev 训练数据用；训练中不调用）

```text
You describe edits. Given an ORIGINAL sentence, an EDITED version of it, the exact word-level changes, and the KIND of edit that was intended, write ONE short English sentence (at most 25 words) that says what the edit changed in meaning, e.g. "Changed who released the training records from a lawyer to the police." or "Replaced the serving suggestion with a claim that the dish is quick and perfect for any night."
When the edit rewrites much of the sentence, summarize it through the intended kind of edit (for example "Shifted the claim from 'should provoke self-examination' to 'is inspiring reforms', making it more certain.").
Cover every change listed in the word-level changes (mention minor ones briefly), not only the main one.
Describe only the change itself. Do not judge whether it is right or wrong, do not mention any source, passage, error, support or accuracy. Output only the sentence.
```

## 单独标注（Claude）

### 合成错误核实（`ragtruth.gen_v2_label.SYSTEM`）

Claude 核实生成器样本：是否真错、显然 / 细微、实际机制、是否等于要求的类型、是否自然；sev / RM2 的训练数据和两轮盲区补样本的核实都用它

```text
You audit sentences that were deliberately edited to plant a specific kind of error. Each item gives the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL (faithful) and EDITED versions of that sentence, and the REQUESTED kind of error.

Judge the EDITED sentence against the source (use the other response sentences only to resolve references):
1. severity:
   - "clear": a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
   - "minor": technically unsupported with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
   - "none": not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source or is stated elsewhere in it.
2. mechanism: the actual kind of error, one of certainty_or_scope_shift, near_synonym_shift, wrong_attribution, contradicts_source, merged_across_passages, unsupported_inference, number_or_date, vague_evaluation_or_addon, added_specific_fact, other; "none" if severity is none. Definitions:
   - certainty_or_scope_shift: shift the certainty or scope of a claim (possible -> certain, some -> all, usually -> always, a specific case -> a general rule, or the reverse)
   - near_synonym_shift: replace a word or phrase with a near-synonym that changes the meaning a little
   - wrong_attribution: attribute a statement, action or property to the wrong person, entity, passage or step
   - contradicts_source: state something the source contradicts (reverse a relation, an outcome or a fact)
   - merged_across_passages: join details from different parts of the source into one claim the source does not make
   - unsupported_inference: draw a conclusion, cause or purpose that the source does not state
   - number_or_date: get a number, date, duration, count or order wrong
   - vague_evaluation_or_addon: add a vague evaluation or add-on that the source does not support
   - added_specific_fact: add a specific fact or detail that the source does not contain
3. matches: true if severity is not none and the actual mechanism is the requested one; otherwise false.
4. type: "conflict" if the source states something incompatible with the edited sentence (including a changed certainty or scope, a meaning-changing near-synonym, wrong attribution, number, date or order); "baseless" if the source says nothing about what it adds; "n/a" if severity is none.
5. natural: "yes" if a real assistant could plausibly have written the edited sentence in this response (fluent, fits the context and style); "no" if it reads garbled, out of place or obviously artificial.

Reply with one JSON object only: {"items": [{"id": ..., "severity": "...", "mechanism": "...", "matches": true|false, "type": "...", "natural": "yes"|"no"}, ...]}, one entry per item, in the given order.
```

### 拒答判定（`ragtruth.refuse_data.SYSTEM`）

Claude 判题目能不能从原文回答

```text
You decide whether a question can be answered from the passages given with it. You receive the exact task an assistant received: the question, the passages, and instructions.

Judge only the passages, not your own knowledge.
- yes: the passages contain the answer to what is asked.
- partial: the passages contain information relevant to the question that lets an assistant give a useful partial answer.
- no: the passages contain nothing that answers or meaningfully addresses the question; the right reply is to say it cannot be answered.

Reply with one JSON object only:
{"answerable": "yes" | "partial" | "no", "evidence": "a short quote from the passages, or empty for no", "note": "one short sentence"}
```

### 拒答改题（`ragtruth.refuse_data.REWRITE_SYSTEM`）

Claude 最小改动问题，造原文答不了的题

```text
You help build training data for deciding when an assistant should refuse to answer from given passages. Real cases where the passages cannot answer look like this: the passages are on the right topic, but the question asks for a specific thing they never cover. Real examples (question, and why the passages could not answer it):
{examples}

For each task below (marked [task <id>], with its question and passages), write a new question that a real user could plausibly ask about the same topic, changed as little as possible, so that these passages no longer answer it, in the same way as the examples: a different specific product, person, place or time; an added condition the passages do not cover; asking why or how when they only say what; asking for a comparison they never make. The new question must stay on the topic of the passages, read naturally (same casual style, typos are fine), and must not be answerable from the passages, even partly.

Reply with one JSON object only, one entry per task in the given order:
{{"results": [{{"id": "<task id>", "question": "the new question", "how": "one short phrase: what you changed"}}, ...]}}
```

## 旧版（已弃用，留作对照）

### RM2 自然度（旧）（`ragtruth.rm_data.RM2G_SYSTEM`）

第一代 RM2：A 回答模型自己写的 / B 事后改出来的；已由 RM2 v3 取代

```text
You see a source (the exact input an assistant received) and one sentence from the assistant's response, with the sentence before and after it for context.

Was this sentence written by the assistant itself while writing its answer, or was it edited afterwards by someone who deliberately planted an error in it?
A. natural: the assistant wrote this sentence itself.
B. synthetic: someone edited this sentence afterwards to plant an error.

Answer with a single letter: A or B.
```

### RM3 改写是否合格（旧）（`ragtruth.rm_data.RM3_SYSTEM`）

RM3 4 选 1；已由 sev 取代

```text
You are checking synthetic training data for a hallucination detector in retrieval-augmented generation.
A generator took a response that was faithful to its source and rewrote one sentence so that it should contain an error (a conflict with the source, or baseless information not in the source). It also wrote the error type, the evidence from the source, and a reason.

Check the source yourself and judge this sample:
A. correct: the rewritten sentence really is wrong with respect to the source, and the type, evidence and reason correctly describe that error.
B. wrong_explanation: the rewritten sentence is wrong, but the type, the evidence or the reason does not correctly describe the error.
C. not_an_error: the rewritten sentence is actually supported by the source, or is only a harmless rewording, so it should not be labeled as an error.
D. original_wrong: the original sentence itself already conflicts with or goes beyond the source.

Use a strict standard. Where the line is:
- It is an error (correct, if the type, evidence and reason fit) whenever the rewritten sentence says anything the source does not state and that cannot be directly inferred from it. This includes: a changed number, date, name, quantity or order; a fact attributed to the wrong entity, passage or step; a strengthened or weakened certainty or scope; a specific added detail; and also an added evaluation, purpose, consequence or recommendation the source does not make ("making it a great choice", "suitable for many occasions", "which is helpful for users"), or a near-synonym that shifts the meaning, tone or certainty ("provided" becoming "promised", "may" becoming "is likely to").
- It is not_an_error only if the change keeps exactly the same meaning: a true synonym, a reordering, or a rewording that adds and removes nothing.
- When you are unsure whether the source supports the change, treat it as an error.

Examples (from other sources):
- Source: "The museum opens at 9 am." Rewrite: "The museum opens at 10 am." -> correct (conflict).
- Source: "The museum opens at 9 am." Rewrite: "The museum, founded in 1920, opens at 9 am." -> correct (baseless).
- Source: "The cafe offers free Wi-Fi." Rewrite: "The cafe offers free Wi-Fi, making it a convenient spot for visitors." -> correct (baseless: an evaluation the source does not make).
- Source: "The government has provided $2 million." Rewrite: "The government has promised $2 million." -> correct (conflict: provided is not the same as promised).
- Source: "The company reported higher sales." Rewrite: "The company posted higher sales." -> not_an_error (same meaning).
- Source: "The bridge opened in 1998." Rewrite: "The bridge opened in 1996." with claimed type baseless -> wrong_explanation (it is a conflict, not baseless).

Answer with a single letter: A, B, C or D.
```

### sev 严重程度（旧）（`ragtruth.sev_rm.SEV_SYSTEM`）

第一代 sev 6 选 1；已由 sev v3 取代

```text
You audit a sentence that was deliberately edited to plant an error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the original version of that sentence, and the editor's claimed error: its type, the source evidence it points to, and its reason.

Judge the edited sentence against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source.
The explanation is right only if the claimed type is right (conflict = the source contradicts it; baseless = the source does not contain it), the claimed evidence is the part of the source the error is about, and the reason describes the actual error.

Options:
A. clear error, and the explanation is right.
B. clear error, but the explanation is wrong.
C. minor error, and the explanation is right.
D. minor error, but the explanation is wrong.
E. not an error.
F. the original sentence is already unfaithful to the source.

Answer with a single letter: A, B, C, D, E or F.
```

### 满意度（单条）（`ragtruth.sat_build.SAT_ABS_SYSTEM`）

sat_abs，备用，没进奖励

```text
You judge one answer to a task the way the person who gave the task would.

Judge only whether an answer is well formed, addresses what was asked and actually solves the task. Do NOT judge whether its claims are supported by the source; that is checked elsewhere.

Bad signs: stops abruptly or is unfinished, mostly stitched source text instead of an answer written for the task, so short that the person does not get what they need, does not address what was asked, leaves out information the task needs, filler or repetition. If the source really does not contain the answer, saying so is the useful answer.

Options:
A. good: the answer does the job, at most with small gaps.
B. not good: the answer is only partly useful or worse.

Reply with one letter.
```

### sev 标注（旧）（`ragtruth.sev_label.SYSTEM`）

第一代 Claude 合成错误标注；已由「合成错误核实」取代

```text
You audit sentences that were deliberately edited to plant an error. Each item gives the source (the exact input an assistant received), the assistant's response split into numbered sentences with the target sentence S<k> already edited, the ORIGINAL version of that sentence before editing, and the editor's claimed error: its type, the source evidence it points to, and its reason.

Judge in this order:
1. original_ok: is the ORIGINAL sentence faithful to the source? (true / false)
2. severity of the EDITED sentence against the source (use the other response sentences only to resolve pronouns and references):
   - "clear": a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
   - "minor": technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
   - "none": not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source.
   When unsure between minor and none, choose minor; when unsure between clear and minor, choose the one a careful reader would more likely pick.
3. explanation_ok: only if severity is clear or minor. true only if all hold: the claimed type is right (conflict = the source contradicts it; baseless = the source does not contain it), the claimed evidence is the part of the source the error is about, and the reason describes the actual error in the edited sentence. Otherwise false. Use "n/a" when severity is none.
4. mechanism: one of wrong_attribution, contradicts_source, merged_across_passages, number_or_date, certainty_or_scope_shift, added_specific_fact, vague_evaluation_or_addon, near_synonym_shift, unsupported_inference, other, none (none only when severity is none).
5. note: one short sentence.

Reply with one JSON object only: {"items": [{"id": ..., "original_ok": true|false, "severity": "clear"|"minor"|"none", "explanation_ok": true|false|"n/a", "mechanism": "...", "note": "..."}, ...]}, one entry per item, in the given order.
```

### 第一代错误改写模型 prompt 模板（`ragtruth.synth.FREE_PROMPT`）

第一代生成器的 user prompt 模板；已由生成器 v2 取代

```text
You are creating training data for a model that detects unsupported or contradicting sentences in an assistant's response. Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence of this response is supported by the source.

{input}

Choose one sentence of this response and rewrite it so that it contains exactly one error of one of these kinds:
- conflict: {conflict_def}
- baseless: {baseless_def}

Choose the sentence, the kind of error and the way of introducing it yourself: pick what a real assistant writing this response would most plausibly get wrong. Some common ways, for reference only (you may use another):
- conflict: {conflict_tactics}
- baseless: {baseless_tactics}

Make the error realistic and hard to notice, like the mistakes real assistants make:
- change as little as possible; keep the wording, style and length close to the original; the response must still read naturally;
- the error must be wrong with respect to the source, not a paraphrase or a harmless rewording;
- no absurd values, no contradiction with the other sentences of the response, no hedges such as "reportedly".

Real sentences with such errors written by assistants on similar tasks (for style only, they belong to other sources):
{examples}

Then analyse both versions of the chosen sentence against the source, as a careful fact-checker would:
- "evidence": copy, word for word, the part of the source that the rewritten sentence contradicts (conflict, never null) or is most closely related to (baseless; null only if the source has nothing related). Copy one contiguous stretch, as short as possible; for a long passage copy its first few words and its last few words joined by " ... " (each part at least 3 words).
- "reason": one English sentence of at most 30 words: what the source says and what the rewritten sentence says, concretely (for example "The source says X, while the sentence says Y" or "The source does not mention Z").
- "original_evidence": the part of the source, word for word, that supports the original sentence (null only if it has no factual content).
- "original_reason": one English sentence of at most 30 words describing how the source supports the original sentence.
Never use the words hallucination, faithful, conflict, baseless, ground truth in the sentence, the evidence or the reasons (only in the "type" field).

{tips}

Output only one JSON object, nothing else:
{{"plan": "which sentence you chose, why a real assistant could get it wrong, and how you introduce the error", "idx": <number of the chosen sentence>, "type": "conflict or baseless", "tactic": "the way you introduced the error", "sentence": "the rewritten sentence", "evidence": "...", "reason": "...", "original_evidence": "...", "original_reason": "..."}}
```

### 第一代生成器的改法清单（`ragtruth.synth.FOCUS_TACTICS`）

```text
[conflict]
- attribute a statement, action or property to the wrong person, entity, passage or step (who said or did what)
- splice a detail from a different part of the source (another passage, step, entity or review) into the claim
- shift a number, date, duration or count slightly (for example merge or split counts that the source gives separately)
- strengthen or weaken certainty or scope (possible vs certain, some vs all)
- change the order of steps or events
- reverse a relation the source states (different vs the same, before vs after)
[baseless]
- generalize what one part of the source says to a broader subject
- link two facts that the source states separately (a cause, a purpose, a lawsuit, a consequence)
```
