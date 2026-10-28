"""给 RAGTruth 人工标注样本补写逐句的 evidence 和 reason（逐句 critic 训练目标的前两项，见 ragtruth.task）。

一条回答一次调用，一次写出所有句子（同一原文只读一遍）；之后由 ragtruth.build 拆成逐句样本。

人工标注只有幻觉片段、类型和标注员的简短备注，没有证据和说明。这里把真值（哪几句有问题、什么类型、
标注员备注）交给较大的模型（默认本机 llama.cpp 上的 Qwen3.5-9B Q4_K_M），让它对每一句写出：
- evidence：原文中与这句对应、矛盾或最相关的逐字片段（长片段可写成「开头 ... 结尾」）；
  原文完全没有相关内容或句子没有事实内容时为 none；
- reason：这句与证据的关系（原文说什么、这句说什么），英文，不出现结论词。

校验（不通过就把问题反馈给模型重写，最多 retries 次；仍不通过的记 analysis_ok=false，训练时剔除）：
- 每句一项，句号齐全；
- 给出的 evidence 逐字出现在原文中（归一化空白、引号，不区分大小写）；
- 真值为 conflict 的句子必须给出原文片段；
- reason 里不出现 hallucination / faithful / conflict / baseless 等结论词，避免分析部分泄露结论；
- reason 与真值一致：无问题的句子不能写「原文没有 / 不符」，有问题的句子必须写出差异（ragtruth.task.reason_consistent）；
  模型确信人工标注有误时标 "disputed": true，这些句子不进训练（实验 4 人工修正的素材）。
prompt 带 ragtruth.task.TIPS（结构化数据、同义改写、列表逐项核对等判定要点）。

用法：
  llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 32768 -np 4 --reasoning off --port 8090 --alias qwen3.5-9b
  python -m ragtruth.annotate --data outputs/external/ragtruth/train_sub.jsonl \
      --out outputs/rt/annotated/train_sub.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth.task import TIPS, evidence_in_source, gold_bad, reason_consistent


def render_response(s: dict) -> str:
    """整条回答（不标待检句），补写时一次覆盖所有句子。"""
    lines = ["[source]", s["source"].strip(), "", "[response]"]
    return "\n".join(lines + [f"[S{i}] {t}" for i, t in enumerate(s["sentences"], 1)])

BANNED = re.compile(r"hallucinat|faithful|conflict|baseless", re.I)

PROMPT = """You are preparing training data for a fact-checking model. Below are a source (the exact input an assistant received) and the assistant's response split into numbered sentences.

{input}

Ground truth from human annotators (use it to write the analysis, but do not reveal it):
{truth}

For EVERY sentence S1 to S{n}, in order, write:
- "evidence": copy, word for word, one contiguous stretch of the source, as short as possible (usually one clause or sentence), that the sentence corresponds to, contradicts, or is most closely related to. Copy it exactly as it appears in the source (same words, same order). For a long passage, copy its first few words and its last few words joined by " ... " (each part at least 3 words). Use null only if the source has nothing related to the sentence, or the sentence has no factual content (greetings, restating the task).{conflict_rule}
- "reason": one English sentence of at most 30 words describing how the sentence relates to the evidence: what the source says and what the sentence says. Describe the relationship concretely (for example "The source says X, while the sentence says Y" or "The source does not mention Z"). Never use the words hallucination, faithful, conflict, or baseless.

Every sentence gets the same kind of analysis, whether or not it has a problem.
The reason must agree with the ground truth: for a sentence not listed above, describe how the source supports it and do not say that anything is missing or different; for a listed sentence, say concretely what is missing from or different in the source.
Human annotators sometimes miss problems or mark too much. If you are confident that the ground truth for a sentence is wrong, add "disputed": true to that object and let the reason describe what you actually see.

{tips}

Output only a JSON array, one object per sentence, nothing else:
[{{"id": "S1", "evidence": "..." or null, "reason": "..."}}, ...]"""


def truth_text(s: dict) -> str:
    gb = gold_bad(s)
    if not gb:
        return "All sentences are supported by the source."
    lines = []
    for i, kind in sorted(gb.items()):
        a, b = s["sent_spans"][i - 1]
        notes = [x for x in s.get("span_notes", []) if x[0] < b and x[1] > a]
        what = ("contradicts the source" if kind == "conflict"
                else "adds information that the source does not contain")
        line = f"- S{i} {what}."
        spans = [s["response"][x[0]:x[1]] for x in s["spans"] if x[0] < b and x[1] > a]
        if spans:
            line += " Problem span: " + "; ".join(f'"{t}"' for t in spans)
        if notes:
            line += " Annotator note: " + " / ".join(n[2] for n in notes)
        lines.append(line)
    return "Sentences not listed are supported by the source.\n" + "\n".join(lines)


def prompt_for(s: dict) -> str:
    conflicts = [f"S{i}" for i, k in sorted(gold_bad(s).items()) if k == "conflict"]
    rule = (f" For {', '.join(conflicts)} the evidence must be the part of the source that the sentence contradicts"
            " (never null).") if conflicts else ""
    return PROMPT.format(input=render_response(s), truth=truth_text(s), n=len(s["sentences"]), conflict_rule=rule,
                         tips=TIPS)


def parse_analysis(text: str) -> list | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return items if isinstance(items, list) else None


def check(items: list | None, s: dict) -> list[str]:
    """返回问题列表，空列表表示通过。"""
    n = len(s["sentences"])
    if items is None:
        return ["output is not a valid JSON array"]
    if [it.get("id") if isinstance(it, dict) else None for it in items] != [f"S{i}" for i in range(1, n + 1)]:
        return [f"the array must have exactly one object per sentence, ids S1..S{n} in order"]
    problems = []
    gb = gold_bad(s)
    for i, it in enumerate(items, 1):
        ev, reason = it.get("evidence"), (it.get("reason") or "").strip()
        disputed = it.get("disputed") is True
        if ev is not None and not (isinstance(ev, str) and evidence_in_source(ev, s["source"])):
            problems.append(f"S{i}: evidence is not copied word for word from the source: {str(ev)[:80]!r}")
        if ev is None and gb.get(i) == "conflict" and not disputed:
            problems.append(f"S{i}: evidence must quote the part of the source it contradicts, not null")
        if not reason:
            problems.append(f"S{i}: reason is empty")
        elif BANNED.search(reason):
            problems.append(f"S{i}: reason must not use the words hallucination, faithful, conflict, baseless")
        elif not disputed and not reason_consistent(reason, gb.get(i, "none")):
            problems.append(f"S{i}: the reason does not agree with the ground truth"
                            + (" (it must say what is missing or different)" if i in gb
                               else " (the sentence is supported; do not say anything is missing or different)")
                            + ', or mark it "disputed": true if you are confident the ground truth is wrong')
    return problems


def annotate_one(llm: LLM, s: dict, retries: int, extra: dict | None = None) -> dict:
    msgs = [{"role": "user", "content": prompt_for(s)}]
    best: tuple[list | None, list[str]] = (None, ["no attempt"])
    for attempt in range(retries + 1):
        text = llm.chat_one(msgs, temperature=0.3 if attempt == 0 else 0.7, **(extra or {}))
        text = text.split("</think>")[-1]   # 思考模式下只取最终回答
        items = parse_analysis(text)
        problems = check(items, s)
        # 保留问题最少的一次：重试时偶尔输出坏 JSON，不能把前一次基本合格的结果丢掉（逐句取舍在 ragtruth.build）
        if items is not None and (best[0] is None or len(problems) < len(best[1])):
            best = (items, problems)
        if not problems:
            break
        msgs = msgs[:1] + [{"role": "assistant", "content": text},
                           {"role": "user", "content": "Fix these problems and output the full JSON array again:\n- "
                                                       + "\n- ".join(problems[:10])}]
    items, problems = best
    ok = not problems
    analysis = ([{"evidence": it.get("evidence"), "reason": (it.get("reason") or "").strip(),
                  "disputed": it.get("disputed") is True} for it in items]
                if items is not None and len(items) == len(s["sentences"]) else None)
    return {"analysis": analysis, "analysis_ok": ok, "analysis_problems": problems[:10], "attempts": attempt + 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/external/ragtruth/train_sub.jsonl")
    ap.add_argument("--out", default="outputs/rt/annotated/train_sub.jsonl")
    ap.add_argument("--raw_out", default=None, help="逐条生成结果（可续跑）；默认 <out>.raw.jsonl")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="Qwen3.5 开思考模式（默认关，llama.cpp 用 --reasoning off 时本参数无效）")
    ap.add_argument("--faithful_responses", type=int, default=-1,
                    help="含幻觉的回答全部补写，全对的回答只抽这么多条；-1 = 全部")
    args = ap.parse_args()

    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=args.max_tokens,
                        concurrency=args.concurrency, timeout=600))
    samples = list(read_jsonl(args.data))
    if args.faithful_responses >= 0:  # 训练集按句下采样无问题句，不必每条全对的回答都补写
        bad = [s for s in samples if s["bad"]]
        good = [s for s in samples if not s["bad"]]
        samples = bad + random.Random(0).sample(good, min(len(good), args.faithful_responses))
    if args.limit:
        samples = samples[: args.limit]
    raw_out = args.raw_out or args.out.replace(".jsonl", ".raw.jsonl")
    extra = None if args.think else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    map_resumable(samples, lambda s: annotate_one(llm, s, args.retries, extra), raw_out, concurrency=args.concurrency)

    res = {r["id"]: r for r in kept(raw_out)}
    rows = [{**s, **{k: res[s["id"]][k] for k in ("analysis", "analysis_ok", "analysis_problems")}}
            for s in samples if s["id"] in res]
    good = [r for r in rows if r["analysis_ok"]]
    print(f"annotated {len(rows)}/{len(samples)}, passed checks {len(good)}; "
          f"wrote {write_jsonl(args.out, rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
