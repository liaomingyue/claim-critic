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


import argparse
import json
from pathlib import Path

SEV = {"clear", "minor"}
TYPES = {"conflict", "baseless"}
MECH = {"wrong_attribution", "near_synonym_shift", "certainty_or_scope_shift", "number_or_date", "merged_across_passages",
        "added_specific_fact", "vague_evaluation_or_addon", "unsupported_inference", "contradicts_source", "other"}
SAT = {"answers_task": {"yes", "partial", "no"}, "completeness": {"complete", "missing_some", "missing_much"},
       "style": {"ok", "problem"}, "sentence_form": {"ok", "problem"}, "overall": {"good", "not_good"}}
FIX_DIR = Path("outputs/rt/minicpm/s1")


def load_fixes(d: Path = FIX_DIR) -> dict:
    from common.llm import kept
    out = {}
    for f in ("fix_raw.jsonl", "fix_single_raw.jsonl"):
        if (d / f).exists():
            for b in kept(d / f):
                for r in b["rows"]:
                    out[r["id"]] = r
    return out


def items(prompts: str = "outputs/rt/minicpm/grpo/train_ref.jsonl", fix_dir: Path = FIX_DIR) -> list[dict]:
    """有改对结果的全部回答（807）：原文、原回答句子、改动；整段重写的给重写全文。"""
    from common.schema import read_jsonl
    from train.ans_reward import sentences
    fixes = load_fixes(fix_dir)
    xs = []
    for r in read_jsonl(prompts):
        if not r.get("ref_text") or r["id"] not in fixes:
            continue
        f = fixes[r["id"]]
        xs.append({"id": r["id"], "source": r["source"], "sentences": sentences(r["ref_text"].strip()),
                   "fixes": [] if f["rewrite"] else f["fixes"], "rewrite": f["rewrite"]})
    return xs


def item_text(x: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["sentences"], 1))
    if x["rewrite"]:
        corr = "The whole response was replaced by this corrected response (there are no per-sentence corrections):\n" + x["rewrite"].strip()
    elif x["fixes"]:
        corr = "\n".join(f'[S{f["idx"]}] -> {f["new"] if f["new"] else "(deleted: nothing supported can be said in its place)"}' for f in x["fixes"])
    else:
        corr = "(none: every sentence is correct; the corrected response is the same as the original)"
    return (f'<item id="{x["id"]}">\n<source>\n{x["source"].strip()}\n</source>\n<original_response>\n{sents}\n</original_response>\n'
            f'<corrections>\n{corr}\n</corrections>\n</item>')


def valid(y, x: dict) -> dict | None:
    if not isinstance(y, dict):
        return None
    want = [f["idx"] for f in x["fixes"]]
    errs = y.get("errors") if isinstance(y.get("errors"), list) else []
    got = {}
    for e in errs:
        if not isinstance(e, dict) or e.get("idx") not in want:
            continue
        if e.get("severity") not in SEV or e.get("type") not in TYPES:
            return None
        got[e["idx"]] = {"idx": e["idx"], "severity": e["severity"], "type": e["type"],
                         "mechanism": e.get("mechanism") if e.get("mechanism") in MECH else "other",
                         **{k: str(e.get(k) or "").strip() for k in ("evidence", "reason", "fixed_evidence", "fixed_reason")}}
    if set(got) != set(want):
        return None
    sat = {}
    for side in ("original", "corrected"):
        s = y.get(side)
        if not isinstance(s, dict) or any(s.get(k) not in v for k, v in SAT.items()):
            return None
        sat[side] = {**{k: s[k] for k in SAT}, "note": str(s.get("note") or "")}
    return {"errors": [got[i] for i in want], **sat}


def cmd_run(a) -> None:
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xs = items()
    calls = [{"id": f"l{k // a.per_call:04d}", "xs": xs[k:k + a.per_call]} for k in range(0, len(xs), a.per_call)]
    assert len(calls) <= a.max_calls, f"{len(calls)} calls > max_calls {a.max_calls}"
    if a.limit_calls:
        calls = calls[:a.limit_calls]
    llm = LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency, max_tokens=32000))

    def run(c: dict) -> dict:
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": "\n\n".join(item_text(x) for x in c["xs"])}]))
        got = {str(y.get("id")): y for y in (o.get("items") or []) if isinstance(y, dict)} if isinstance(o, dict) else {}
        rows = [{"id": x["id"], **v} for x in c["xs"] if (v := valid(got.get(x["id"]), x)) is not None]
        return {"id": c["id"], "rows": rows, "missing": [x["id"] for x in c["xs"] if x["id"] not in {r["id"] for r in rows}]}

    map_resumable(calls, run, out / "label_raw.jsonl", concurrency=a.concurrency)
    if a.limit_calls:
        print("CORE_LABEL_TRIAL", len(calls), flush=True)
        return
    got = {r["id"] for b in kept(out / "label_raw.jsonl") for r in b["rows"]}
    miss = [x for x in xs if x["id"] not in got]
    if miss:
        assert len(miss) <= a.max_single, f"{len(miss)} missing > max_single {a.max_single}"
        map_resumable([{"id": f"s-{x['id']}", "xs": [x]} for x in miss], run, out / "label_single_raw.jsonl", concurrency=a.concurrency)
    print("CORE_LABEL_DONE", json.dumps({"answers": len(xs), "calls": len(calls), "single": len(miss)}), flush=True)


def load_labels(d: str | Path = "outputs/rt/core") -> dict:
    from common.llm import kept
    out = {}
    for f in ("label_raw.jsonl", "label_single_raw.jsonl"):
        p = Path(d) / f
        if p.exists():
            for b in kept(p):
                for r in b["rows"]:
                    out[r["id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out", default="outputs/rt/core")
    ap.add_argument("--per_call", type=int, default=10)
    ap.add_argument("--max_calls", type=int, default=90)
    ap.add_argument("--limit_calls", type=int, default=0, help="只跑前 N 次（试跑看质量）")
    ap.add_argument("--max_single", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    {"run": cmd_run}[ap.parse_args().cmd](ap.parse_args())


if __name__ == "__main__":
    main()
