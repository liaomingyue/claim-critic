"""抽查 RM3 判「错了但解释不对」（B）的合成句是不是真错（抽 20 条）：
a0（起点 gen-seed4-mc）快评合格样本里 82–85% 是 B 主导（P(A) < 0.7、P(A) + P(B) ≥ 0.7）；err_mode ab 把它们算真错，这里让 Claude 按严格边界复核。
每批 10 条，只判目标句（生成器改写过的那句）对原文有没有错，不看生成器写的类型 / 证据 / 理由。

  python -m ragtruth.b_check --items outputs/rt/gentarget/grpo4_a0/b_check_items.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.synth import parse_json_obj

SYSTEM = """You check sentences in responses written from a source. Each item gives a source, a response split into numbered sentences, and one target sentence. Judge only the target sentence against the source; use the other sentences only to resolve pronouns and references.

The target sentence is an error whenever it says anything the source does not state and that cannot be directly inferred from it; when you are unsure whether the source supports it, call it an error. Errors include changed or invented numbers, dates, names or quantities; facts tied to the wrong entity, passage or step; changed certainty or scope; added details, evaluations, purposes or consequences; near-synonyms that shift the meaning; details from different passages joined into a claim the source does not make. Not errors: true synonyms, faithful summaries and reorderings, leaving information out, what follows directly from the source.

For each item give: "id"; "is_error": true or false; "type": "conflict", "baseless" or "none"; "severity": "clear", "minor" or "none"; "note": one short sentence.
Reply with one JSON object only: {"items": [{...}, ...]}, one entry per item, in the given order."""


def text(x: dict) -> str:
    sents = "\n".join(f"[S{k}] {t}" for k, t in enumerate(x["sentences"], 1))
    return f"<item id=\"{x['id']}\">\n<source>\n{x['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n\nTarget sentence: S{x['idx']}\n</item>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max_calls", type=int, default=2)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    a = ap.parse_args()
    items = list(read_jsonl(a.items))
    batches = [{"id": f"b{k // a.batch}", "xs": items[k:k + a.batch]} for k in range(0, len(items), a.batch)]
    assert len(batches) <= a.max_calls
    llm = LLM(LLMConfig(model=a.model, provider="claude_cli", effort=a.effort, concurrency=2))
    out = Path(a.items).with_name("b_check_raw.jsonl")

    def run(b: dict) -> dict:
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(text(x) for x in b["xs"])}]))
        ys = o.get("items") if isinstance(o, dict) else None
        ok = isinstance(ys, list) and [y.get("id") for y in ys] == [x["id"] for x in b["xs"]]
        return {"id": b["id"], "rows": [{**{k: x[k] for k in ("id", "step", "type", "p_a", "p_correct", "sentence", "original")}, **{k: x[k] for k in ("grp", "p_halluc_2b", "p_natural") if k in x}, "claude": y}
                                        for x, y in zip(b["xs"], ys)] if ok else None}

    map_resumable(batches, run, out, concurrency=2)
    rows = [r for b in kept(out) for r in (b["rows"] or [])]
    Path(a.items).with_name("b_check.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    c = [r["claude"] for r in rows]
    print("B_CHECK", json.dumps({"n": len(rows), "is_error": sum(bool(x.get("is_error")) for x in c), "severity": dict(collections.Counter(x.get("severity") for x in c)),
                                 "type_match": sum(x.get("type") == r["type"] for x, r in zip(c, rows)), "by_step": {s: f"{sum(bool(r['claude'].get('is_error')) for r in rows if r['step'] == s)}/{sum(r['step'] == s for r in rows)}" for s in (0, 50)}},
                                ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
