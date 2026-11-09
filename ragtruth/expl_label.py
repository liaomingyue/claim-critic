"""SFT 错误生成器解释质量审计：类型 / 证据 / 理由分开判（2026-11-08，用户：看 SFT 生成器的错误是否真错、证据和理由效果）。

样本：outputs/rt/core/sevpool.jsonl（改写模型 v0 = 种子 SFT，在 D4 训练池上的原始输出，未按 critic 筛）里，
Claude 已判为真错（sevpool_labels.jsonl：severity clear / minor）的 524 条。旧标注只有合并的 explanation_ok，这里拆成三项：
  type_ok      声明的类型对不对（conflict = 原文与之矛盾；baseless = 原文没有）；另给 true_type
  evidence_ok  声明的证据是不是错误所涉及的那段原文（baseless 时：与该说法最相关、能看出原文没说的那段）
  reason_ok    理由是否说中了改写句里真正的错
  actual_error 一句话写出真正的错（作为判别器的参考标签）

  python -m ragtruth.expl_label build                       # → outputs/rt/core/expl_items.jsonl
  python -m ragtruth.expl_label label --batch 20 --max_calls 27
  python -m ragtruth.expl_label report
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.sev_label import family

D = Path("outputs/rt/core")

SYSTEM = """You audit the explanation attached to a deliberately planted error. Each item gives the source (the exact input an assistant received), the assistant's response split into numbered sentences with the target sentence S<k> already edited, the ORIGINAL version of that sentence, and the editor's claimed error: its type, the source evidence it points to, and its reason. The EDITED sentence has already been confirmed to contain an error against the source (severity given). Do not re-judge whether it is an error; judge the explanation.

Judge each part separately:
1. true_type: "conflict" if the source contradicts the edited claim (wrong attribution, reversed relation, wrong number/date/order, opposite statement), "baseless" if the source simply does not contain or support it (added specifics, evaluations, unsupported inference, scope or certainty added).
2. type_ok: true if the claimed type equals true_type.
3. evidence_ok: true if the claimed evidence is the part of the source that the error is about (for conflict: the passage text the edited sentence contradicts; for baseless: the source text the claim builds on, from which one can see the added part is not stated). false if it points to an unrelated or only loosely related part, or is not taken from the source.
4. reason_ok: true if the reason correctly identifies what is wrong in the edited sentence (the actual changed claim and why it is wrong). false if it describes a different change, misstates the source, or is too vague to locate the error.
5. actual_error: one short sentence naming the actual error.

Reply with one JSON object only: {"items": [{"id": ..., "true_type": "conflict"|"baseless", "type_ok": true|false, "evidence_ok": true|false, "reason_ok": true|false, "actual_error": "..."}, ...]}, one entry per item, in the given order."""


def text(x: dict) -> str:
    sents = "\n".join(f"[S{i}] {t}" for i, t in enumerate(x["context"], 1))
    k = int(x["idx"])
    return (f"<item id=\"{x['id']}\">\n<source>\n{x['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Target sentence: S{k}\nORIGINAL S{k}: {x['original']}\nEDITED S{k}: {x['sentence']}\nSeverity: {x['severity']}\n"
            f"Claimed type: {x['type']}\nClaimed evidence: {x['evidence']}\nClaimed reason: {x['reason']}\n</item>")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def cmd_build(a) -> None:
    pool = {x["id"]: x for x in read_jsonl(D / "sevpool.jsonl")}
    out = []
    for r in read_jsonl(D / "sevpool_labels.jsonl"):
        c = r["claude"]
        if r["hid"] not in pool or c.get("severity") not in ("clear", "minor") or c.get("original_ok") is False:
            continue
        x = pool[r["hid"]]
        ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
        out.append({**{k: x[k] for k in ("id", "source_id", "task_type", "type", "idx", "sentence", "original", "evidence", "reason", "tactic_raw", "source")},
                    "context": ctx, "fam": family(x.get("tactic_raw")), "severity": c["severity"], "mechanism": c.get("mechanism"),
                    "expl_ok_old": c.get("explanation_ok"), "ev_verbatim": bool(norm(x["evidence"])) and norm(x["evidence"]) in norm(x["source"])})
    out.sort(key=lambda x: x["id"])
    (D / "expl_items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("EXPL_BUILD", len(out), dict(collections.Counter(r["severity"] for r in out)), flush=True)


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    items = list(read_jsonl(D / "expl_items.jsonl"))
    batches = [{"id": f"b{k // a.batch}", "xs": items[k:k + a.batch]} for k in range(0, len(items), a.batch)]
    assert len(batches) <= a.max_calls, f"{len(batches)} calls > max_calls {a.max_calls}"
    llm = LLM(LLMConfig(model=a.model, provider="claude_cli", effort=a.effort, concurrency=a.concurrency))
    raw = D / "expl_items_raw.jsonl"

    def run(b: dict) -> dict:
        t = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(text(x) for x in b["xs"])}])
        o = parse_json_obj(t)
        ys = o.get("items") if isinstance(o, dict) else None
        by = {str(y.get("id")): y for y in ys if isinstance(y, dict)} if isinstance(ys, list) else {}
        hit = [(x, by[x["id"]]) for x in b["xs"] if x["id"] in by]
        return {"id": b["id"], "missing": [x["id"] for x in b["xs"] if x["id"] not in by],
                "rows": [{"hid": x["id"], "claude": y} for x, y in hit] if hit else None, **({} if hit else {"raw_tail": str(t)[-400:]})}

    map_resumable(batches, run, raw, concurrency=a.concurrency)
    rows = [r for b in kept(raw) for r in (b["rows"] or [])]
    (D / "expl_labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("EXPL_LABEL", len(rows), "failed batches", sum(b["rows"] is None for b in kept(raw)), flush=True)


def cmd_report(a) -> None:
    it = {x["id"]: x for x in read_jsonl(D / "expl_items.jsonl")}
    rows = [(it[r["hid"]], r["claude"]) for r in read_jsonl(D / "expl_labels.jsonl") if r["hid"] in it]
    pc = lambda v, n: f"{100 * v / max(n, 1):.0f}%"

    def line(name, rs):
        n = len(rs)
        f = lambda k: sum(c.get(k) is True for _, c in rs)
        allok = sum(all(c.get(k) is True for k in ("type_ok", "evidence_ok", "reason_ok")) for _, c in rs)
        return f"| {name} | {n} | {pc(f('type_ok'), n)} | {pc(f('evidence_ok'), n)} | {pc(f('reason_ok'), n)} | {pc(allok, n)} |"

    print("| 分组 | n | 类型对 | 证据对 | 理由对 | 三项全对 |\n| --- | --- | --- | --- | --- | --- |")
    print(line("全部", rows))
    for key, name in (("severity", "严重度"), ("type", "声明类型"), ("task_type", "任务"), ("fam", "改法"), ("ev_verbatim", "证据是原文原话")):
        for v in sorted({x[key] for x, _ in rows}, key=str):
            print(line(f"{name}={v}", [(x, c) for x, c in rows if x[key] == v]))
    conf = collections.Counter((x["type"], c.get("true_type")) for x, c in rows)
    print("\n声明类型 → 实际类型", dict(conf))
    old = collections.Counter((x["expl_ok_old"], all(c.get(k) is True for k in ("type_ok", "evidence_ok", "reason_ok"))) for x, c in rows)
    print("旧 explanation_ok vs 新三项全对", dict(old))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "label", "report"])
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max_calls", type=int, default=27)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    {"build": cmd_build, "label": cmd_label, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
