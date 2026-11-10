"""错误改写模型 v2 采样的 Claude 标注（约 50 次调用）。

一份标注三个用途：① 生成器可控性（按 gold.COARSE 6 粗类）；② 新版 sev 训练数据（输入带要求的机制 + 改动描述 + 逐词改动）；③ 验证自然度模型。
样本：gen-v2-9b 700 + gen-v2-mc 300（合格样本，按要求的机制分层），留出原文上的采样（genpool_eval）。
给 Claude：原文、改写后的整条回答、原句 / 改写句、要求的机制（名字 + 说明，同 gen_v2.MECH）；不给生成器自己写的描述。
每条判：severity（clear / minor / none）、mechanism（实际机制，9 细类 + other；none 时 none）、matches（实际是否就是要求的那种）、
type（conflict / baseless，同种子集约定）、natural（真实助手会不会在这条回答里这样写：yes / no）。

  python -m ragtruth.gen_v2_label items            # → outputs/rt/core/gen_v2_label/items.jsonl
  python -m ragtruth.gen_v2_label label --max_calls 50 --concurrency 8
  python -m ragtruth.gen_v2_label report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.gen_v2 import MECH

D = Path("outputs/rt/core")
OUT = D / "gen_v2_label"   # --out_dir 可改（同一提示词的另一批）
MECH_LIST = ", ".join(list(MECH) + ["other"])

SYSTEM = f"""You audit sentences that were deliberately edited to plant a specific kind of error. Each item gives the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL (faithful) and EDITED versions of that sentence, and the REQUESTED kind of error.

Judge the EDITED sentence against the source (use the other response sentences only to resolve references):
1. severity:
   - "clear": a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
   - "minor": technically unsupported with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
   - "none": not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source or is stated elsewhere in it.
2. mechanism: the actual kind of error, one of {MECH_LIST}; "none" if severity is none. Definitions:
""" + "\n".join(f"   - {k}: {v}" for k, v in MECH.items()) + """
3. matches: true if severity is not none and the actual mechanism is the requested one; otherwise false.
4. type: "conflict" if the source states something incompatible with the edited sentence (including a changed certainty or scope, a meaning-changing near-synonym, wrong attribution, number, date or order); "baseless" if the source says nothing about what it adds; "n/a" if severity is none.
5. natural: "yes" if a real assistant could plausibly have written the edited sentence in this response (fluent, fits the context and style); "no" if it reads garbled, out of place or obviously artificial.

Reply with one JSON object only: {"items": [{"id": ..., "severity": "...", "mechanism": "...", "matches": true|false, "type": "...", "natural": "yes"|"no"}, ...]}, one entry per item, in the given order."""


def text(x: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["context"], 1))
    return (f'<item id="{x["aid"]}">\n<source>\n{x["source"].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n\n'
            f'Edited sentence: S{x["idx"]}\nORIGINAL: {x["original"]}\nEDITED: {x["sentence"]}\n'
            f'REQUESTED kind of error: {x["mechanism"]}: {MECH[x["mechanism"]]}\n</item>')


def cmd_items(a) -> None:
    global OUT
    OUT = D / a.out_dir
    out = []
    for g, f, n in (("9b", "samples_v2_9b.jsonl", a.n9), ("2b", "samples_v2_mc.jsonl", a.n2)):
        xs = sorted((x for x in read_jsonl(D / f) if x.get("why") == "ok"), key=lambda x: x["id"])
        by = collections.defaultdict(list)
        for x in xs:
            by[x["mechanism"]].append(x)
        rng = random.Random(f"gen_v2_label|{g}")
        per = -(-n // len(by))
        pick = []
        for m in sorted(by):
            rng.shuffle(by[m])
            pick += by[m][:per]
        rng.shuffle(pick)
        out += [{**x, "aid": f"{g}/{x['id']}", "group": g} for x in pick[:n]]
    random.Random("gen_v2_label|mix").shuffle(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print("GENV2_LABEL_ITEMS", len(out), dict(collections.Counter(x["group"] for x in out)), flush=True)


def cmd_label(a) -> None:
    global OUT
    OUT = D / a.out_dir
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    items = list(read_jsonl(OUT / "items.jsonl"))
    batches = [{"id": f"b{k // a.batch}", "xs": items[k:k + a.batch]} for k in range(0, len(items), a.batch)]
    assert len(batches) <= a.max_calls, f"{len(batches)} calls > max_calls {a.max_calls}"
    llm = LLM(LLMConfig(model="claude-opus-5-5", provider="claude_cli", effort="high", concurrency=a.concurrency))
    raw = OUT / "labels_raw.jsonl"

    def run(b):
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(text(x) for x in b["xs"])}]))
        ys = {str(y.get("id")): y for y in (o.get("items") or []) if isinstance(y, dict)} if isinstance(o, dict) else {}
        return {"id": b["id"], "rows": [{"aid": x["aid"], "claude": ys[x["aid"]]} for x in b["xs"] if x["aid"] in ys] or None}

    map_resumable(batches, run, raw, concurrency=a.concurrency)
    rows = [r for b in kept(raw) for r in (b["rows"] or [])]
    (OUT / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("GENV2_LABEL", len(rows), "of", len(items), "failed batches", sum(b["rows"] is None for b in kept(raw)), flush=True)


def cmd_report(a) -> None:
    global OUT
    OUT = D / a.out_dir
    from ragtruth.gold import COARSE
    it = {x["aid"]: x for x in read_jsonl(OUT / "items.jsonl")}
    rows = [(it[r["aid"]], r["claude"]) for r in read_jsonl(OUT / "labels.jsonl") if r["aid"] in it]
    pc = lambda v, n: f"{100 * v / max(n, 1):.0f}%"
    for g in ("9b", "2b"):
        rs = [(x, c) for x, c in rows if x["group"] == g]
        n = len(rs)
        err = [(x, c) for x, c in rs if c.get("severity") in ("clear", "minor")]
        coarse_ok = sum(COARSE.get(c.get("mechanism"), "?") == COARSE[x["mechanism"]] for x, c in err)
        print(f"\n== {g}  n={n}  真错 {pc(len(err), n)}（显然 {pc(sum(c.get('severity') == 'clear' for _, c in rs), n)}）"
              f"  细类一致 {pc(sum(bool(c.get('matches')) for _, c in rs), n)}  粗类一致（占全部）{pc(coarse_ok, n)}  自然 {pc(sum(c.get('natural') == 'yes' for _, c in rs), n)}")
        print("| 要求的机制 | n | 真错 | 细类一致 | 粗类一致 | 自然 | 实际最多 |")
        for m in MECH:
            ms = [(x, c) for x, c in rs if x["mechanism"] == m]
            if not ms:
                continue
            e = [(x, c) for x, c in ms if c.get("severity") in ("clear", "minor")]
            top = collections.Counter(c.get("mechanism") for _, c in ms).most_common(2)
            print(f"| {m} | {len(ms)} | {pc(len(e), len(ms))} | {pc(sum(bool(c.get('matches')) for _, c in ms), len(ms))} | "
                  f"{pc(sum(COARSE.get(c.get('mechanism'), '?') == COARSE[m] for _, c in e), len(ms))} | {pc(sum(c.get('natural') == 'yes' for _, c in ms), len(ms))} | {top} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "label", "report"])
    ap.add_argument("--n9", type=int, default=700)
    ap.add_argument("--n2", type=int, default=300)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out_dir", default="gen_v2_label")
    ap.add_argument("--max_calls", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    {"items": cmd_items, "label": cmd_label, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
