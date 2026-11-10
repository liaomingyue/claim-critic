"""造错样本统一审计（查种子集正确性 + 2B / 9B 错误改写模型同口径比较；约 40 次 Claude 调用）。

同一个提示词审三组，类型按种子集（ragtruth/core_label.py）的约定：conflict = 原文说的与之不相容（含范围 / 确定性偏移、改变意思的近义词），
baseless = 原文没说它添加的内容。每条判：
  severity     改写句（种子集里是真实错句）对原文：clear / minor / none
  true_type    conflict / baseless（severity none 时 n/a）
  evidence_ok  声明的证据是不是错误所涉及的那段原文
  reason_ok    理由是否说中真正的错
  fix_evidence / fix_reason   evidence_ok / reason_ok 为 false 时给出改正版（证据逐字摘自原文），用于修种子集
组：
  seed  D3 种子 SFT 训练集（outputs/rt/core/seed_sft_train.jsonl）随机 200 条
  2b    改写模型 v0（gen-core-mc）在 sevpool 上的样本，和 9B 同一批提示配对
  9b    gen-core-9b 在同一批提示上的样本（samples_9b_audit.jsonl，每个提示取第一条合格的）

  python -m ragtruth.gen_audit build --group seed --n 200
  python -m ragtruth.gen_audit build --group pair --n 300        # 2b + 9b 同提示配对
  python -m ragtruth.gen_audit label --group seed --max_calls 10
  python -m ragtruth.gen_audit report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")
OUT = D / "gen_audit"

SYSTEM = """You audit sentences that contain a planted or real error, together with the explanation attached to the error. Each item gives the source (the exact input an assistant received), the assistant's response split into numbered sentences, the target sentence S<k> in its ERRONEOUS version, the CORRECT version of the same sentence, and the claimed explanation: type, the source evidence it points to, and a reason.

Judge in this order:
1. severity of the ERRONEOUS sentence against the source (use other sentences only to resolve references):
   - "clear": a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
   - "minor": technically unsupported with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
   - "none": not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source.
2. true_type (only if severity is not none, else "n/a"): "conflict" if the source states something incompatible with the erroneous sentence (this includes a changed certainty or scope such as possible vs certain or some vs all, a near-synonym that changes the meaning, wrong attribution, wrong number, date or order); "baseless" if the source says nothing about what the sentence adds.
3. evidence_ok (true / false, "n/a" if severity is none): the claimed evidence is the part of the source the error is about (for conflict, what the source says instead; for baseless, the most closely related part).
4. reason_ok (true / false, "n/a" if severity is none): the reason names the actual erroneous claim and correctly says why it is wrong.
5. If evidence_ok is false: fix_evidence = the right evidence, copied word for word from the source, one contiguous stretch, as short as possible (empty if the source has nothing related). Otherwise "".
6. If reason_ok is false: fix_reason = one concrete English sentence of at most 30 words ("The source says X, while the sentence says Y" or "The source does not mention Z"). Otherwise "".

Reply with one JSON object only: {"items": [{"id": ..., "severity": "clear"|"minor"|"none", "true_type": "conflict"|"baseless"|"n/a", "evidence_ok": true|false|"n/a", "reason_ok": true|false|"n/a", "fix_evidence": "...", "fix_reason": "..."}, ...]}, one entry per item, in the given order."""


def text(x: dict) -> str:
    k = int(x["idx"])
    sents = "\n".join(f"[S{i}] {t}" for i, t in enumerate(x["context"], 1))
    return (f"<item id=\"{x['aid']}\">\n<source>\n{x['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Target sentence: S{k}\nERRONEOUS S{k}: {x['sentence']}\nCORRECT S{k}: {x['original']}\n"
            f"Claimed type: {x['type']}\nClaimed evidence: {x['evidence']}\nClaimed reason: {x['reason']}\n</item>")


def lst(v):
    return v if isinstance(v, list) else json.loads(v)


def from_sample(x: dict, group: str) -> dict:
    return {"aid": f"{group}/{x['id']}", "group": group, "id": x["id"], "prompt_id": x.get("prompt_id"), "task_type": x["task_type"], "idx": int(x["idx"]),
            "context": lst(x["context"]), "source": x["source"], "sentence": x["sentence"], "original": x["original"], "type": x["type"],
            "evidence": x["evidence"], "reason": x["reason"], "tactic_raw": x.get("tactic_raw")}


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_build(a) -> None:
    if a.group == "seed":
        P = {p["id"]: p for p in read_jsonl(D / "pairs.jsonl")}
        L = sorted(read_jsonl(D / "seed_sft_train.jsonl"), key=lambda x: x["id"])
        out = []
        for x in random.Random("gen_audit|seed").sample(L, a.n):
            p, y = P[x["id"]], json.loads(x["completion"].replace("<|im_end|>", "").strip())
            sents, k = list(lst(p["sentences"])), int(p["idx"])
            ctx = sents[:k - 1] + [p["real"]] + sents[k:]
            out.append({"aid": f"seed/{x['id']}", "group": "seed", "id": x["id"], "task_type": x["task_type"], "idx": k, "context": ctx, "source": p["source"],
                        "sentence": p["real"], "original": p["repaired"], "type": y["type"], "evidence": y["evidence"], "reason": y["reason"],
                        "severity_label": p["severity"], "mechanism": p["mechanism"]})
        write(OUT / "items_seed.jsonl", out)
        print("BUILD seed", len(out), dict(collections.Counter((r["type"], r["severity_label"]) for r in out)), flush=True)
        return
    s2 = {x["prompt_id"]: x for x in read_jsonl(D / "sevpool.jsonl")}
    s9 = {}
    for x in read_jsonl(D / "samples_9b_audit.jsonl"):
        if x.get("why") == "ok" and x["prompt_id"] not in s9:
            s9[x["prompt_id"]] = x
    both = sorted(set(s2) & set(s9))
    pick = random.Random("gen_audit|pair").sample(both, min(a.n, len(both)))
    out = [from_sample(s2[p], "2b") for p in pick] + [from_sample(s9[p], "9b") for p in pick]
    random.Random(1).shuffle(out)   # 两组混在同一批里送审，避免批次效应
    write(OUT / "items_pair.jsonl", out)
    n9 = collections.Counter(x["prompt_id"] for x in read_jsonl(D / "samples_9b_audit.jsonl"))
    ok9 = collections.Counter(x["prompt_id"] for x in read_jsonl(D / "samples_9b_audit.jsonl") if x.get("why") == "ok")
    print("BUILD pair", len(pick), "prompts;", f"9b ok-rate {sum(ok9.values())}/{sum(n9.values())};", f"prompts with 2b & 9b {len(both)}", flush=True)


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    items = list(read_jsonl(OUT / f"items_{a.group}.jsonl"))
    batches = [{"id": f"b{k // a.batch}", "xs": items[k:k + a.batch]} for k in range(0, len(items), a.batch)]
    assert len(batches) <= a.max_calls, f"{len(batches)} calls > max_calls {a.max_calls}"
    llm = LLM(LLMConfig(model=a.model, provider="claude_cli", effort=a.effort, concurrency=a.concurrency))
    raw = OUT / f"items_{a.group}_raw.jsonl"

    def run(b: dict) -> dict:
        t = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(text(x) for x in b["xs"])}])
        o = parse_json_obj(t)
        ys = o.get("items") if isinstance(o, dict) else None
        by = {str(y.get("id")): y for y in ys if isinstance(y, dict)} if isinstance(ys, list) else {}
        hit = [(x, by[x["aid"]]) for x in b["xs"] if x["aid"] in by]
        return {"id": b["id"], "missing": [x["aid"] for x in b["xs"] if x["aid"] not in by],
                "rows": [{"aid": x["aid"], "claude": y} for x, y in hit] if hit else None, **({} if hit else {"raw_tail": str(t)[-400:]})}

    map_resumable(batches, run, raw, concurrency=a.concurrency)
    rows = [r for b in kept(raw) for r in (b["rows"] or [])]
    write(OUT / f"labels_{a.group}.jsonl", rows)
    print("GEN_AUDIT_LABEL", a.group, len(rows), "of", len(items), "failed batches", sum(b["rows"] is None for b in kept(raw)), flush=True)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def table(rows: list[tuple[dict, dict]]) -> dict:
    n = len(rows)
    sev = collections.Counter(c.get("severity") for _, c in rows)
    err = [(x, c) for x, c in rows if c.get("severity") in ("clear", "minor")]
    m = max(len(err), 1)
    t = sum(x["type"] == c.get("true_type") for x, c in err)
    e = sum(c.get("evidence_ok") is True for _, c in err)
    r = sum(c.get("reason_ok") is True for _, c in err)
    al = sum(x["type"] == c.get("true_type") and c.get("evidence_ok") is True and c.get("reason_ok") is True for x, c in err)
    return {"n": n, "clear": sev["clear"] / n, "minor": sev["minor"] / n, "none": sev["none"] / n, "type_ok|err": t / m, "evidence_ok|err": e / m,
            "reason_ok|err": r / m, "all_ok|err": al / m, "err_and_all_ok": al / n, "clear_and_all_ok": sum(
                c.get("severity") == "clear" and x["type"] == c.get("true_type") and c.get("evidence_ok") is True and c.get("reason_ok") is True for x, c in rows) / n}


def cmd_report(a) -> None:
    res = {}
    for g, f in (("seed", "seed"), ("2b", "pair"), ("9b", "pair")):
        ip, lp = OUT / f"items_{f}.jsonl", OUT / f"labels_{f}.jsonl"
        if not lp.exists():
            continue
        it = {x["aid"]: x for x in read_jsonl(ip)}
        rows = [(it[r["aid"]], r["claude"]) for r in read_jsonl(lp) if r["aid"] in it and it[r["aid"]]["group"] == g]
        res[g] = table(rows)
        for task in ("QA", "Summary"):
            res[f"{g}/{task}"] = table([(x, c) for x, c in rows if x["task_type"] == task])
        if g == "seed":
            res["seed/fix_evidence_verbatim"] = sum(bool(norm(c.get("fix_evidence"))) and norm(c.get("fix_evidence")) in norm(x["source"])
                                                    for x, c in rows if c.get("evidence_ok") is False) / max(sum(c.get("evidence_ok") is False for _, c in rows), 1)
            res["seed/type_confusion"] = {f"{k[0]}->{k[1]}": v for k, v in collections.Counter((x["type"], c.get("true_type")) for x, c in rows).items()}
            res["seed/sev_label_vs_audit"] = {f"{k[0]}->{k[1]}": v for k, v in collections.Counter((x["severity_label"], c.get("severity")) for x, c in rows).items()}
    keys = ["n", "clear", "minor", "none", "type_ok|err", "evidence_ok|err", "reason_ok|err", "all_ok|err", "err_and_all_ok", "clear_and_all_ok"]
    print("| 组 | " + " | ".join(keys) + " |\n|" + " --- |" * (len(keys) + 1))
    for g, t in res.items():
        if isinstance(t, dict) and "n" in t:
            print(f"| {g} | " + " | ".join(str(t[k]) if k == "n" else f"{100 * t[k]:.0f}%" for k in keys) + " |")
    for g, t in res.items():
        if not (isinstance(t, dict) and "n" in t):
            print(g, t)
    write(OUT / "report.jsonl", [{"group": g, **(t if isinstance(t, dict) else {"value": t})} for g, t in res.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "label", "report"])
    ap.add_argument("--group", default="seed", choices=["seed", "pair", "sevnew"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max_calls", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    {"build": cmd_build, "label": cmd_label, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
