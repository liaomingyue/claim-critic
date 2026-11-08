"""严重程度 RM 的 Claude 标注（2026-11-04，用户：从 2B 错误生成器里抽样本给 Claude 标选择题——显然错误 / 非显然错误 / 没有错误，
再判生成器给的证据和理由对不对；先标 100 条看看，别一次全跑）。

样本：h1 造数据（outputs/rt/gentarget/harvest_h1.jsonl，2B 生成器 a7 第 50 步，过硬规则 why=ok 的 40707 条），去掉已有 Claude 标签的 504 条，
每篇原文最多 2 条。分层：改法 2/3 取目标类（张冠李戴 / 关系或顺序反转 / 数字 / 跨段拼接），1/3 取其余；
过 RM3 门槛（≥ 0.7）且 2B critic P < 0.5 占 40%、过门槛且 P ≥ 0.5 占 20%、没过门槛（没有 critic 分）占 40%。
Claude 分字段标：original_ok、severity（clear / minor / none）、explanation_ok（类型 + 证据 + 理由都对得上真正的错误；没错时 n/a）、mechanism、note。
训练时合并成 6 选 1：A 显然错 + 解释对、B 显然错 + 解释不对、C 非显然错 + 解释对、D 非显然错 + 解释不对、E 没有错、F 原句本身有错。

  python -m ragtruth.sev_label pick --n 100 --out outputs/rt/sevlabel/items_p1.jsonl     # 远端（读共享盘 harvest）
  python -m ragtruth.sev_label label --items outputs/rt/sevlabel/items_p1.jsonl --max_calls 10
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl

MECH = ("wrong_attribution", "contradicts_source", "merged_across_passages", "number_or_date", "certainty_or_scope_shift", "added_specific_fact",
        "vague_evaluation_or_addon", "near_synonym_shift", "unsupported_inference", "other", "none")

SYSTEM = """You audit sentences that were deliberately edited to plant an error. Each item gives the source (the exact input an assistant received), the assistant's response split into numbered sentences with the target sentence S<k> already edited, the ORIGINAL version of that sentence before editing, and the editor's claimed error: its type, the source evidence it points to, and its reason.

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

Reply with one JSON object only: {"items": [{"id": ..., "original_ok": true|false, "severity": "clear"|"minor"|"none", "explanation_ok": true|false|"n/a", "mechanism": "...", "note": "..."}, ...]}, one entry per item, in the given order."""

FAMILY = (("attribute", "attribution"), ("reverse", "contradiction"), ("order", "contradiction"), ("number", "number"), ("count", "number"), ("splice", "merge"))


def family(t: str) -> str:
    t = (t or "").lower()
    return next((f for k, f in FAMILY if k in t), "other")


def option(c: dict) -> str | None:
    """Claude 的字段 → 6 选 1；字段自相矛盾返回 None。"""
    if c.get("original_ok") is False:
        return "F"
    s, e = c.get("severity"), c.get("explanation_ok")
    if s == "none":
        return "E" if e in ("n/a", None) else None
    if s not in ("clear", "minor") or not isinstance(e, bool):
        return None
    return {("clear", True): "A", ("clear", False): "B", ("minor", True): "C", ("minor", False): "D"}[(s, e)]


V2 = (("gm_mid", .35), ("gm_hi", .25), ("gm_lo", .15), ("ng_mid", .15), ("ng_lo", .10))
V3 = (("gm_mid", .40), ("gm_hi", .30), ("gm_lo", .15), ("ng_mid", .15))
V4 = (("gm_mid", .70), ("gm_lo", .30))   # 第四批（用户：再补标一批盲区样本）：只抽过 RM3 门槛、critic < 0.5 的   # 第三批（用户）：去掉 RM3 < 0.3 那一层（两批里显然错误 0/20）


def cell_v2(x: dict) -> str | None:
    """第二批（用户：显然错误太少，抽样时多加一些；按试标里显然错误比例定的分层）。"""
    pc, ph = float(x["p_correct"]), x.get("p_halluc")
    if pc >= 0.7 and ph is not None:
        ph = float(ph)
        return "gm_hi" if ph >= 0.5 else "gm_mid" if ph >= 0.2 else "gm_lo"
    if pc >= 0.7:
        return None   # 过门槛但原句被判错（没打 critic 分），不抽
    return "ng_mid" if pc >= 0.3 else "ng_lo"


def cmd_pick_v2(a) -> None:
    """每层里矛盾 / 数字类改法优先占一半，不够再由其余改法补。"""
    done = {r["hid"] for r in read_jsonl(a.labeled)} | {x["id"] for p in a.exclude for x in read_jsonl(p)}
    xs = [x for x in read_jsonl(a.harvest) if x.get("why") == "ok" and x["id"] not in done and x.get("p_correct") not in (None, "")]
    rng = random.Random(f"{a.seed}|sevlabel_v2")
    xs.sort(key=lambda x: x["id"])
    rng.shuffle(xs)
    per, out = collections.Counter(), []
    for cell, w in {"v2": V2, "v3": V3, "v4": V4}[a.scheme]:
        n = round(a.n * w)
        pool = [x for x in xs if cell_v2(x) == cell]
        pri = [x for x in pool if family(x.get("tactic_raw")) in ("contradiction", "number")]
        rest = [x for x in pool if family(x.get("tactic_raw")) not in ("contradiction", "number")]
        got = []
        for src, cap in ((pri, n // 2), (rest, n), (pri, n)):   # 先矛盾 / 数字到一半，再其余，最后矛盾 / 数字补满
            for x in src:
                if len(got) >= cap:
                    break
                if x in got or per[x["source_id"]] >= 2:
                    continue
                got.append(x)
                per[x["source_id"]] += 1
        for x in got:
            out.append({**{k2: x.get(k2) for k2 in ("id", "source_id", "task_type", "type", "idx", "sentence", "original", "evidence", "reason", "tactic_raw",
                                                     "p_correct", "p_a", "p_halluc", "orig_p", "source", "context")}, "fam": family(x.get("tactic_raw")), "cell": cell})
    rng.shuffle(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("PICK_V2", a.out, len(out), dict(collections.Counter((r["cell"], r["fam"]) for r in out)), flush=True)


def cmd_pick(a) -> None:
    done = {r["hid"] for r in read_jsonl(a.labeled)}
    xs = [x for x in read_jsonl(a.harvest) if x.get("why") == "ok" and x["id"] not in done and x.get("p_correct") not in (None, "")]
    for x in xs:
        x["fam"] = family(x.get("tactic_raw"))
        pc, ph = float(x["p_correct"]), x.get("p_halluc")
        x["cell"] = "gate_miss" if pc >= 0.7 and ph is not None and float(ph) < 0.5 else "gate_hit" if pc >= 0.7 and ph is not None else "nogate"
    rng = random.Random(f"{a.seed}|sevlabel")
    xs.sort(key=lambda x: x["id"])
    rng.shuffle(xs)
    quota = {(cell, tgt): round(a.n * w * (2 / 3 if tgt else 1 / 3)) for cell, w in (("gate_miss", .4), ("gate_hit", .2), ("nogate", .4)) for tgt in (True, False)}
    per, got, out = collections.Counter(), collections.Counter(), []
    for x in xs:
        k = (x["cell"], x["fam"] != "other")
        if got[k] >= quota.get(k, 0) or per[x["source_id"]] >= 2:
            continue
        got[k] += 1
        per[x["source_id"]] += 1
        out.append({k2: x.get(k2) for k2 in ("id", "source_id", "task_type", "type", "idx", "sentence", "original", "evidence", "reason", "tactic_raw", "fam", "cell",
                                             "p_correct", "p_a", "p_halluc", "orig_p", "source", "context")})
    rng.shuffle(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("PICK", a.out, len(out), dict(collections.Counter((r["cell"], r["fam"]) for r in out)), flush=True)


def text(x: dict) -> str:
    ctx = x["context"]   # 改过的整条回答（句子列表）
    k = int(x["idx"])
    sents = "\n".join(f"[S{i}] {t}" for i, t in enumerate(ctx, 1))
    return (f"<item id=\"{x['id']}\">\n<source>\n{x['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Target sentence: S{k}\nORIGINAL S{k}: {x['original']}\nEDITED S{k}: {x['sentence']}\n"
            f"Claimed type: {x['type']}\nClaimed evidence: {x['evidence']}\nClaimed reason: {x['reason']}\n</item>")


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    items = list(read_jsonl(a.items))
    batches = [{"id": f"b{k // a.batch}", "xs": items[k:k + a.batch]} for k in range(0, len(items), a.batch)]
    assert len(batches) <= a.max_calls, f"{len(batches)} calls > max_calls {a.max_calls}"
    llm = LLM(LLMConfig(model=a.model, provider="claude_cli", effort=a.effort, concurrency=a.concurrency))
    raw = Path(a.items).with_name(Path(a.items).stem + "_raw.jsonl")

    def run(b: dict) -> dict:
        t = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(text(x) for x in b["xs"])}])
        o = parse_json_obj(t)
        ys = o.get("items") if isinstance(o, dict) else None
        by = {str(y.get("id")): y for y in ys if isinstance(y, dict)} if isinstance(ys, list) else {}   # 按 id 对齐：Claude 漏回一条只丢这一条，不丢整批
        hit = [(x, by[x["id"]]) for x in b["xs"] if x["id"] in by]
        ok = bool(hit)
        return {"id": b["id"], "missing": [x["id"] for x in b["xs"] if x["id"] not in by],
                "rows": [{"hid": x["id"], **{k: x[k] for k in ("type", "fam", "cell", "p_correct", "p_halluc", "tactic_raw")}, "claude": y, "option": option(y)}
                         for x, y in hit] if ok else None, **({} if ok else {"raw_tail": str(t)[-400:], "got_ids": [str(y.get("id")) for y in ys] if isinstance(ys, list) else None})}

    map_resumable(batches, run, raw, concurrency=a.concurrency)
    rows = [r for b in kept(raw) for r in (b["rows"] or [])]
    out = Path(a.items).with_name(Path(a.items).stem + "_labels.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    c = collections.Counter
    print("SEV_LABEL", json.dumps({"n": len(rows), "batches_failed": sum(b["rows"] is None for b in kept(raw)), "option": dict(c(r["option"] for r in rows)),
                                   "severity": dict(c(r["claude"].get("severity") for r in rows)),
                                   "by_cell": {k: dict(c(r["option"] for r in rows if r["cell"] == k)) for k in sorted({r["cell"] for r in rows})},
                                   "by_fam": {k: dict(c(r["option"] for r in rows if r["fam"] == k)) for k in sorted({r["fam"] for r in rows})},
                                   "mechanism": dict(c(r["claude"].get("mechanism") for r in rows).most_common())}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "pick_v2", "label"])
    ap.add_argument("--harvest", default="outputs/rt/gentarget/harvest_h1.jsonl")
    ap.add_argument("--labeled", default="configs/data/rm3v2_labels.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", default="v2", choices=["v2", "v3", "v4"])
    ap.add_argument("--exclude", nargs="*", default=[], help="pick_v2：已经抽过的 items 文件，跳过其中的 id")
    ap.add_argument("--out", default="outputs/rt/sevlabel/items_p1.jsonl")
    ap.add_argument("--items", default="outputs/rt/sevlabel/items_p1.jsonl")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max_calls", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    {"pick": cmd_pick, "pick_v2": cmd_pick_v2, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
