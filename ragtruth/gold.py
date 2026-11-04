"""核心种子集和金标（2026-11-01，用户：选 100 条回答，完全按 Claude 的标准标注，作为核心种子集和核心标准）。规范见 docs/gold_standard.md。

  python -m ragtruth.gold pilot    # 第 1 步：选 10 条回答，每条独立标两遍 → outputs/rt/gold/pilot_labels.jsonl
  python -m ragtruth.gold review   # 比较两遍，挑约 20 句给用户看 → outputs/rt/gold/pilot_review.json
  python -m ragtruth.gold seed     # 种子集 100 条回答（含试标的 10 条）：先随机补到 40 条，再分批补 60 条缺的错误机制，每条两遍，不一致的第三遍裁决 → seed_final.jsonl
  python -m ragtruth.gold test     # 测试集 402 条回答全部标一遍，不挑分布 → test_final.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/gold")
POOL = "outputs/external/ragtruth/train_pool.jsonl"
TEST = "outputs/external/ragtruth/test_pp.jsonl"
VERSION = "v1"
MECH = ("wrong_attribution", "number_or_date", "contradicts_source", "merged_across_passages", "added_specific_fact", "vague_evaluation_or_addon",
        "certainty_or_scope_shift", "near_synonym_shift", "unsupported_inference", "self_description", "other", "none")

SYSTEM = """You label hallucinations in responses written from a source, for a gold-standard dataset. Compare every sentence of the response with the source yourself. Use earlier sentences only to resolve pronouns and references; judge each sentence's own claims against the source.

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

Reply with one JSON object only: {"sentences": [{"idx": 1, "is_error": true|false, "type": "...", "mechanism": "...", "severity": "...", "evidence": "...", "note": "..."}, ...]} with exactly one entry per sentence, in order."""


# 统计覆盖度用的粗类：细类之间两遍分歧主要在「添加细节 / 含糊附加语 / 无依据推断」（试标 12/18 一致），不用细类定覆盖目标
COARSE = {"wrong_attribution": "attribution", "number_or_date": "number", "contradicts_source": "contradiction", "merged_across_passages": "merge",
          "added_specific_fact": "adds", "vague_evaluation_or_addon": "adds", "unsupported_inference": "adds", "certainty_or_scope_shift": "scope", "near_synonym_shift": "scope"}
QUANT = re.compile(r"\b(all|every|always|never|most|many|will|must|only|definitely|likely|usually|everyone|none)\b", re.I)


def looks_like(r: dict, group: str) -> float:
    """标注前猜这条回答可能含哪类错误，只用来决定先标哪条（没有模型或标签信息，猜错只是这条对覆盖帮助小）。"""
    text = " ".join(r["sentences"])
    if group == "number":
        return len(re.findall(r"\d+", text))
    if group == "attribution":
        return len(set(re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(x.split(" ", 1)[-1] for x in r["sentences"]))))
    if group == "merge":
        return r["source"].lower().count("passage") if r["task_type"] == "QA" else 0
    if group == "scope":
        return len(QUANT.findall(text))
    return len(r["sentences"])   # adds、contradiction 没有好用的特征，句子多的回答机会多


def resolve(passes: list[list[dict]]) -> list[dict]:
    """逐句取多数；两遍一致直接用第一遍，不一致时第三遍裁决（明显 / 轻微、机制都用多数一方的第一份）。"""
    out = []
    for xs in zip(*passes):
        votes = [x["is_error"] for x in xs]
        win = sum(votes) * 2 > len(votes)
        x = next(x for x in xs if x["is_error"] == win)
        out.append({**x, "agreed": len(set(votes)) == 1})
    return out


def user_msg(r: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(r["sentences"], 1))
    return f"<source>\n{r['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>"


def valid(o, n: int) -> list[dict] | None:
    """结构不对就返回 None（重试），不猜。"""
    xs = o.get("sentences") if isinstance(o, dict) else None
    if not isinstance(xs, list) or len(xs) != n:
        return None
    for k, x in enumerate(xs, 1):
        if not isinstance(x, dict) or x.get("idx") != k or not isinstance(x.get("is_error"), bool) or x.get("mechanism") not in MECH or x.get("severity") not in ("clear", "minor", "none") or x.get("type") not in ("conflict", "baseless", "none"):
            return None
        if x["is_error"] == (x["severity"] == "none") or x["is_error"] == (x["type"] == "none") or x["is_error"] == (x["mechanism"] == "none"):
            return None
    return xs


def excluded_sources() -> set[str]:
    """测试集、校准集的原文不进种子集。"""
    pool = {r["id"]: r["source_id"] for r in read_jsonl(POOL)}
    out = {r["source_id"] for r in read_jsonl(TEST)}
    out |= {pool[r["group"]] for r in read_jsonl("outputs/rt/rm/rm1/calib.jsonl") if r["group"] in pool}
    return out


def pick_pilot(rows: list[dict], excl: set[str], n: int, seed: int) -> list[dict]:
    """QA、Summary 各一半，生成模型轮流取，每个原文只留一条回答。"""
    rows = sorted((r for r in rows if r["task_type"] in ("QA", "Summary") and r["source_id"] not in excl and r.get("sentences")), key=lambda r: r["id"])
    random.Random(f"{seed}|gold").shuffle(rows)
    models = sorted({r["model"] for r in rows})
    out, used = [], set()
    for k in range(n):
        task, model = ("QA", "Summary")[k % 2], models[k % len(models)]
        r = next(r for r in rows if r["task_type"] == task and r["model"] == model and r["source_id"] not in used)
        used.add(r["source_id"])
        out.append(r)
    return out


def cmd_pilot(args) -> None:
    D.mkdir(parents=True, exist_ok=True)
    picks = pick_pilot(list(read_jsonl(POOL)), excluded_sources(), args.n, args.seed)
    with open(D / "pilot_ids.jsonl", "w", encoding="utf-8") as f:
        for r in picks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("picked", collections.Counter((r["task_type"], r["model"]) for r in picks), "sentences", sum(len(r["sentences"]) for r in picks), flush=True)
    assert 2 * len(picks) <= 24, "用户同意的第 1 步约 20 次调用"
    items = [{"id": f"{r['id']}#p{p}", "r": r} for r in picks for p in (1, 2)]
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(it: dict) -> dict | None:
        r = it["r"]
        for _ in range(2):
            out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}])
            xs = valid(parse_json_obj(out), len(r["sentences"]))
            if xs is not None:
                return {"id": it["id"], "resp_id": r["id"], "pass": int(it["id"][-1]), "version": VERSION, "sentences": xs}
        print("invalid output", it["id"], out[:200], flush=True)
        return None

    map_resumable(items, run, D / "pilot_labels.jsonl", concurrency=args.concurrency)
    cmd_review(args)


def label(llm, items: list[dict], path: Path, concurrency: int) -> None:
    """items：{id, r}。结果可续跑；输出不合格的这次不写，下次重跑时再试（金标不能缺句）。"""
    if path.exists():
        rows = [x for x in read_jsonl(path) if not x.get("_dropped")]
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    def run(it: dict) -> dict | None:
        r = it["r"]
        for _ in range(3):
            out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}])
            xs = valid(parse_json_obj(out), len(r["sentences"]))
            if xs is not None:
                return {"id": it["id"], "resp_id": r["id"], "pass": int(it["id"].rsplit("#p", 1)[1]), "version": VERSION, "sentences": xs}
        print("invalid output", it["id"], out[:200], flush=True)
        return None

    map_resumable(items, run, path, concurrency=concurrency)


def write_final(picks: dict[str, dict], passes: dict, path: Path) -> list[dict]:
    rows = []
    for rid in sorted(passes):
        r = picks[rid]
        for x in resolve([passes[rid][p] for p in sorted(passes[rid])]):
            rows.append({"id": f"{rid}-S{x['idx']}", "resp_id": rid, "idx": x["idx"], "task_type": r["task_type"], "model": r["model"], "source_id": r["source_id"], "sentence": r["sentences"][x["idx"] - 1],
                         "old_label": bool(r["bad"].get(str(x["idx"])) or r["bad"].get(x["idx"])), "version": VERSION, **{k: x[k] for k in ("is_error", "type", "mechanism", "severity", "evidence", "note", "agreed")}})
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return rows


def seed_passes(picks: dict[str, dict], llm, args) -> dict:
    """每条两遍（试标的 10 条已有两遍），两遍不一致的回答加第三遍。"""
    path = D / "seed_labels.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept(D / "pilot_labels.jsonl")), encoding="utf-8")   # 同一版规范，试标算种子
    label(llm, [{"id": f"{rid}#p{p}", "r": r} for rid, r in picks.items() for p in (1, 2)], path, args.concurrency)
    by = collections.defaultdict(dict)
    for x in kept(path):
        by[x["resp_id"]][x["pass"]] = x["sentences"]
    third = [rid for rid, ps in by.items() if rid in picks and 1 in ps and 2 in ps and any(a["is_error"] != b["is_error"] for a, b in zip(ps[1], ps[2]))]
    label(llm, [{"id": f"{rid}#p3", "r": picks[rid]} for rid in sorted(third)], path, args.concurrency)
    by = collections.defaultdict(dict)
    for x in kept(path):
        by[x["resp_id"]][x["pass"]] = x["sentences"]
    return {rid: ps for rid, ps in by.items() if rid in picks}


def coverage(rows: list[dict]) -> dict:
    c = collections.Counter(COARSE.get(x["mechanism"], "other") for x in rows if x["is_error"])
    return {g: c[g] for g in ("attribution", "number", "contradiction", "merge", "adds", "scope", "other")}


def cmd_seed(args) -> None:
    excl = excluded_sources()
    pool = sorted((r for r in read_jsonl(POOL) if r["task_type"] in ("QA", "Summary") and r["source_id"] not in excl and r.get("sentences")), key=lambda r: r["id"])
    ids = D / "seed_ids.jsonl"
    picks = {r["id"]: r for r in read_jsonl(ids if ids.exists() else D / "pilot_ids.jsonl")}
    used = {r["source_id"] for r in picks.values()}
    random.Random(f"{args.seed}|seed").shuffle(pool)
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def save() -> None:
        ids.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picks.values()), encoding="utf-8")

    def take(pred_key, n: int) -> None:
        """任务、生成模型轮流；pred_key 给分，分高的先选。"""
        models = sorted({r["model"] for r in pool})
        for _ in range(n):
            cnt = collections.Counter((r["task_type"], r["model"]) for r in picks.values())
            task, model = min(((t, m) for t in ("QA", "Summary") for m in models), key=lambda x: (cnt[x], x))
            cand = [r for r in pool if r["task_type"] == task and r["model"] == model and r["source_id"] not in used]
            r = max(cand, key=pred_key) if pred_key else cand[0]
            picks[r["id"]] = r
            used.add(r["source_id"])

    if len(picks) < args.n_random:
        take(None, args.n_random - len(picks))
        save()
    assert len(picks) <= 100, "用户同意的种子集是 100 条回答"
    rows = write_final(picks, seed_passes(picks, llm, args), D / "seed_final.jsonl")
    print("random", len(picks), "coverage", coverage(rows), flush=True)
    while len(picks) < args.n_seed:
        cov = coverage(rows)
        short = min((g for g in cov if g not in ("adds", "other")), key=lambda g: (cov[g], g))
        take(lambda r: looks_like(r, short), min(args.batch, args.n_seed - len(picks)))
        save()
        rows = write_final(picks, seed_passes(picks, llm, args), D / "seed_final.jsonl")
        print("targeted for", short, "now", len(picks), "coverage", coverage(rows), flush=True)
    err = [x for x in rows if x["is_error"]]
    rep = {"responses": len(picks), "responses_with_error": len({x["resp_id"] for x in err}), "sentences": len(rows), "errors": len(err), "clear": sum(x["severity"] == "clear" for x in err), "conflict": sum(x["type"] == "conflict" for x in err),
           "old_label_positive": sum(x["old_label"] for x in rows), "agreed_sentences": sum(x["agreed"] for x in rows), "coverage": coverage(rows),
           "tasks": collections.Counter(r["task_type"] for r in picks.values()), "models": collections.Counter(r["model"] for r in picks.values())}
    (D / "seed_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SEED_DONE", json.dumps(rep, ensure_ascii=False), flush=True)


def cmd_test(args) -> None:
    picks = {r["id"]: r for r in read_jsonl(TEST)}
    assert len(picks) <= 402
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))
    label(llm, [{"id": f"{rid}#p1", "r": picks[rid]} for rid in sorted(picks)], D / "test_labels.jsonl", args.concurrency)
    by = {x["resp_id"]: {1: x["sentences"]} for x in kept(D / "test_labels.jsonl")}
    rows = write_final(picks, by, D / "test_final.jsonl")
    err = [x for x in rows if x["is_error"]]
    rep = {"responses": len(by), "missing": sorted(set(picks) - set(by)), "responses_with_error": len({x["resp_id"] for x in err}), "sentences": len(rows), "errors": len(err), "clear": sum(x["severity"] == "clear" for x in err), "conflict": sum(x["type"] == "conflict" for x in err),
           "old_pp_positive": sum(x["old_label"] for x in rows), "same_as_pp": sum(x["old_label"] == x["is_error"] for x in rows), "coverage": coverage(rows)}
    (D / "test_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("TEST_DONE", json.dumps(rep, ensure_ascii=False), flush=True)


def cmd_review(args) -> None:
    picks = {r["id"]: r for r in read_jsonl(D / "pilot_ids.jsonl")}
    by = collections.defaultdict(dict)
    for x in kept(D / "pilot_labels.jsonl"):
        by[x["resp_id"]][x["pass"]] = x["sentences"]
    rows, agree = [], collections.Counter()
    for rid, passes in sorted(by.items()):
        if len(passes) < 2:
            print("missing pass", rid)
            continue
        r = picks[rid]
        for a, b in zip(passes[1], passes[2]):
            i = a["idx"]
            agree["sentences"] += 1
            agree["is_error_same"] += a["is_error"] == b["is_error"]
            agree["severity_same"] += a["severity"] == b["severity"]
            agree["errors_p1"] += a["is_error"]
            agree["clear_p1"] += a["severity"] == "clear"
            old = bool(r["bad"].get(str(i)) or r["bad"].get(i))
            rows.append({"resp_id": rid, "idx": i, "task": r["task_type"], "model": r["model"], "sentence": r["sentences"][i - 1], "old_label": old, "p1": a, "p2": b,
                         "disagree": a["is_error"] != b["is_error"], "sev_differs": a["severity"] != b["severity"]})
    rank = lambda x: (not x["disagree"], not x["sev_differs"], not (x["p1"]["is_error"] != x["old_label"]), x["p1"]["severity"] != "minor", x["resp_id"], x["idx"])
    review = sorted(rows, key=rank)[:args.review_n]
    mech = collections.Counter(x["p1"]["mechanism"] for x in rows if x["p1"]["is_error"])
    summary = {"version": VERSION, **agree, "old_positive": sum(x["old_label"] for x in rows), "error_and_old_differ": sum(x["p1"]["is_error"] != x["old_label"] for x in rows), "mechanism": dict(mech.most_common())}
    srcs = {x["resp_id"]: picks[x["resp_id"]]["source"] for x in review}
    (D / "pilot_review.json").write_text(json.dumps({"summary": summary, "items": review, "sources": srcs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "review", "seed", "test"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--review_n", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=40)
    ap.add_argument("--n_seed", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    {"pilot": cmd_pilot, "review": cmd_review, "seed": cmd_seed, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
