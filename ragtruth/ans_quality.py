"""回答质量抽查（2026-11-02，用户：幻觉降了但没答对问题就没意义；抽查约 30 次 Claude 调用）。

critic 只看忠实度，看不出答非所问。这里让 Claude 判回答本身有没有用：dev 上抽 15 个原文（Summary 8、QA 7），第 0 步和检查点各取第一条，
打乱后逐条判断，Claude 不知道回答来自哪个模型。样本来自 train.ans_grpo 的 quick_samples。

  python -m ragtruth.ans_quality label --step 100 [--samples_dir outputs/rt/ansgrpo/run]
  python -m ragtruth.ans_quality report --step 100

15 对只能抓大幅变化，不能证明质量没变。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/ansgrpo/quality")
SYSTEM = """You judge how useful an answer is to the person who gave the task. You receive the exact task the assistant received (instructions plus source material) and its answer.

Judge usefulness only. Whether an individual claim is supported by the source is checked elsewhere, so do not lower the score just for an unsupported detail unless it makes the answer wrong on the main point.

For a question: does it answer what was asked, is the main point right according to the source, does it leave out information from the source that the question needs. If the source really does not contain the answer, saying so is the useful answer.
For a summary: does it cover the main points of the source, does it stay on what matters, or is it mostly copied fragments or side details.

Reply with one JSON object only:
{"answers_task": "yes" | "partly" | "no", "misses_key_info": true | false, "mostly_copied_fragments": true | false, "score": 1-5, "note": "one short sentence"}

score: 5 fully does the job; 4 does the job with small gaps; 3 partly useful; 2 mostly misses; 1 useless."""


def valid(o) -> dict | None:
    if not isinstance(o, dict) or o.get("answers_task") not in ("yes", "partly", "no") or o.get("score") not in (1, 2, 3, 4, 5):
        return None
    if not isinstance(o.get("misses_key_info"), bool) or not isinstance(o.get("mostly_copied_fragments"), bool):
        return None
    return {k: o.get(k) for k in ("answers_task", "misses_key_info", "mostly_copied_fragments", "score", "note")}


def pick_pairs(dev: list[dict], base: dict, ckpt: dict, n_qa: int, n_summary: int, seed: int) -> list[dict]:
    """→ 打乱的条目。条目 id 不含模型信息，arm 只存在结果里，不进提示词。"""
    rng = random.Random(f"{seed}|ansquality")
    out = []
    for t, n in (("QA", n_qa), ("Summary", n_summary)):
        ids = sorted(r["id"] for r in dev if r["task_type"] == t and r["id"] in base and r["id"] in ckpt)
        rng.shuffle(ids)
        for i in ids[:n]:
            for arm, rows in (("base", base), ("ckpt", ckpt)):
                out.append({"source_id": i, "task_type": t, "arm": arm, "answer": rows[i]["outputs"][0]})
    rng.shuffle(out)
    for k, x in enumerate(out):
        x["id"] = f"q{k:03d}"
    return out


def summarize(rows: list[dict]) -> dict:
    by = {}
    for x in rows:
        by.setdefault(x["source_id"], {})[x["arm"]] = x
    pairs = [v for v in by.values() if len(v) == 2]
    arm = lambda a, f: round(statistics.mean(f(p[a]) for p in pairs), 3) if pairs else None
    rep = {"pairs": len(pairs), "calls": len(rows)}
    for t in ("all", "QA", "Summary"):
        ps = [p for p in pairs if t == "all" or p["base"]["task_type"] == t]
        if not ps:
            continue
        m = lambda a, f: round(statistics.mean(f(p[a]) for p in ps), 3)
        d = [p["ckpt"]["score"] - p["base"]["score"] for p in ps]
        rep[t] = {"n": len(ps), "score": {a: m(a, lambda x: x["score"]) for a in ("base", "ckpt")}, "answers_yes": {a: m(a, lambda x: x["answers_task"] == "yes") for a in ("base", "ckpt")},
                  "misses_key": {a: m(a, lambda x: x["misses_key_info"]) for a in ("base", "ckpt")}, "copied": {a: m(a, lambda x: x["mostly_copied_fragments"]) for a in ("base", "ckpt")},
                  "worse": sum(v < 0 for v in d), "better": sum(v > 0 for v in d), "same": sum(v == 0 for v in d), "mean_delta": round(statistics.mean(d), 3)}
    return rep


def cmd_label(args) -> None:
    load = lambda s: {r["id"]: r for r in read_jsonl(Path(args.samples_dir) / f"quick_samples_a_s{s}.jsonl")}
    items = pick_pairs(list(read_jsonl("outputs/rt/ansgrpo/dev.jsonl")), load(0), load(args.step), args.n_qa, args.n_summary, args.seed)
    assert len(items) <= 30, "用户同意的是约 30 次调用"
    src = {r["id"]: r["source"] for r in read_jsonl("outputs/rt/ansgrpo/dev.jsonl")}
    D.mkdir(parents=True, exist_ok=True)
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(it: dict) -> dict | None:
        out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"[task]\n{src[it['source_id']]}\n\n[answer]\n{it['answer']}"}])
        o = valid(parse_json_obj(out))
        if o is None:
            print("invalid output", it["id"], out[:200], flush=True)
            return None
        return {**it, **o}

    map_resumable(items, run, D / f"step{args.step}.jsonl", concurrency=args.concurrency)
    cmd_report(args)


def cmd_report(args) -> None:
    rep = summarize(kept(D / f"step{args.step}.jsonl"))
    (D / f"step{args.step}.report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "report"])
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--samples_dir", default="outputs/rt/ansgrpo/run")
    ap.add_argument("--n_qa", type=int, default=7)
    ap.add_argument("--n_summary", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    {"label": cmd_label, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
