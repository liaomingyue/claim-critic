"""回答满意度模型的标注（只是护栏，不让模型为了降幻觉去钻空子，比如故意打断、抄袭、字数很少、答非所问；不需要很准）。

Claude 看同一问题的两条回答，一次给出相对判断（A / B / 差不多）和两条各自的绝对分数、毛病，不管幻觉，只管有没有切中需求、解决问题。
样本来自回答 GRPO 每步存盘的回答（只有训练原文，不碰 dev 和评测集），必须包含走偏的版本，否则模型没见过走偏的样子：
  cross：早期版本 vs 后期版本；late：后期版本内部；contrast：后期里长度或照抄差异大的。
一部分对子 A、B 对调重判一次，看判断是否一致。Claude 调用摊开跑（--gap）。

  python -m ragtruth.sat_data pick --samples outputs/rt/ansgrpo/run/samples   # 试标：90 对 + 10 次重判 = 100 次调用
  python -m ragtruth.sat_data label
  python -m ragtruth.sat_data report
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import statistics
import time
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/ansgrpo/sat")
FLAWS = ("truncated", "copied_fragments", "too_short", "off_topic", "misses_key_info", "padding", "none")
SYSTEM = """You compare two answers to the same task the way the person who gave the task would. You receive the task (instructions plus source material) and answers A and B.

Judge only whether an answer is well formed, addresses what was asked and actually solves the task. Do NOT judge whether its claims are supported by the source; hallucination is checked elsewhere. An unsupported detail must not lower the score unless it makes the answer useless.

Flaws to look for:
- truncated: stops abruptly or leaves the answer unfinished
- copied_fragments: mostly stitched source text instead of an answer or summary written for the task
- too_short: so short that the person does not get what they need
- off_topic: does not address what was asked
- misses_key_info: leaves out information from the source that the task needs
- padding: filler or repetition that adds nothing
- none: no flaw

If the source really does not contain the answer, saying so is the useful answer. Use "tie" when neither is clearly better; do not force a choice.

Reply with one JSON object only:
{"winner": "A" | "B" | "tie", "score_a": 1-5, "score_b": 1-5, "flaws_a": [...], "flaws_b": [...], "note": "one short sentence"}

score: 5 fully does the job; 4 does the job with small gaps; 3 partly useful; 2 mostly misses; 1 useless."""


def valid(o) -> dict | None:
    if not isinstance(o, dict) or o.get("winner") not in ("A", "B", "tie"):
        return None
    if o.get("score_a") not in (1, 2, 3, 4, 5) or o.get("score_b") not in (1, 2, 3, 4, 5):
        return None
    for k in ("flaws_a", "flaws_b"):
        if not isinstance(o.get(k), list) or any(f not in FLAWS for f in o[k]):
            return None
    return {k: o.get(k) for k in ("winner", "score_a", "score_b", "flaws_a", "flaws_b", "note")}


def load_samples(d: Path, train_ids: set[str]) -> dict[str, list[dict]]:
    """{原文: [回答]}。不要截断以外的硬规则样本也留着（拒答、太短、照抄正是要教给模型的毛病）。"""
    by = collections.defaultdict(list)
    for f in sorted(d.glob("step*.jsonl.gz"), key=lambda p: int(p.name[4:].split(".")[0])):
        step = int(f.name[4:].split(".")[0])
        for l in gzip.open(f, "rt", encoding="utf-8"):
            s = json.loads(l)
            if s["prompt_id"] in train_ids and s["text"].strip():
                by[s["prompt_id"]].append({"step": step, "text": s["text"].strip(), "chars": s["chars"], "copy": s["copy"], "why": s["why"], "p_mean": s.get("p_mean")})
    return by


def pick_pairs(by: dict[str, list[dict]], n: dict[str, int], early_max: int, late_min: int, seed: int) -> list[dict]:
    """每个原文最多用一次；QA、Summary 轮流取。"""
    rng = random.Random(f"{seed}|sat")
    ids = sorted(by)
    rng.shuffle(ids)
    used, out = set(), []

    def take(bucket, make, k):
        got = 0
        for i in ids:
            if got >= k:
                break
            if i in used:
                continue
            p = make(by[i])
            if p:
                used.add(i)
                out.append({"prompt_id": i, "bucket": bucket, "a": p[0], "b": p[1]})
                got += 1

    def cross(xs):
        e, l = [x for x in xs if x["step"] <= early_max], [x for x in xs if x["step"] >= late_min]
        return (rng.choice(e), rng.choice(l)) if e and l else None

    def late(xs):
        l = [x for x in xs if x["step"] >= late_min]
        if len(l) < 2:
            return None
        a, b = rng.sample(l, 2)
        return (a, b) if a["text"] != b["text"] else None

    def contrast(xs):
        l = [x for x in xs if x["step"] >= late_min]
        if len(l) < 2:
            return None
        s = sorted(l, key=lambda x: x["chars"])
        lo, hi = s[0], s[-1]
        if lo["chars"] * 1.5 <= hi["chars"]:
            return lo, hi
        c = sorted(l, key=lambda x: x["copy"])
        return (c[0], c[-1]) if c[-1]["copy"] - c[0]["copy"] >= 0.3 else None

    for bucket, make in (("contrast", contrast), ("cross", cross), ("late", late)):
        take(bucket, make, n.get(bucket, 0))
    return out


def to_items(pairs: list[dict], repeats: int, seed: int) -> list[dict]:
    """随机定谁是 A，末尾再加 repeats 次 A、B 对调的重判。"""
    rng = random.Random(f"{seed}|satorder")
    items = []
    for n, p in enumerate(pairs):
        flip = rng.random() < .5
        a, b = (p["b"], p["a"]) if flip else (p["a"], p["b"])
        items.append({"id": f"p{n:04d}", "pair": n, "prompt_id": p["prompt_id"], "bucket": p["bucket"], "repeat": False, "A": a, "B": b})
    for x in rng.sample(items, min(repeats, len(items))):
        items.append({**x, "id": x["id"] + "r", "repeat": True, "A": x["B"], "B": x["A"]})
    rng.shuffle(items)
    return items


def cmd_pick(args) -> None:
    train = {r["id"]: r for r in read_jsonl("outputs/rt/ansgrpo/train.jsonl")}
    used = {x["prompt_id"] for f in D.glob("items*.jsonl") for x in read_jsonl(f)}   # 各批原文不重复
    by = load_samples(Path(args.samples), set(train) - used)
    pairs = pick_pairs(by, {"contrast": args.contrast, "cross": args.cross, "late": args.late}, args.early_max, args.late_min, args.seed)
    items = to_items(pairs, args.repeats, args.seed)
    assert len(items) <= args.max_calls, "超出 Claude 调用预算"
    D.mkdir(parents=True, exist_ok=True)
    with open(D / f"items{args.tag}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("PICKED", len(items), dict(collections.Counter(x["bucket"] for x in items)), dict(collections.Counter(train[x["prompt_id"]]["task_type"] for x in items)), flush=True)


def cmd_label(args) -> None:
    src = {r["id"]: r for r in read_jsonl("outputs/rt/ansgrpo/train.jsonl")}
    items = list(read_jsonl(D / f"items{args.tag}.jsonl"))
    assert len(items) <= args.max_calls, "超出 Claude 调用预算"
    llm = LLM(LLMConfig(model=args.claude, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(it: dict) -> dict | None:
        time.sleep(args.gap)
        out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"[task]\n{src[it['prompt_id']]['source']}\n\n[answer A]\n{it['A']['text']}\n\n[answer B]\n{it['B']['text']}"}])
        o = valid(parse_json_obj(out))
        if o is None:
            print("invalid output", it["id"], out[:200], flush=True)
            return None
        return {**it, "task_type": src[it["prompt_id"]]["task_type"], **o}

    map_resumable(items, run, D / f"labels{args.tag}.jsonl", concurrency=args.concurrency)
    cmd_report(args)


def repeat_agreement(rows: list[dict]) -> dict:
    """对调后的判断要反过来才算一致。"""
    first = {x["pair"]: x for x in rows if not x["repeat"]}
    flip = {"A": "B", "B": "A", "tie": "tie"}
    same = [flip[x["winner"]] == first[x["pair"]]["winner"] for x in rows if x["repeat"] and x["pair"] in first]
    return {"n": len(same), "agree": round(sum(same) / len(same), 3) if same else None}


def summarize(rows: list[dict]) -> dict:
    base = [x for x in rows if not x["repeat"]]
    n = max(1, len(base))
    decisive = [x for x in base if x["winner"] != "tie"]
    win = lambda x: x["A"] if x["winner"] == "A" else x["B"]
    lose = lambda x: x["B"] if x["winner"] == "A" else x["A"]
    scores = [x["score_a"] for x in base] + [x["score_b"] for x in base]
    flaws = collections.Counter(f for x in base for k in ("flaws_a", "flaws_b") for f in x[k])
    rep = {"pairs": len(base), "calls": len(rows), "tie": round(1 - len(decisive) / n, 3), "repeat": repeat_agreement(rows),
           "scores": dict(sorted(collections.Counter(scores).items())), "low_scores(<=3)": sum(s <= 3 for s in scores), "flaws": dict(flaws.most_common()),
           "longer_wins": round(statistics.mean(win(x)["chars"] > lose(x)["chars"] for x in decisive), 3) if decisive else None, "by_bucket": {}, "by_task": dict(collections.Counter(x["task_type"] for x in base))}
    for b in sorted({x["bucket"] for x in base}):
        xs = [x for x in base if x["bucket"] == b]
        d = [x for x in xs if x["winner"] != "tie"]
        rep["by_bucket"][b] = {"n": len(xs), "tie": round(1 - len(d) / len(xs), 3)}
        if b == "cross" and d:   # 后期版本赢的比例：明显低于一半就是走偏
            rep["by_bucket"][b]["late_wins"] = round(statistics.mean(win(x)["step"] > lose(x)["step"] for x in d), 3)
    return rep


def cmd_report(args) -> None:
    rep = summarize(kept(D / f"labels{args.tag}.jsonl"))
    (D / f"report{args.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "label", "report"])
    ap.add_argument("--samples", default="outputs/rt/ansgrpo/run/samples")
    ap.add_argument("--tag", default="", help="第二批起的文件名后缀，如 _b2")
    ap.add_argument("--contrast", type=int, default=35)
    ap.add_argument("--cross", type=int, default=35)
    ap.add_argument("--late", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--early_max", type=int, default=20)
    ap.add_argument("--late_min", type=int, default=150)
    ap.add_argument("--max_calls", type=int, default=100, help="Claude 调用上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gap", type=float, default=20.0, help="每次 Claude 调用前等几秒")
    args = ap.parse_args()
    {"pick": cmd_pick, "label": cmd_label, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
