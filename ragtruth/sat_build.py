"""满意度模型的训练数据（2026-11-02，用户：只是护栏，防止为了降幻觉故意打断、抄袭、字数很少、答非所问；相对、绝对都做，两张卡一起训）。

  sat_pair：任务 + 回答 A + 回答 B → A / B（哪条更让用户满意）。差不多的对子不要，每对 A、B 对调各一行，避免偏向某个位置。
  sat_abs：任务 + 一条回答 → A 好（Claude ≥ 4 分）/ B 不好（≤ 3 分）。
Claude 标注（ragtruth.sat_data）里这些毛病很少，所以程序再造四类反例，每类 --n_synth 条（原文和标注对子不重叠，不花 Claude 调用）：
  truncated 砍到一半、停在句子中间；too_short 只留第一句；copied 拼接原文句子；off_topic 换成同任务别的问题的回答。
按原文分 train / calib / test（用不同原文），数据写到 outputs/rt/rm/sat_{pair,abs}，和其他 RM 同一套训练、校准流程（ragtruth.rm_eval）。

  python -m ragtruth.sat_build
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.convert import split_sentences
from ragtruth.sat_data import D as SAT, load_samples

OUT = Path("outputs/rt/rm")
MAX_TOKENS = 4096
JUDGE = """Judge only whether an answer is well formed, addresses what was asked and actually solves the task. Do NOT judge whether its claims are supported by the source; that is checked elsewhere.

Bad signs: stops abruptly or is unfinished, mostly stitched source text instead of an answer written for the task, so short that the person does not get what they need, does not address what was asked, leaves out information the task needs, filler or repetition. If the source really does not contain the answer, saying so is the useful answer."""
SAT_PAIR_SYSTEM = f"""You compare two answers to the same task the way the person who gave the task would.

{JUDGE}

Options:
A. answer A is better.
B. answer B is better.

Reply with one letter."""
SAT_ABS_SYSTEM = f"""You judge one answer to a task the way the person who gave the task would.

{JUDGE}

Options:
A. good: the answer does the job, at most with small gaps.
B. not good: the answer is only partly useful or worse.

Reply with one letter."""


def pair_user(source: str, a: str, b: str) -> str:
    return f"[task]\n{source.strip()}\n\n[answer A]\n{a.strip()}\n\n[answer B]\n{b.strip()}"


def abs_user(source: str, answer: str) -> str:
    return f"[task]\n{source.strip()}\n\n[answer]\n{answer.strip()}"


def sents(text: str) -> list[str]:
    return [text[a:b] for a, b in split_sentences(text)]


def degrade(kind: str, answer: str, source: str, other: str, rng: random.Random) -> str | None:
    """把一条正常回答改成某种毛病。条件不够时返回 None。"""
    ss = sents(answer)
    if kind == "truncated":
        if len(answer) < 80:
            return None
        cut = rng.randint(len(answer) // 3, len(answer) * 2 // 3)
        out = answer[:cut].rstrip()
        out = out[:out.rfind(" ")] if " " in out else out   # 停在词边界、句子中间
        return out.rstrip(" .,;:!?") or None
    if kind == "too_short":
        return ss[0] if len(ss) >= 3 else None
    if kind == "copied":
        body = source.split("\n", 1)[1] if "\n" in source else source
        cand = [s for s in sents(body) if len(s) >= 30 and "passage" not in s.lower() and "output:" not in s.lower()]
        if len(cand) < 3:
            return None
        k = rng.randint(2, min(5, len(cand)))
        start = rng.randint(0, len(cand) - k)
        return " ".join(cand[start:start + k])
    if kind == "off_topic":
        return other if other != answer else None
    raise ValueError(kind)


def split_of(prompt_id: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}|sat|{prompt_id}".encode()).hexdigest(), 16) % 100
    return "calib" if h < 10 else "test" if h < 25 else "train"


def pair_rows(pid: str, source: str, x: str, y: str, winner: str, origin: str, extra: dict) -> list[dict]:
    """winner 是 x 或 y。A、B 对调各一行。"""
    out = []
    for k, (a, b) in enumerate(((x, y), (y, x))):
        lab = "A" if (winner == "x") == (k == 0) else "B"
        out.append({"id": f"{origin}-{pid}-{extra['n']}-{k}", "prompt": [{"role": "system", "content": SAT_PAIR_SYSTEM}, {"role": "user", "content": pair_user(source, a, b)}], "completion": lab, "label": lab, "group": pid, "origin": origin, **{k2: v for k2, v in extra.items() if k2 != "n"}})
    return out


def abs_row(pid: str, source: str, ans: str, good: bool, origin: str, extra: dict) -> dict:
    lab = "A" if good else "B"
    return {"id": f"{origin}-{pid}-{extra['n']}", "prompt": [{"role": "system", "content": SAT_ABS_SYSTEM}, {"role": "user", "content": abs_user(source, ans)}], "completion": lab, "label": lab, "group": pid, "origin": origin, **{k2: v for k2, v in extra.items() if k2 != "n"}}


def build_labeled(labels: list[dict], src: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    pair, abso = [], []
    for n, x in enumerate(labels):
        pid, s = x["prompt_id"], src[x["prompt_id"]]["source"]
        if x["winner"] != "tie":
            pair += pair_rows(pid, s, x["A"]["text"], x["B"]["text"], "x" if x["winner"] == "A" else "y", "claude", {"n": n, "bucket": x["bucket"], "task_type": x["task_type"]})
        for side, score in (("A", x["score_a"]), ("B", x["score_b"])):
            abso.append(abs_row(pid, s, x[side]["text"], score >= 4, "claude", {"n": f"{n}{side}", "bucket": x["bucket"], "task_type": x["task_type"], "score": score}))
    return pair, abso


def build_synth(by: dict[str, list[dict]], src: dict[str, dict], n_each: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(f"{seed}|satsynth")
    pool = collections.defaultdict(list)
    ok = {}
    for pid in sorted(by):
        good = sorted((a for a in by[pid] if a["why"] == "ok"), key=lambda a: a["step"])
        if good:
            ok[pid] = good
            pool[src[pid]["task_type"]].append(pid)
    ids = sorted(ok)
    rng.shuffle(ids)
    pair, abso, counts = [], [], collections.Counter()
    i = 0
    for kind in ("truncated", "too_short", "copied", "off_topic"):
        got = 0
        while got < n_each and i < 4 * len(ids) * n_each:
            pid = ids[i % len(ids)]
            i += 1
            a = rng.choice(ok[pid])["text"]
            t = src[pid]["task_type"]
            other_id = rng.choice(pool[t])
            other = rng.choice(ok[other_id])["text"] if other_id != pid else a
            bad = degrade(kind, a, src[pid]["source"], other, rng)
            if not bad:
                continue
            n = f"{kind}{got}"
            win = rng.random() < .5   # 正常回答放哪边随机，然后两种顺序都放
            pair += pair_rows(pid, src[pid]["source"], a, bad, "x", f"synth_{kind}", {"n": n, "task_type": t}) if win else pair_rows(pid, src[pid]["source"], bad, a, "y", f"synth_{kind}", {"n": n, "task_type": t})
            abso.append(abs_row(pid, src[pid]["source"], bad, False, f"synth_{kind}", {"n": n, "task_type": t}))
            got += 1
        counts[kind] = got
    return pair, abso, dict(counts)


def write_split(name: str, rows: list[dict], seed: int, too_long=None) -> dict:
    d = OUT / name
    (d).mkdir(parents=True, exist_ok=True)
    dropped = 0
    parts = {"train": [], "calib": [], "test": []}
    for r in sorted(rows, key=lambda r: r["id"]):
        if too_long and too_long(r["prompt"], MAX_TOKENS):
            dropped += 1
            continue
        parts[split_of(r["group"], seed)].append(r)
    meta = {"max_tokens": MAX_TOKENS, "epochs": 2, "too_long": dropped}
    for sp, xs in parts.items():
        assert len({x["id"] for x in xs}) == len(xs), sp
        with open(d / f"{sp}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        meta[sp] = {"n": len(xs), "label": dict(collections.Counter(x["label"] for x in xs)), "origin": dict(collections.Counter(x["origin"] for x in xs))}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="outputs/rt/ansgrpo/run/samples")
    ap.add_argument("--n_synth", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from ragtruth.rm_data import too_long
    src = {r["id"]: r for r in read_jsonl("outputs/rt/ansgrpo/train.jsonl")}
    labels = [x for f in ("labels.jsonl", "labels_b2.jsonl") for x in read_jsonl(SAT / f) if not x.get("_dropped") and not x.get("repeat")]
    used = {x["prompt_id"] for x in labels}
    p1, a1 = build_labeled(labels, src)
    by = load_samples(Path(args.samples), set(src) - used)
    p2, a2, counts = build_synth(by, src, args.n_synth, args.seed)
    res = {"labels": len(labels), "tie": sum(x["winner"] == "tie" for x in labels), "synth": counts, "sat_pair": write_split("sat_pair", p1 + p2, args.seed, too_long), "sat_abs": write_split("sat_abs", a1 + a2, args.seed, too_long)}
    (SAT / "build.meta.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SAT_BUILD", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
