"""拒答 RM 的数据：QA 原文到底能不能回答。拒答只和题目有关、和回答无关，每个原文 Claude 判一次。

真正答不了的题很少（训练池 708 个 QA 原文里，6 个 RAGTruth 模型一半以上拒答的只有 5.4%），所以按 RAGTruth 模型的拒答数分层多抽：
  ≥ 3 个模型拒答的全要，2 个的全要，其余随机补到 --n。不碰评测集、校准集、金标种子、回答 GRPO 的 dev 原文。
Claude 判三档：yes（原文有答案）/ partial（有相关信息，能部分回答）/ no（没有相关信息）。和 MiniCPM 的系统提示一致：只有 no 才该拒答。
RM 选项：A 应该回答（yes / partial），B 可以拒答（no）。数据按原文分 train / calib / test，写到 outputs/rt/rm/refuse。

  python -m ragtruth.refuse_data pick
  python -m ragtruth.refuse_data label [--gap 5]          # 一题一次（最早标的 11 题）
  python -m ragtruth.refuse_data label_batch --per 10      # 任务不难，一次给 Claude 多题；已标过的跳过
  python -m ragtruth.refuse_data pick_rest && python -m ragtruth.refuse_data label_batch --per 20   # 剩下的原文一次 20 题补标
  python -m ragtruth.refuse_data build
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.gold import POOL, excluded_sources
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/refuse")
OUT = Path("outputs/rt/rm/refuse")
SYSTEM = """You decide whether a question can be answered from the passages given with it. You receive the exact task an assistant received: the question, the passages, and instructions.

Judge only the passages, not your own knowledge.
- yes: the passages contain the answer to what is asked.
- partial: the passages contain information relevant to the question that lets an assistant give a useful partial answer.
- no: the passages contain nothing that answers or meaningfully addresses the question; the right reply is to say it cannot be answered.

Reply with one JSON object only:
{"answerable": "yes" | "partial" | "no", "evidence": "a short quote from the passages, or empty for no", "note": "one short sentence"}"""
RM_SYSTEM = """You see a question-answering task an assistant received: the question, the passages and instructions. Decide whether the passages let the assistant answer the question.

Options:
A. answer: the passages contain the answer, or relevant information for a useful partial answer.
B. refuse: the passages contain nothing that answers or meaningfully addresses the question.

Reply with one letter."""


def strata(rows: list[dict], excl: set[str]) -> dict[str, dict]:
    """{原文: {row, refused}}，refused = 这篇原文上 RAGTruth 各模型回答里拒答的个数。"""
    by = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["task_type"] != "QA" or r["source_id"] in excl:
            continue
        x = by.setdefault(r["source_id"], {"row": r, "refused": 0, "n": 0})
        x["n"] += 1
        x["refused"] += "unable to answer" in r["response"].lower()
    return by


def pick(by: dict[str, dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}|refuse")
    ids = sorted(by)
    hi = [i for i in ids if by[i]["refused"] >= 2]
    rest = [i for i in ids if by[i]["refused"] < 2]
    rng.shuffle(rest)
    chosen = hi + rest[:max(0, n - len(hi))]
    return [{"id": i, "source_id": i, "source": by[i]["row"]["source"], "refused_models": by[i]["refused"], "n_models": by[i]["n"]} for i in sorted(chosen)]


def cmd_pick_rest(args) -> None:
    """剩下没标的 QA 原文全部追加到 items.jsonl（一次给 Claude 20 题，快速挑出要拒答的）。排除规则同 pick。"""
    excl = excluded_sources() | {r["source_id"] for r in read_jsonl("outputs/rt/gold/seed_ids.jsonl")} | {r["source_id"] for r in read_jsonl("outputs/rt/ansgrpo/dev.jsonl")}
    by = strata(list(read_jsonl(POOL)), excl)
    have = {x["id"] for x in read_jsonl(D / "items.jsonl")}
    new = [{"id": i, "source_id": i, "source": by[i]["row"]["source"], "refused_models": by[i]["refused"], "n_models": by[i]["n"]} for i in sorted(by) if i not in have]
    with open(D / "items.jsonl", "a", encoding="utf-8", newline="\n") as f:
        for x in new:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("PICKED_REST", len(new), "total", len(have) + len(new), dict(sorted(collections.Counter(x["refused_models"] for x in new).items())), flush=True)


def cmd_pick(args) -> None:
    excl = excluded_sources() | {r["source_id"] for r in read_jsonl("outputs/rt/gold/seed_ids.jsonl")} | {r["source_id"] for r in read_jsonl("outputs/rt/ansgrpo/dev.jsonl")}
    items = pick(strata(list(read_jsonl(POOL)), excl), args.n, args.seed)
    assert len(items) <= args.max_calls, "超出 Claude 调用预算"
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("PICKED", len(items), "by refused models", dict(sorted(collections.Counter(x["refused_models"] for x in items).items())), flush=True)


def valid(o) -> dict | None:
    if not isinstance(o, dict) or o.get("answerable") not in ("yes", "partial", "no"):
        return None
    return {"answerable": o["answerable"], "evidence": str(o.get("evidence") or ""), "note": str(o.get("note") or "")}


def cmd_label(args) -> None:
    items = list(read_jsonl(D / "items.jsonl"))
    assert len(items) <= args.max_calls, "超出 Claude 调用预算"
    llm = LLM(LLMConfig(model=args.claude, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(it: dict) -> dict | None:
        time.sleep(args.gap)
        out = llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["source"]}])
        o = valid(parse_json_obj(out))
        if o is None:
            print("invalid output", it["id"], out[:200], flush=True)
            return None
        return {**{k: v for k, v in it.items() if k != "source"}, **o}

    map_resumable(items, run, D / "labels.jsonl", concurrency=args.concurrency)
    rows = kept(D / "labels.jsonl")
    print("LABEL_DONE", len(rows), dict(collections.Counter(r["answerable"] for r in rows)),
          {k: dict(collections.Counter(r["answerable"] for r in rows if r["refused_models"] == k)) for k in sorted({r["refused_models"] for r in rows})}, flush=True)


BATCH_SYSTEM = SYSTEM.replace("You receive the exact task an assistant received: the question, the passages, and instructions.",
                              "You receive several separate tasks, each marked [task <id>], each the exact input an assistant received: the question, the passages, and instructions. Judge each task on its own passages only.").replace(
    """Reply with one JSON object only:
{"answerable": "yes" | "partial" | "no", "evidence": "a short quote from the passages, or empty for no", "note": "one short sentence"}""",
    """Reply with one JSON object only, one entry per task in the given order:
{"results": [{"id": "<task id>", "answerable": "yes" | "partial" | "no", "evidence": "a short quote from the passages, or empty for no", "note": "one short sentence"}, ...]}""")


def batches(items: list[dict], done: set[str], per: int) -> list[dict]:
    todo = [x for x in items if x["id"] not in done]
    return [{"batch": f"b{k // per:03d}-" + todo[k]["id"], "items": todo[k:k + per]} for k in range(0, len(todo), per)]


def parse_batch(out: str, items: list[dict]) -> list[dict]:
    """只收 id 对得上、档位合法的；缺的下次再标。"""
    o = parse_json_obj(out)
    res = o.get("results") if isinstance(o, dict) else None
    want = {x["id"]: x for x in items}
    rows = []
    for r in res if isinstance(res, list) else []:
        v = valid(r) if isinstance(r, dict) else None
        if v and str(r.get("id")) in want:
            it = want.pop(str(r["id"]))
            rows.append({**{k: x for k, x in it.items() if k != "source"}, **v})
    return rows


def cmd_label_batch(args) -> None:
    items = list(read_jsonl(D / "items.jsonl"))
    done = {r["id"] for f in ("labels.jsonl", "labels_batch.jsonl") if (D / f).exists() for r in kept(D / f) if "answerable" in r}
    bs = batches(items, done, args.per)
    assert len(bs) <= args.max_calls, "超出 Claude 调用预算"
    llm = LLM(LLMConfig(model=args.claude, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(b: dict) -> list[dict] | None:
        time.sleep(args.gap)
        user = "\n\n".join(f"[task {x['id']}]\n{x['source'].strip()}" for x in b["items"])
        rows = parse_batch(llm.chat_one([{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": user}]), b["items"])
        if len(rows) < len(b["items"]):
            print("partial batch", b["batch"], len(rows), "/", len(b["items"]), flush=True)
        return [{**r, "batch": b["batch"]} for r in rows] or None

    print("BATCHES", len(bs), "items", sum(len(b["items"]) for b in bs), flush=True)
    map_resumable(bs, run, D / "labels_batch.jsonl", key="batch", concurrency=args.concurrency)
    rows = all_labels()
    print("LABEL_DONE", len(rows), "of", len(items), dict(collections.Counter(r["answerable"] for r in rows)),
          {k: dict(collections.Counter(r["answerable"] for r in rows if r["refused_models"] == k)) for k in sorted({r["refused_models"] for r in rows})}, flush=True)


def all_labels() -> list[dict]:
    by = {}
    for f in ("labels.jsonl", "labels_batch.jsonl"):
        if (D / f).exists():
            for r in kept(D / f):
                if "answerable" in r:
                    by.setdefault(r["id"], r)
    return [by[k] for k in sorted(by)]


def split_of(source_id: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}|refuse|{source_id}".encode()).hexdigest(), 16) % 100
    return "calib" if h < 15 else "test" if h < 35 else "train"


def build_rows(labels: list[dict], src: dict[str, str]) -> list[dict]:
    out = []
    for r in labels:
        lab = "B" if r["answerable"] == "no" else "A"
        out.append({"id": f"refuse-{r['source_id']}", "prompt": [{"role": "system", "content": RM_SYSTEM}, {"role": "user", "content": src[r["source_id"]].strip()}],
                    "completion": lab, "label": lab, "group": r["source_id"], "answerable": r["answerable"], "refused_models": r["refused_models"], "task_type": "QA"})
    return out


QA_RE = re.compile(r"^(?P<head>.*?question:\n)(?P<q>.*?)\n(?P<mid>Bear in mind.*?passages:\n)(?P<passages>.*?)(?P<tail>\n\nIn case the passages.*)$", re.S)
WORD = re.compile(r"[a-z]{4,}")


def splice(a: str, b: str) -> str | None:
    """a 的问题配 b 的段落（其余提示词不变）。格式对不上返回 None。"""
    ma, mb = QA_RE.match(a), QA_RE.match(b)
    if not ma or not mb:
        return None
    return ma["head"] + ma["q"] + "\n" + ma["mid"] + mb["passages"] + ma["tail"]


def overlap(a: str, b: str) -> float:
    """问题和段落的词重叠（4 个字母以上的词）：防止换来的段落碰巧也能回答。"""
    ma, mb = QA_RE.match(a), QA_RE.match(b)
    q, p = set(WORD.findall(ma["q"].lower())), set(WORD.findall(mb["passages"].lower()))
    return len(q & p) / max(1, len(q))


def synth_rows(labels: list[dict], src: dict[str, str], n: int, seed: int) -> list[dict]:
    """程序造的"答不了"：Claude 判为 yes 的题，问题配另一篇话题不相干的原文段落（问题里的词在新段落里出现不到 20%）。只用训练集原文。"""
    rng = random.Random(f"{seed}|refusesynth")
    ok = sorted(r["source_id"] for r in labels if r["answerable"] == "yes" and split_of(r["source_id"], seed) == "train" and QA_RE.match(src[r["source_id"]]))
    out, tries = [], 0
    while len(out) < n and tries < 50 * n and len(ok) > 1:
        tries += 1
        a, b = rng.sample(ok, 2)
        if overlap(src[a], src[b]) >= 0.2:
            continue
        out.append({"id": f"refuse-synth-{a}-{b}", "prompt": [{"role": "system", "content": RM_SYSTEM}, {"role": "user", "content": splice(src[a], src[b]).strip()}],
                    "completion": "B", "label": "B", "group": a, "answerable": "no", "refused_models": -1, "task_type": "QA", "origin": "synth_mismatch"})
    return out


PASSAGE = re.compile(r"(?m)^(?=passage \d+:)")
NORM = re.compile(r"\W+")


def drop_answer_passage(source: str, evidence: str) -> str | None:
    """删掉 Claude 证据所在的那一段，其余段落重新编号，提示里 three passages 改成 two passages。
    证据对不上任何一段、或同时出现在几段里时返回 None（删了也可能还答得了）。"""
    m = QA_RE.match(source)
    if not m or not evidence.strip():
        return None
    parts = [x for x in PASSAGE.split(m["passages"]) if x.strip()]
    key = NORM.sub(" ", evidence.lower()).strip()[:60]
    hit = [i for i, x in enumerate(parts) if key and key in NORM.sub(" ", x.lower())]
    if len(parts) != 3 or len(hit) != 1:
        return None
    keep = [re.sub(r"^passage \d+:", f"passage {k + 1}:", x) for k, x in enumerate(p for i, p in enumerate(parts) if i != hit[0])]
    mid = m["mid"].replace("following three passages", "following two passages")
    if mid == m["mid"]:
        return None
    return m["head"] + m["q"] + "\n" + mid + "".join(keep).rstrip("\n") + m["tail"]


REWRITE_SYSTEM = """You help build training data for deciding when an assistant should refuse to answer from given passages. Real cases where the passages cannot answer look like this: the passages are on the right topic, but the question asks for a specific thing they never cover. Real examples (question, and why the passages could not answer it):
{examples}

For each task below (marked [task <id>], with its question and passages), write a new question that a real user could plausibly ask about the same topic, changed as little as possible, so that these passages no longer answer it, in the same way as the examples: a different specific product, person, place or time; an added condition the passages do not cover; asking why or how when they only say what; asking for a comparison they never make. The new question must stay on the topic of the passages, read naturally (same casual style, typos are fine), and must not be answerable from the passages, even partly.

Reply with one JSON object only, one entry per task in the given order:
{{"results": [{{"id": "<task id>", "question": "the new question", "how": "one short phrase: what you changed"}}, ...]}}"""


def replace_question(source: str, q: str) -> str | None:
    m = QA_RE.match(source)
    if not m or not q.strip() or "\n" in q.strip():
        return None
    return m["head"] + q.strip() + "\n" + m["mid"] + m["passages"] + m["tail"]


def cmd_nearmiss(args) -> None:
    """拒答题至少 50 道，构造的要和真实拒答题很像。真实的只有 27 道（段落话题对，但问的是段落没讲到的具体东西）。两种造法，都只用训练集原文：
      rewrite：Claude 判 yes 的题，拿训练集里真实的拒答题当范例，让 Claude 把问题改一点点（换对象、加限定、问为什么、要比较），使原段落答不了；
      drop：Claude 判 yes 的题，删掉答案所在的那一段。
    两种合在一起，再一次 20 题单独交给 Claude 复核；判 no 的当拒答样本（B），还答得了的当难的"应该回答"样本（A）。"""
    rng = random.Random(f"{args.seed}|nearmiss")
    src = {x["id"]: x for x in read_jsonl(D / "items.jsonl")}
    labels = all_labels()
    yes = [r for r in labels if r["answerable"] == "yes" and split_of(r["source_id"], args.seed) == "train"]
    rng.shuffle(yes)
    real = [r for r in labels if r["answerable"] == "no" and split_of(r["source_id"], args.seed) == "train"]   # 范例只用训练集的，评测集不进构造
    ex = "\n".join(f"- {QA_RE.match(src[r['id']]['source'])['q'].strip()} — {r['note']}" for r in real if QA_RE.match(src[r["id"]]["source"]))
    llm = LLM(LLMConfig(model=args.claude, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))
    rw_items = [{"id": r["id"], "source": src[r["id"]]["source"]} for r in yes[:args.n_rewrite]]
    rw_bs = [{"batch": f"rw{k // 10:03d}-" + rw_items[k]["id"], "items": rw_items[k:k + 10]} for k in range(0, len(rw_items), 10)]
    sys_rw = REWRITE_SYSTEM.format(examples=ex)

    def gen(b: dict) -> list[dict] | None:
        time.sleep(args.gap)
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": sys_rw}, {"role": "user", "content": "\n\n".join(f"[task {x['id']}]\n{x['source'].strip()}" for x in b["items"])}]))
        res = o.get("results") if isinstance(o, dict) else None
        want = {x["id"] for x in b["items"]}
        rows = [{"id": str(r["id"]), "question": str(r.get("question") or ""), "how": str(r.get("how") or ""), "batch": b["batch"]} for r in (res if isinstance(res, list) else []) if isinstance(r, dict) and str(r.get("id")) in want]
        return rows or None

    print("REWRITE examples", len(real), "items", len(rw_items), "batches", len(rw_bs), flush=True)
    map_resumable(rw_bs, gen, D / "rewrites.jsonl", key="batch", concurrency=args.concurrency)
    cands = []
    for r in kept(D / "rewrites.jsonl"):
        new = replace_question(src[r["id"]]["source"], r["question"])
        if new:
            cands.append({"id": f"rw-{r['id']}", "source_id": r["id"], "source": new, "refused_models": -2, "n_models": 0, "method": "rewrite", "how": r["how"]})
    for r in yes[args.n_rewrite:]:
        new = drop_answer_passage(src[r["id"]]["source"], r.get("evidence", ""))
        if new and len([c for c in cands if c["method"] == "drop"]) < args.n_nearmiss:
            cands.append({"id": f"nm-{r['id']}", "source_id": r["id"], "source": new, "refused_models": -2, "n_models": 0, "method": "drop"})
    cands = sorted(cands, key=lambda x: x["id"])
    with open(D / "items_nearmiss.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in cands:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    bs = batches(cands, set(), args.per)
    assert len(bs) + len(rw_bs) <= args.max_calls

    def run(b: dict) -> list[dict] | None:
        time.sleep(args.gap)
        user = "\n\n".join(f"[task {x['id']}]\n{x['source'].strip()}" for x in b["items"])
        rows = parse_batch(llm.chat_one([{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": user}]), b["items"])
        return [{**r, "batch": b["batch"]} for r in rows] or None

    print("NEARMISS candidates", len(cands), "batches", len(bs), flush=True)
    map_resumable(bs, run, D / "labels_nearmiss.jsonl", key="batch", concurrency=args.concurrency)
    rows = [r for r in kept(D / "labels_nearmiss.jsonl") if "answerable" in r]
    meth = {x["id"]: x["method"] for x in cands}
    print("NEARMISS_DONE", len(rows), {m: dict(collections.Counter(r["answerable"] for r in rows if meth.get(r["id"]) == m)) for m in ("rewrite", "drop")}, flush=True)


def nearmiss_rows() -> list[dict]:
    if not (D / "labels_nearmiss.jsonl").exists():
        return []
    items = list(read_jsonl(D / "items_nearmiss.jsonl"))
    src = {x["id"]: x["source"] for x in items}
    meth = {x["id"]: x.get("method", "drop") for x in items}
    out = []
    for r in kept(D / "labels_nearmiss.jsonl"):
        if "answerable" not in r:
            continue
        lab = "B" if r["answerable"] == "no" else "A"
        out.append({"id": f"refuse-{r['id']}", "prompt": [{"role": "system", "content": RM_SYSTEM}, {"role": "user", "content": src[r["id"]].strip()}],
                    "completion": lab, "label": lab, "group": r["source_id"], "answerable": r["answerable"], "refused_models": -2, "task_type": "QA", "origin": "nearmiss_" + meth.get(r["id"], "drop")})
    return out


def cmd_build(args) -> None:
    src = {x["id"]: x["source"] for x in read_jsonl(D / "items.jsonl")}
    labels = all_labels()
    rows = build_rows(labels, src)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        parts[split_of(r["group"], args.seed)].append(r)
    meta = {"max_tokens": 3072, "epochs": 2}
    for sp in ("train", "calib", "test"):
        xs = parts[sp]
        if sp == "train":   # 真实的"答不了"太少（350 题里 23 道）：训练集加程序造的问题、段落错配；校准、测试只用 Claude 标的真实题
            xs = xs + nearmiss_rows() + synth_rows(labels, src, args.n_synth, args.seed)
        with open(OUT / f"{sp}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        meta[sp] = {"n": len(xs), "label": dict(collections.Counter(x["label"] for x in xs)), "origin": dict(collections.Counter(x.get("origin", "claude") for x in xs))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "pick_rest", "label", "label_batch", "nearmiss", "build"])
    ap.add_argument("--n_nearmiss", type=int, default=40, help="nearmiss：删答案段落的条数")
    ap.add_argument("--n_rewrite", type=int, default=80, help="nearmiss：改问题的条数（主要方法，最像真实拒答题）")
    ap.add_argument("--per", type=int, default=10, help="label_batch：每次调用几题")
    ap.add_argument("--n_synth", type=int, default=180, help="build：训练集里程序造的「答不了」条数")
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--max_calls", type=int, default=400, help="Claude 调用上限（约 300–400 次）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--gap", type=float, default=5.0)
    args = ap.parse_args()
    {"pick": cmd_pick, "pick_rest": cmd_pick_rest, "nearmiss": cmd_nearmiss, "label": cmd_label, "label_batch": cmd_label_batch, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
