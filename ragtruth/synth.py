"""合成错误句：把正确回答里的一句改成有问题的句子，成对产出逐句训练样本（改后句 = 幻觉，原句 = 无幻觉）。

种子：train_pool（与 train_sub、测试集不重叠）里人工标注整条回答都没问题的回答，按任务类型分层，
每条回答挑一句有事实内容的句子；类型 conflict / baseless 各半，每条再随机指定一种改法（tactic），让错误分布更散。

生成（默认服务器上的 Qwen3.5-35B-A3B Int4）：一次调用写出
- 改后的句子；
- 改后句的 evidence / reason（训练目标，写法与 ragtruth.annotate 一致）；
- 原句的 evidence / reason（成对加入的无幻觉样本的训练目标）。
few-shot 用 train_sub 里同任务、同类型的真实幻觉句（人工标注 + 35B 补写的证据和说明），让错误风格接近真实回答。

校验（不通过把问题反馈给模型重写，最多 retries 次）：改动幅度、长度、证据逐字可定位、conflict 必须给证据、
说明不含结论词且与类型一致（ragtruth.task.sentence_target_ok）。
复核（verify）：另起一次调用，把原句和改后句以随机顺序交给模型逐一对照原文判断，原句须判为有依据、改后句须判为
与原文矛盾 / 原文没有且与指定类型一致，不满足的丢弃。复核只看对错不看难易，不会把难样本筛掉。

用法：
  vllm serve /root/autodl-tmp/models/Qwen3.5-35B-A3B-GPTQ-Int4 --served-model-name gen --max-model-len 16384 \
      --limit-mm-per-prompt '{"image":0,"video":0}'
  python -m ragtruth.synth --n 3000 --out outputs/rt/synth/r1.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth.annotate import render_response
from ragtruth.task import (LABEL_WORDS, TIPS, evidence_in_source, expand_claims, repair_evidence,
                           sentence_target_ok)

TYPE_DEF = {
    "conflict": "the sentence contradicts the source: it states something the source says differently.",
    "baseless": "the sentence adds information that the source does not contain (it may still sound plausible or "
                "even be true in the real world).",
}
TACTICS = {
    "conflict": ["a number, quantity or rating", "a date, time or duration", "a name, place or other entity",
                 "an attribute or property of something", "who did what (swap the roles of entities)",
                 "negation or polarity (allowed vs not allowed, increase vs decrease)",
                 "cause and effect, or the order of events"],
    "baseless": ["one extra specific detail (a number, name, place or date)", "one extra item in a list",
                 "a reason or motive the source does not give", "a conclusion or generalization beyond the source",
                 "an evaluation or recommendation the source does not make",
                 "a plausible real-world fact that is not in the source"],
}

PROMPT = """You are creating training data for a model that detects unsupported or contradicting sentences in an assistant's response. Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence of this response is supported by the source.

{input}

Rewrite sentence S{idx} so that it contains exactly one error of this kind:
- {kind}: {type_def}
- Use this tactic: {tactic}.

Make the error realistic and hard to notice, like the mistakes real assistants make:
- change as little as possible; keep the wording, style and length close to the original; the response must still read naturally;
- the error must be wrong with respect to the source, not a paraphrase or a harmless rewording;
- no absurd values, no contradiction with the other sentences of the response, no hedges such as "reportedly".

Real {kind} sentences written by assistants on similar tasks (for style only, they belong to other sources):
{examples}

Then analyse both versions of S{idx} against the source, as a careful fact-checker would:
- "evidence": copy, word for word, the part of the source that the rewritten sentence {evidence_what}. Copy one contiguous stretch, as short as possible; for a long passage copy its first few words and its last few words joined by " ... " (each part at least 3 words).{evidence_null}
- "reason": one English sentence of at most 30 words: what the source says and what the rewritten sentence says, concretely (for example "The source says X, while the sentence says Y" or "The source does not mention Z").
- "original_evidence": the part of the source, word for word, that supports the original S{idx} (null only if it has no factual content).
- "original_reason": one English sentence of at most 30 words describing how the source supports the original S{idx}.
Never use the words hallucination, faithful, conflict, baseless, ground truth.

{tips}

Output only one JSON object, nothing else:
{{"plan": "where and how you introduce the error", "sentence": "the rewritten S{idx}", "evidence": "...", "reason": "...", "original_evidence": "...", "original_reason": "..."}}"""

EVIDENCE_WHAT = {"conflict": "contradicts",
                 "baseless": "is most closely related to (the passage it builds on)"}
EVIDENCE_NULL = {"conflict": " Never null.",
                 "baseless": " Use null only if the source has nothing related to the sentence."}

VERIFY_PROMPT = """Below are a source (the exact input an assistant received) and the assistant's response. Two versions of sentence S{idx} are given. Judge each version against the source only, not against world knowledge.

{input}

[version A of S{idx}] {a}
[version B of S{idx}] {b}

For each version answer one of:
- supported: everything it states is supported by the source (paraphrases and simple inferences count as supported);
- contradicts: it states something the source says differently;
- unsupported: it adds information that the source does not contain.
Also say whether each version reads naturally as part of the response.

Output only JSON: {{"A": "supported|contradicts|unsupported", "B": "supported|contradicts|unsupported", "A_natural": true, "B_natural": true}}"""
VERIFY_TYPE = {"contradicts": "conflict", "unsupported": "baseless", "supported": "none"}

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ---------- 种子和示例 ----------

def eligible_sentences(s: dict, min_chars: int = 40) -> list[int]:
    """可改写的句号（1 起）：有一定长度、不是以冒号结尾的引导句。"""
    return [i for i, t in enumerate(s["sentences"], 1) if len(t) >= min_chars and not t.rstrip().endswith(":")]


def pick_seeds(pool: list[dict], n: int, seed: int = 0, exclude: set[tuple[str, int]] | None = None) -> list[dict]:
    """全对的回答按任务类型分层抽 n 条，每条挑一句、指定类型（conflict / baseless 交替）和改法。
    exclude：之前各轮用过的 (回答 id, 句号)，迭代时每轮改的都是没改过的句子。按 source 整体排除不够用：
    train_pool 里全对回答对应的 source 约 2,080 个，第 1 轮就用掉 1,310 个。"""
    rng = random.Random(seed)
    exclude = exclude or set()
    free = lambda s: [i for i in eligible_sentences(s) if (s["id"], i) not in exclude]  # noqa: E731
    by_task = collections.defaultdict(list)
    for s in pool:
        if s["label"] == "faithful" and not s["bad"] and free(s):
            by_task[s["task_type"]].append(s)
    tasks = sorted(by_task)
    out = []
    for k, task in enumerate(tasks):
        want = n // len(tasks) + (k < n % len(tasks))
        for j, s in enumerate(rng.sample(by_task[task], min(want, len(by_task[task])))):
            kind = ("conflict", "baseless")[j % 2]
            idx = rng.choice(free(s))
            out.append({"id": f"{s['id']}-S{idx}-syn", "resp": s, "idx": idx, "type": kind,
                        "tactic": rng.choice(TACTICS[kind])})
    rng.shuffle(out)
    return out


def example_pool(annotated: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """train_sub 里训练目标合格的真实幻觉句，按 (任务类型, 类型) 分组。"""
    pool = collections.defaultdict(list)
    for s in annotated:
        for c in expand_claims(s, require_analysis=True):
            if c["label"] == "hallucination":
                pool[(c["task_type"], c["type"])].append(c)
    return pool


def format_examples(examples: list[dict]) -> str:
    out = []
    for c in examples:
        ev = "none" if c["evidence"] is None else f'"{c["evidence"]}"'
        out.append(f'- sentence: {c["sentences"][c["idx"] - 1]}\n  evidence: {ev}\n  reason: {c["reason"]}')
    return "\n".join(out)


HARD_EXAMPLES = """
Earlier rewrites (on other sources) that a strong detector failed to catch; they show how subtle a good error can be. Do not copy their content or tactic:
{hard}
"""


def format_hard_examples(recs: list[dict]) -> str:
    """难样本示例：原句 → 改后句 → 原因（合成记录格式，见 ragtruth.synth / ragtruth.mine）。"""
    return "\n".join(f'- original: {r["original"]}\n  rewritten: {r["sentence"]}\n  why it is wrong: {r["reason"]}'
                     for r in recs)


def build_prompt(seed: dict, examples: list[dict], hard_examples: list[dict] | None = None) -> str:
    """hard_examples：之前被强 critic 漏检、且过了全文核对的合成样本，作为额外示例（2026-10-27 用户建议，
    增加多样性和难度）；None 时与原 prompt 完全相同。"""
    kind = seed["type"]
    p = PROMPT.format(input=render_response(seed["resp"]), idx=seed["idx"], kind=kind, type_def=TYPE_DEF[kind],
                      tactic=seed["tactic"], examples=format_examples(examples) or "(none)",
                      evidence_what=EVIDENCE_WHAT[kind], evidence_null=EVIDENCE_NULL[kind], tips=TIPS)
    if hard_examples:
        marker = "\nThen analyse both versions"
        p = p.replace(marker, HARD_EXAMPLES.format(hard=format_hard_examples(hard_examples)) + marker, 1)
    return p


# ---------- 自由选择版：句子、类型、改法都由生成器自己定（2026-10-28 用户提出） ----------
# 指定版随机挑句、指定类型和改法，会把错误分布推离真实分布（真实幻觉集中在某些位置和某些句子上）；
# 自由版只给整条回答和改法清单（仅供参考），生成器自己判断哪里最像真实模型会出错。生成器经过真实错误 SFT
# （ragtruth.realgen build --free）后，选择本身就学的是真实分布；DPO 的 KL 约束让它不偏离太远。

FREE_PROMPT = """You are creating training data for a model that detects unsupported or contradicting sentences in an assistant's response. Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence of this response is supported by the source.

{input}

Choose one sentence of this response and rewrite it so that it contains exactly one error of one of these kinds:
- conflict: {conflict_def}
- baseless: {baseless_def}

Choose the sentence, the kind of error and the way of introducing it yourself: pick what a real assistant writing this response would most plausibly get wrong. Some common ways, for reference only (you may use another):
- conflict: {conflict_tactics}
- baseless: {baseless_tactics}

Make the error realistic and hard to notice, like the mistakes real assistants make:
- change as little as possible; keep the wording, style and length close to the original; the response must still read naturally;
- the error must be wrong with respect to the source, not a paraphrase or a harmless rewording;
- no absurd values, no contradiction with the other sentences of the response, no hedges such as "reportedly".

Real sentences with such errors written by assistants on similar tasks (for style only, they belong to other sources):
{examples}

Then analyse both versions of the chosen sentence against the source, as a careful fact-checker would:
- "evidence": copy, word for word, the part of the source that the rewritten sentence contradicts (conflict, never null) or is most closely related to (baseless; null only if the source has nothing related). Copy one contiguous stretch, as short as possible; for a long passage copy its first few words and its last few words joined by " ... " (each part at least 3 words).
- "reason": one English sentence of at most 30 words: what the source says and what the rewritten sentence says, concretely (for example "The source says X, while the sentence says Y" or "The source does not mention Z").
- "original_evidence": the part of the source, word for word, that supports the original sentence (null only if it has no factual content).
- "original_reason": one English sentence of at most 30 words describing how the source supports the original sentence.
Never use the words hallucination, faithful, conflict, baseless, ground truth in the sentence, the evidence or the reasons (only in the "type" field).

{tips}

Output only one JSON object, nothing else:
{{"plan": "which sentence you chose, why a real assistant could get it wrong, and how you introduce the error", "idx": <number of the chosen sentence>, "type": "conflict or baseless", "tactic": "the way you introduced the error", "sentence": "the rewritten sentence", "evidence": "...", "reason": "...", "original_evidence": "...", "original_reason": "..."}}"""


def build_prompt_free(resp: dict, examples: list[dict]) -> str:
    return FREE_PROMPT.format(input=render_response(resp), conflict_def=TYPE_DEF["conflict"],
                              baseless_def=TYPE_DEF["baseless"], conflict_tactics="; ".join(TACTICS["conflict"]),
                              baseless_tactics="; ".join(TACTICS["baseless"]),
                              examples=format_examples(examples) or "(none)", tips=TIPS)


def free_examples(pool: dict, task_type: str, rng: random.Random, n: int = 3, exclude_source: str | None = None) -> list[dict]:
    """自由版的示例：同任务、两种类型混在一起抽（不暗示该造哪种错）。"""
    cands = [c for t in ("conflict", "baseless") for c in pool.get((task_type, t), [])
             if c["source"] != exclude_source]
    return rng.sample(cands, min(n, len(cands)))


def check_free(obj: dict | None, resp: dict) -> tuple[dict | None, dict, list[str]]:
    """先解析生成器自己选的句号、类型、改法，组成和指定版一样的种子，再走 check。返回 (种子, 整理后的结果, 问题)。"""
    if obj is None:
        return None, {}, ["output is not a valid JSON object"]
    try:
        idx = int(obj.get("idx"))
    except (TypeError, ValueError):
        return None, {}, ["idx must be the number of the chosen sentence"]
    kind = str(obj.get("type") or "").strip().lower()
    if not 1 <= idx <= len(resp["sentences"]):
        return None, {}, [f"idx must be between 1 and {len(resp['sentences'])}"]
    if kind not in ("conflict", "baseless"):
        return None, {}, ["type must be conflict or baseless"]
    seed = {"resp": resp, "idx": idx, "type": kind, "tactic": str(obj.get("tactic") or "").strip()}
    res, problems = check(obj, seed)
    return seed, res, problems


def pick_responses(pool: list[dict], n: int, seed: int = 0, exclude: set[str] | None = None) -> list[dict]:
    """自由版的种子：全对、至少两句可改的回答，按任务类型分层抽 n 条（不指定句子）。"""
    rng = random.Random(seed)
    exclude = exclude or set()
    by_task = collections.defaultdict(list)
    for s in sorted(pool, key=lambda s: s["id"]):
        if s["label"] == "faithful" and not s["bad"] and s["id"] not in exclude and len(eligible_sentences(s)) >= 2:
            by_task[s["task_type"]].append(s)
    tasks = sorted(by_task)
    out = []
    for k, task in enumerate(tasks):
        want = n // len(tasks) + (k < n % len(tasks))
        out += [{"id": f"{s['id']}-free", "resp": s, "free": True}
                for s in rng.sample(by_task[task], min(want, len(by_task[task])))]
    rng.shuffle(out)
    return out


# ---------- 解析和校验 ----------

def parse_json_obj(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.split("</think>")[-1], re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _evidence(ev, source: str):
    """null / "none" → None；字符串先做模糊修正。"""
    if ev is None or (isinstance(ev, str) and ev.strip().lower() in ("", "none", "null")):
        return None
    return repair_evidence(ev, source) or ev if isinstance(ev, str) else ev


def check(obj: dict | None, seed: dict, max_edit_frac: float = 0.6, max_len_ratio: float = 1.6) -> tuple[dict, list[str]]:
    """返回 (整理后的结果, 问题列表)；问题列表为空表示通过。"""
    if obj is None:
        return {}, ["output is not a valid JSON object"]
    src, orig = seed["resp"]["source"], seed["resp"]["sentences"][seed["idx"] - 1]
    new = (obj.get("sentence") or "").strip()
    res = {"sentence": new, "evidence": _evidence(obj.get("evidence"), src), "reason": (obj.get("reason") or "").strip(),
           "original_evidence": _evidence(obj.get("original_evidence"), src),
           "original_reason": (obj.get("original_reason") or "").strip(), "plan": obj.get("plan")}
    problems = []
    if not new:
        return res, ["the rewritten sentence is empty"]
    d = edit_distance(tokens(orig), tokens(new))
    if d == 0:
        problems.append("the rewritten sentence is identical to the original")
    elif d > max(8, max_edit_frac * len(tokens(orig))):
        problems.append(f"the rewrite changes too much ({d} tokens); change as little as possible")
    if len(new) > max_len_ratio * len(orig) + 20:
        problems.append("the rewritten sentence is much longer than the original; keep the length close")
    if LABEL_WORDS.search(new):
        problems.append("the rewritten sentence must not contain words such as hallucination or faithful")
    for key, ev, reason, kind in (("", res["evidence"], res["reason"], seed["type"]),
                                  ("original_", res["original_evidence"], res["original_reason"], "none")):
        if isinstance(ev, str) and not evidence_in_source(ev, src):
            problems.append(f"{key}evidence is not copied word for word from the source: {ev[:80]!r}")
        elif ev is None and kind == "conflict":
            problems.append("evidence must quote the part of the source that the rewritten sentence contradicts")
        elif not sentence_target_ok(ev, reason, kind, src):
            what = ("say concretely what is missing from or different in the source" if kind != "none"
                    else "describe how the source supports it, without saying anything is missing or different")
            problems.append(f"{key}reason must be non-empty, avoid the banned words, and {what}")
    return res, problems


def synth_one(gen: LLM, seed: dict, examples: list[dict], retries: int, extra: dict | None = None) -> dict:
    """通过校验返回整理后的结果；重试用完仍不通过返回 {"failed": 最后一次的问题列表}（统计失败原因用）。"""
    msgs = [{"role": "user", "content": build_prompt(seed, examples)}]
    for attempt in range(retries + 1):
        text = gen.chat_one(msgs, temperature=0.8 if attempt == 0 else 0.6, **(extra or {}))
        res, problems = check(parse_json_obj(text), seed)
        if not problems:
            return {**res, "attempts": attempt + 1}
        msgs = msgs[:1] + [{"role": "assistant", "content": text},
                           {"role": "user", "content": "Fix these problems and output the full JSON object again:\n- "
                                                       + "\n- ".join(problems)}]
    return {"failed": problems}


def verify_one(gen: LLM, seed: dict, new: str, rng: random.Random, extra: dict | None = None) -> dict:
    """原句、改后句随机排成 A / B，交给模型逐一对照原文判断。"""
    orig = seed["resp"]["sentences"][seed["idx"] - 1]
    swap = rng.random() < 0.5
    a, b = (new, orig) if swap else (orig, new)
    text = gen.chat_one([{"role": "user", "content": VERIFY_PROMPT.format(
        input=render_response(seed["resp"]), idx=seed["idx"], a=a, b=b)}], temperature=0.0, **(extra or {}))
    obj = parse_json_obj(text) or {}
    o, n = ("A", "B") if not swap else ("B", "A")   # 原句、改后句各在哪个位置
    v_orig, v_new = VERIFY_TYPE.get(str(obj.get(o)).lower()), VERIFY_TYPE.get(str(obj.get(n)).lower())
    # natural 只记录不筛：35B 试跑 736 条里 223 条类型对但判为不自然，抽查多数读着正常（加 "highly"、"version 2.4"），
    # 复核模型看出哪句是改过的后倾向于说它不自然
    return {"orig": v_orig, "new": v_new, "natural": obj.get(f"{n}_natural"), "ok": verify_ok(v_orig, v_new, seed["type"])}


def verify_ok(v_orig: str | None, v_new: str | None, want: str) -> bool:
    return v_orig == "none" and v_new == want


# ---------- 成对的逐句样本 ----------

def to_claims(rec: dict) -> list[dict]:
    """一条合成结果 → [改后句（幻觉）, 原句（无幻觉）] 两条逐句样本，字段同 ragtruth.task.expand_claims。"""
    keep = {k: rec[k] for k in ("source", "task_type", "model")}
    i, sents = rec["idx"], rec["sentences"]
    new_sents = sents[:i - 1] + [rec["sentence"]] + sents[i:]
    base = {**keep, "split": "train", "origin": "synth", "idx": i, "coarse": False, "pair_id": rec["id"]}
    return [
        {**base, "id": f"{rec['id']}-S{i}", "response_id": rec["id"], "sentences": new_sents,
         "label": "hallucination", "type": rec["type"], "evidence": rec["evidence"], "reason": rec["reason"]},
        {**base, "id": f"{rec['id']}-orig-S{i}", "response_id": f"{rec['id']}-orig", "sentences": sents,
         "label": "faithful", "type": "none", "evidence": rec["original_evidence"], "reason": rec["original_reason"]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
    ap.add_argument("--examples", default="outputs/rt/annotated/train_sub35.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_shots", type=int, default=3)
    ap.add_argument("--out", default="outputs/rt/synth/r1.jsonl")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="之前各轮的合成结果（jsonl），它们改过的句子这轮不再用")
    args = ap.parse_args()

    gen = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=args.max_tokens,
                        concurrency=args.concurrency, timeout=600))
    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    used = {(r["response_id"], r["idx"]) for f in args.exclude for r in read_jsonl(f)
            if "response_id" in r and "idx" in r}   # 校验失败的记录没有这两个字段
    seeds = pick_seeds(list(read_jsonl(args.pool)), args.n, args.seed, used)
    print(f"excluding {len(used)} used sentences; {len(seeds)} seeds")
    pool = example_pool(list(read_jsonl(args.examples)))

    def run(seed: dict) -> dict | None:
        rng = random.Random(seed["id"])
        cands = pool.get((seed["resp"]["task_type"], seed["type"])) or []
        res = synth_one(gen, seed, rng.sample(cands, min(args.n_shots, len(cands))), args.retries, extra)
        if "failed" in res:
            return {"id": seed["id"], "type": seed["type"], "task_type": seed["resp"]["task_type"], **res}
        ver = None if args.no_verify else verify_one(gen, seed, res["sentence"], rng, extra)
        r = seed["resp"]
        return {"id": seed["id"], "response_id": r["id"], "idx": seed["idx"], "type": seed["type"],
                "tactic": seed["tactic"], "task_type": r["task_type"], "model": r["model"], "source": r["source"],
                "sentences": r["sentences"], "original": r["sentences"][seed["idx"] - 1], **res, "verify": ver}

    raw = args.out.replace(".jsonl", ".raw.jsonl")
    map_resumable(seeds, run, raw, concurrency=args.concurrency)
    rows = kept(raw)
    fails = collections.Counter(re.sub(r"[:(].*", "", p) for r in rows if "failed" in r for p in r["failed"])
    rows = [r for r in rows if "failed" not in r]
    ok = [r for r in rows if r["verify"] is None or verify_ok(r["verify"]["orig"], r["verify"]["new"], r["type"])]
    print(f"synthesized {len(rows)}/{len(seeds)} passed checks, {len(ok)} passed verify; "
          f"wrote {write_jsonl(args.out, ok)} -> {args.out}")
    print("by task/type:", collections.Counter((r["task_type"], r["type"]) for r in ok))
    print("check fails:", fails.most_common(8))
    print("verify fails:", collections.Counter((r["verify"]["orig"], r["verify"]["new"]) for r in rows
                                               if r["verify"] and r not in ok))


if __name__ == "__main__":
    main()
