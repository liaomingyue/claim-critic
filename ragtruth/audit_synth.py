"""Claude 抽查合成样本的正确性（2026-10-29，用户同意，约 $3）：RM3 打算直接用 35B 的判断，这里看它判对、判错的两组
各有没有明显的系统性问题；同时复核反馈 critic 一次都没识破（识破率 = 0）的有效样本：按方案这类先复核再用。

三组（同一个 gen_dpo 目录，默认 outputs/rt/closed/gtarget）：
  rate0    35B 判有效、critic-clean-gfree 判 4 次全没识破的，全部
  valid    35B 判有效、识破率 > 0 的，随机 n 条
  rejected 规则通过、35B 判不是错误（无效）的，随机 n 条

Claude 看到原文、整条回答、原句和改句、生成器给的类型 / 证据 / 理由，不知道分在哪组。输出：
  category  correct（改句确有错，类型、证据、理由都对）/ not_an_error（改句其实没错或只是同义改写）/
            wrong_explanation（改句有错，但类型、证据或理由不对）/ original_wrong（原句本身就与原文不符）
  natural   yes / no：真实助手写这条回答时会不会这样写错

用法：python -m ragtruth.audit_synth --dir outputs/rt/closed/gtarget --tag cleangfree --n 20
     本地 35B（思考模式）判同一批、和 Claude 比一致率（2026-10-29，用户：先用 35B）：
     python -m ragtruth.audit_synth --local --model judge --only outputs/rt/closed/gtarget/audit_synth.jsonl          --out outputs/rt/closed/gtarget/audit_synth_35b.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable

CATS = ("correct", "not_an_error", "wrong_explanation", "original_wrong", "uncertain")

SYSTEM = """You are checking synthetic training data for a hallucination detector in retrieval-augmented generation.
A generator took a response that was faithful to its source and rewrote one sentence so that it should contain an error (a conflict with the source, or baseless information not in the source). It also wrote the error type, the evidence from the source, and a reason.

Check the source yourself and judge this sample:
- correct: the rewritten sentence really is wrong with respect to the source, and the type, evidence and reason correctly describe that error.
- not_an_error: the rewritten sentence is actually supported by the source, or is only a harmless rewording, so it should not be labeled as an error.
- wrong_explanation: the rewritten sentence is wrong, but the type, the evidence or the reason does not correctly describe the error.
- original_wrong: the original sentence itself already conflicts with or goes beyond the source.

Also judge whether the error is natural: would a real assistant writing this response plausibly make this mistake (yes), or does it look artificial, absurd or obviously planted (no)?

Answer in this format:
<reason>one or two sentences</reason>
<category>correct|not_an_error|wrong_explanation|original_wrong</category>
<natural>yes|no</natural>"""


# v2（2026-10-29）：35B 思考模式用 v1 时偏宽松（Claude 判「其实没错」的 40 条里 15 条判成正确），且 19% 思考兜圈子到
# token 上限没给结论。v2 写清边界并给通用例子（不取自被评测的 83 条），要求简短推理；配合 --samples 多次采样取投票比例当概率。
SYSTEM_V2 = SYSTEM.replace("Also judge whether", """Where the line is:
- It is an error (correct, if the type, evidence and reason fit) only if the rewritten sentence makes a specific claim that the source contradicts or does not contain: a changed number, date, name, quantity or order; a fact attributed to the wrong entity, passage or step; a strengthened or weakened certainty or scope ("may" becoming "will", "some" becoming "all"); or a specific added detail the source does not give.
- It is not_an_error if the change keeps the meaning or only adds something vague: a synonym or near-synonym with the same meaning, a generic evaluative or purpose phrase that makes no specific factual claim ("making it a great choice", "suitable for many occasions", "which is helpful for users"), a reasonable inference the source directly supports, or a harmless reordering.

Examples (from other sources):
- Source: "The museum opens at 9 am." Rewrite: "The museum opens at 10 am." -> correct (conflict).
- Source: "The museum opens at 9 am." Rewrite: "The museum, founded in 1920, opens at 9 am." -> correct (baseless: a specific date the source does not give).
- Source: "The cafe offers free Wi-Fi." Rewrite: "The cafe offers free Wi-Fi, making it a convenient spot for visitors." -> not_an_error (vague purpose phrase).
- Source: "The company reported higher sales." Rewrite: "The company posted higher sales." -> not_an_error (same meaning).
- Source: "The bridge opened in 1998." Rewrite: "The bridge opened in 1996." with claimed type baseless -> wrong_explanation (it is a conflict, not baseless).

Keep your reasoning short: check the changed part against the source once, decide, and answer.

Also judge whether""")


# v3（2026-10-29，用户：边界不明确的偏向严格）：凡是原文没说、也不能直接推出的内容都算错，包括泛泛的目的 / 评价短语
# 和改变语气、确定性的近义替换；只有意思完全不变的改写才算 not_an_error；拿不准时判为有错。Claude 和 35B 用同一份定义。
SYSTEM_V3 = SYSTEM.replace("Also judge whether", """Use a strict standard. Where the line is:
- It is an error (correct, if the type, evidence and reason fit) whenever the rewritten sentence says anything the source does not state and that cannot be directly inferred from it. This includes: a changed number, date, name, quantity or order; a fact attributed to the wrong entity, passage or step; a strengthened or weakened certainty or scope; a specific added detail; and also an added evaluation, purpose, consequence or recommendation the source does not make ("making it a great choice", "suitable for many occasions", "which is helpful for users"), or a near-synonym that shifts the meaning, tone or certainty ("provided" becoming "promised", "may" becoming "is likely to").
- It is not_an_error only if the change keeps exactly the same meaning: a true synonym, a reordering, or a rewording that adds and removes nothing.
- When you are unsure whether the source supports the change, treat it as an error.

Examples (from other sources):
- Source: "The museum opens at 9 am." Rewrite: "The museum opens at 10 am." -> correct (conflict).
- Source: "The museum opens at 9 am." Rewrite: "The museum, founded in 1920, opens at 9 am." -> correct (baseless).
- Source: "The cafe offers free Wi-Fi." Rewrite: "The cafe offers free Wi-Fi, making it a convenient spot for visitors." -> correct (baseless: an evaluation the source does not make).
- Source: "The government has provided $2 million." Rewrite: "The government has promised $2 million." -> correct (conflict: provided is not the same as promised).
- Source: "The company reported higher sales." Rewrite: "The company posted higher sales." -> not_an_error (same meaning).
- Source: "The bridge opened in 1998." Rewrite: "The bridge opened in 1996." with claimed type baseless -> wrong_explanation (it is a conflict, not baseless).

Keep your reasoning short: check the changed part against the source once, decide, and answer.

Also judge whether""")


# v4（2026-10-29，用户：加一个不确定选项，让 35B 主动说不确定，送 Claude）：在 v3 的严格口径上加 uncertain。
# 何时用 uncertain 按 35B 与 Claude 的实际分歧写：意思是否改变取决于理解的近义替换 / 泛泛补充、原句本身可能不被支持、
# conflict 与 baseless 都说得通、证据只覆盖一部分。uncertain 不进训练标签，只用于分流给 Claude。
SYSTEM_V4 = (SYSTEM_V3
             .replace("- original_wrong: the original sentence itself already conflicts with or goes beyond the source.",
                      "- original_wrong: the original sentence itself already conflicts with or goes beyond the source.\n"
                      "- uncertain: you cannot decide with confidence between the categories above.")
             .replace("- When you are unsure whether the source supports the change, treat it as an error.",
                      "- Answer uncertain, instead of guessing, in cases like these: whether a near-synonym or a vague addition "
                      "changes the meaning depends on how the source is read; the original sentence itself may not be fully "
                      "supported by the source; both conflict and baseless describe the error equally well; the evidence covers "
                      "only part of the change; or the source passage needed to decide is ambiguous or garbled. If you can decide, "
                      "decide: use uncertain only when you would honestly be guessing.")
             .replace("<category>correct|not_an_error|wrong_explanation|original_wrong</category>",
                      "<category>correct|not_an_error|wrong_explanation|original_wrong|uncertain</category>"))


# v5（2026-10-30，用户：结合 Claude 的结论改提示词）：35B 判 correct、Claude 不同意的 49 条里，问题集中在
#   ① 冲突 / 无依据分不清、照单接受生成器报的类型（约 15）；② 不检查原句，原句里已有原文没说的评价、夸张（13）；
#   ③ 严格口径用过头，把删减、同指称改写、能直接推出的内容判成错（12）；④ 证据 / 理由没对准真正改错的地方（约 9）。
# 改成逐步检查、每步单独给结论，最后按规则合成类别；例子都是自编的，不取自这些样本。
SYSTEM_V5 = """You are a skeptical reviewer of synthetic training data for a hallucination detector in retrieval-augmented generation.
A generator took a response that was supposed to be faithful to its source and rewrote one sentence so that it should contain an error. It also wrote the error type (conflict or baseless), the evidence from the source, and a reason. Many of these samples are flawed; your job is to find the flaw if there is one. Check the source yourself; do not trust the generator's claims.

Work through these four steps in order.

Step 1. The original sentence. Check every part of the original sentence against the source, with the strict standard below. Is anything in it not stated in the source and not directly inferable from it, or contradicted by it? Typical cases: an evaluation or summary judgment the source does not make ("convenient", "ample", "popular", "is key to"), a detail taken from the wrong passage, a wrong number or citation. If the original already has such a problem, the sample is original_wrong, whatever the rewrite does.

Step 2. The change. Compare the rewritten sentence with the original and list exactly what changed. Then decide whether the changed part is an error, using this standard:
- It IS an error if the changed part says something the source does not state and that cannot be directly inferred from it: a changed number, date, name or quantity; a fact moved to the wrong entity, step or passage; a strengthened or weakened certainty or scope; an added specific detail; an added evaluation, purpose, consequence or comparison the source does not make; a near-synonym that changes meaning, tone or certainty.
- It is NOT an error (not_an_error) if the change only: removes something (an omission is not a hallucination unless what remains becomes false, e.g. dropping "unless your doctor says otherwise" turns a conditional rule into an absolute one); shortens or rewords a name for the same entity; spells out what the original already means ("free Wi-Fi" becoming "free Wi-Fi for guests"); keeps a list correct by naming only some of its items after "including" or "such as"; adds empty filler with no factual content ("through their various functions"); or states something the source supports when read as a whole, even if in different words or in another passage.

Step 3. The type. Decide the right type for the error yourself, then compare it with the claimed type.
- conflict: the source states something about this point, and the rewritten sentence says something incompatible with it (a different value, the opposite, a stated limit made absolute, a property given to the wrong thing).
- baseless: the source says nothing about this point either way, and the rewritten sentence adds it. In structured data, a field that is missing, null or None means "not stated": a claim about it is baseless, not conflict. A field that is True or False, or has a value, is stated: contradicting it is conflict.
If the change contains both kinds, the type must match the main error the reason describes.

Step 4. The evidence and reason. Does the claimed evidence quote the part of the source that actually shows the error (for conflict: the statement that is contradicted)? Does the reason describe the actual changed part and why it is wrong? If the rewrite changes several things, the reason must cover the changes that make it wrong. A reason built on a misreading of the source, or pointing at an unrelated part, is wrong.

Final category:
- original_wrong if step 1 found a problem;
- otherwise not_an_error if step 2 found no error;
- otherwise wrong_explanation if the claimed type differs from yours in step 3, or step 4 failed;
- otherwise correct.

Examples (made up for illustration):
- Source: "Lot parking: yes. Garage: no." Original: "The restaurant has ample parking." Rewrite: "The restaurant has ample parking, including a garage." -> original_wrong ("ample" is not in the source).
- Source: "Tickets cost $20." Rewrite: "Tickets cost over $20." claimed baseless -> wrong_explanation (the source states the price; this is a conflict).
- Source: "Delivery: None." Rewrite: "They do not deliver." claimed conflict -> wrong_explanation (None means not stated; this is baseless).
- Original: "Contact the Office of the City Clerk." Rewrite: "Contact the City Clerk." -> not_an_error (same entity).
- Source: "Bake for 20 minutes." Rewrite: "Bake for 20 minutes until golden." claimed baseless, reason: "the source does not mention golden color" -> correct.

Keep each step short. Answer in this format:
<original_ok>yes|no</original_ok>
<change_is_error>yes|no</change_is_error>
<right_type>conflict|baseless|none</right_type>
<explanation_ok>yes|no</explanation_ok>
<reason>one or two sentences</reason>
<category>correct|not_an_error|wrong_explanation|original_wrong</category>
<natural>yes|no</natural>"""


def render(rec: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(rec["sentences"], 1))
    return (f"<source>\n{rec['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Rewritten sentence: S{rec['idx']}\nOriginal: {rec['original']}\nRewritten: {rec['sentence']}\n"
            f"Claimed type: {rec['type']}\nClaimed evidence: {rec['evidence']}\nClaimed reason: {rec['reason']}")


def category_probs(choice) -> dict[str, float] | None:
    """一次输出里 <category> 标签后第一个 token 的概率，按四个类别的首 token 归一化（单次采样得到连续概率）。
    思考模式下只看 </think> 之后的那个 <category>。"""
    import math
    toks = (choice.logprobs.content if choice.logprobs else None) or []
    heads = {"correct": "correct", "not": "not_an_error", "wrong": "wrong_explanation", "original": "original_wrong",
             "uncertain": "uncertain"}
    text, start = "", 0
    for k, t in enumerate(toks):   # 从 </think> 之后开始找
        text += t.token
        if "</think>" in text[-12:]:
            start = k + 1
    seen = ""
    for t in toks[start:]:
        if seen.rstrip().endswith("<category>"):
            mass = {c: 0.0 for c in CATS}
            for c in t.top_logprobs:
                h = c.token.strip().lower()
                for pre, cat in heads.items():
                    if h and (pre.startswith(h) or h.startswith(pre)):
                        mass[cat] += math.exp(c.logprob)
                        break
            z = sum(mass.values())
            return {c: v / z for c, v in mass.items()} if z > 0 else None
        seen += t.token
    return None


def tag_of(out: str, name: str, allowed: tuple[str, ...]) -> str | None:
    m = re.findall(rf"<{name}>\s*([a-z_]+)\s*</{name}>", out)
    return m[-1] if m and m[-1] in allowed else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/rt/closed/gtarget")
    ap.add_argument("--tag", default="cleangfree", help="hard_<tag>.jsonl：反馈 critic 的识破率")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--local", action="store_true", help="用本地 vLLM 上的模型（思考模式）代替 Claude")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--only", default="", help="只判这个 jsonl 里出现过的 id（例如 Claude 判过的那批，用来比一致率）")
    ap.add_argument("--all", action="store_true", help="判全部候选（规则通过的），不抽样")
    ap.add_argument("--prompt", choices=["v1", "v2", "v3", "v4", "v5"], default="v1")
    ap.add_argument("--samples", type=int, default=1, help="--local：每条采样几次，类别取多数，另存各类比例")
    ap.add_argument("--logprobs", action="store_true", help="--local：读 <category> 处的 token 概率作为各类概率（单次采样）")
    ap.add_argument("--no_think", action="store_true", help="--local：关闭思考模式")
    ap.add_argument("--shard", type=int, default=0, help="多台机器分片：只判 id 排序后第 k % nshard == shard 条")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--steal", action="store_true", help="分片跑完后接着判其他分片还没出现在 --steal_from 文件里的（倒序，减少撞车）")
    ap.add_argument("--steal_from", nargs="*", default=[], help="其他分片的输出文件")
    ap.add_argument("--finish", nargs="*", default=[],
                    help="补结论：这些输出文件里没给结论（思考写到上限）的样本，把已写的分析交回模型、关掉思考只要最终答案")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    out = args.out or str(d / "audit_synth.jsonl")
    hard = d / f"hard_{args.tag}.jsonl"   # 只用来填 detect_rate，本地缺这个文件时留空
    rate = {r["id"]: r["detect_rate"] for r in kept(hard)} if hard.exists() else {}
    groups = collections.defaultdict(list)
    for r in sorted(kept(d / "valid.jsonl"), key=lambda r: r["id"]):
        for c in r["cands"]:
            if not c.get("rec"):
                continue
            key = f"{r['id']}#{c['j']}"
            rec = {**c["rec"], "id": key}
            if c["valid"]:
                groups["rate0" if rate.get(key) == 0 else "valid"].append(rec)
            elif c["rule_ok"]:
                groups["rejected"].append(rec)
    rng = random.Random(args.seed)
    if args.only or args.all:
        want = {r["id"] for r in kept(args.only)} if args.only else None
        items = [{"id": r["id"], "group": g, "rec": r} for g, rs in sorted(groups.items()) for r in rs
                 if want is None or r["id"] in want]
    else:
        items = [{"id": r["id"], "group": "rate0", "rec": r} for r in groups["rate0"]]
        for g in ("valid", "rejected"):
            items += [{"id": r["id"], "group": g, "rec": r} for r in rng.sample(groups[g], min(args.n, len(groups[g])))]
    rng.shuffle(items)
    if args.finish:   # 补结论只处理这些文件里 category 为空的样本，分析取自它们的 output（思考没结束时就是思考本身）
        rows = [r for f in args.finish if Path(f).exists() for r in kept(f)]
        concluded = {r["id"] for r in rows if r.get("category") is not None}   # 分片撞车时同一条可能别处已有结论
        prior = {r["id"]: r for r in rows if r.get("category") is None and r["id"] not in concluded}
        items = [{**it, "analysis": prior[it["id"]]["output"]} for it in items if it["id"] in prior]
    if args.nshard > 1:
        items = sorted(items, key=lambda it: it["id"])
        if args.steal:   # 判所有分片里还没人判过的，倒序
            done = {r["id"] for f in args.steal_from if Path(f).exists() for r in kept(f)}
            items = [it for it in reversed(items) if it["id"] not in done]
        else:
            items = [it for k, it in enumerate(items) if k % args.nshard == args.shard]
    print(json.dumps({g: len(v) for g, v in groups.items()}), "auditing", len(items))
    if args.local:
        llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=args.max_tokens,
                            temperature=0.0 if args.samples == 1 else 0.6, concurrency=args.concurrency, timeout=1800))
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": not args.no_think}}}
    else:
        llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))
        extra = {}

    system = {"v1": SYSTEM, "v2": SYSTEM_V2, "v3": SYSTEM_V3, "v4": SYSTEM_V4, "v5": SYSTEM_V5}[args.prompt]

    def run(it: dict) -> dict:
        rec = it["rec"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": render(rec)}]
        lp = None
        if it.get("analysis") is not None:   # 补结论：新的一轮，关思考，只要最终答案
            msgs = msgs + [{"role": "assistant", "content": "(analysis cut off)"},
                           {"role": "user", "content": "Your analysis was cut off before you answered. Here it is:\n<analysis>\n"
                            + it["analysis"][:20000] + "\n</analysis>\nDo not analyse again. Based on this analysis, give your "
                            "final answer now, exactly in the required format (reason, category, natural)."}]
        if args.logprobs:
            ex = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if it.get("analysis") is not None
                  else extra)
            resp = llm.client.chat.completions.create(model=args.model, messages=msgs, n=1,
                                                      max_tokens=1024 if it.get("analysis") is not None else args.max_tokens,
                                                      temperature=0.0, logprobs=True, top_logprobs=20, **ex)
            outs = [(resp.choices[0].message.content or "").split("</think>")[-1]]
            lp = category_probs(resp.choices[0])
        else:
            outs = [o.split("</think>")[-1] for o in llm.chat(msgs, n=args.samples, **extra)]
        cats = [tag_of(o, "category", CATS) for o in outs]
        votes = collections.Counter(c for c in cats if c)
        o = outs[0]
        return {"id": it["id"], "group": it["group"], "type": rec["type"], "tactic": rec.get("tactic"),
                "detect_rate": rate.get(it["id"]), "original": rec["original"], "sentence": rec["sentence"],
                "category": votes.most_common(1)[0][0] if votes else None,
                "probs": lp or {c: votes[c] / len(outs) for c in CATS}, "probs_from": "logprob" if lp else "votes",
                "unparsed": sum(c is None for c in cats), "finished": it.get("analysis") is not None,
                "natural": tag_of(o, "natural", ("yes", "no")), "output": o}

    map_resumable(items, run, out, concurrency=args.concurrency)
    res = list(kept(out))
    summ = {g: {"n": sum(r["group"] == g for r in res),
                "category": dict(collections.Counter(r["category"] for r in res if r["group"] == g)),
                "natural_no": sum(r["natural"] == "no" for r in res if r["group"] == g)}
            for g in ("rate0", "valid", "rejected")}
    Path(out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
