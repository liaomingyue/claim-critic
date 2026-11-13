"""按错误操作清单（ragtruth/error_ops.py）建错误合成模型的训练数据。

和 gen_v2 的区别只在指令：原来给 9 个机制名之一，现在给 20 种操作之一（名字 + 说明）。输入输出格式不变：
给原文和整段回答，模型自己挑一句来改，输出改第几句、改写后的句子、一句改动描述。

样本来源：种子错误对（outputs/rt/core/pairs.jsonl：真实模型写错的句子 + 改对的句子）+ 每条的操作标签
（outputs/rt/analysis/ops_label/pairs_labels.jsonl，Claude 对着原文标）。只留单一操作的样本：
一条错误同时做了两种操作时，拿它教「只做这一种」信号是混的。有前提规则的操作，改对的句子不满足规则的也去掉，
训练时和使用时派任务的条件保持一致。

  python -m ragtruth.gen_ops examples  # 每种操作挑 2 条例句对 → configs/rt/op_examples.json
  python -m ragtruth.gen_ops build     # → outputs/rt/core/seed_ops_{train,holdout}.jsonl（--card：提示词里放整张操作卡片）
  python -m ragtruth.gen_ops sample --model gen --tokenizer <基座> --out samples_ops_mc.jsonl   # 留出回答 × 每种操作各采 1 条（要先起 vLLM）
"""
import argparse
import collections
import json
import random
import difflib
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.error_ops import OPS, PRE, card
from ragtruth.gen_v2 import INSTR, describe, edits, norm, parse

D = Path("outputs/rt/core")
LABELS = Path("outputs/rt/analysis/ops_label/pairs_labels.jsonl")
EXAMPLES = Path("configs/rt/op_examples.json")   # 每种操作的例句对（cmd_examples 从种子错误对里挑，进仓库；被挑中的不再进训练）


def examples() -> dict:
    return json.loads(EXAMPLES.read_text(encoding="utf-8"))


# 指令放在原文和回答之后的版本：提示词中位数四千多字符，指令放在开头时离模型开始写的位置很远
TAIL = """You are creating training data for a model that detects unsupported or contradicted sentences in responses written from a source.
Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence is faithful to the source.

<source>
{source}
</source>

<response>
{sentences}
</response>

Rewrite exactly ONE sentence of the response so that it contains an error of this kind:
{mech_name}: {mech_desc}

Choose the sentence where a real assistant could most plausibly make this kind of mistake. The rewritten sentence must still read naturally in the response.

Output only one JSON object, nothing else:
{{"idx": <number of the rewritten sentence>, "sentence": "<the full rewritten sentence>", "edit": "<one sentence describing exactly what you changed>"}}"""


# 只靠提示词（不微调）时用的两版：strict 把「先按卡片挑能做的句子、只改必须改的词、对着原文确认真的错了」写成步骤，输出里先交代换掉的词；
# plan 再让模型先写一句打算（哪一句、原文怎么说、准备改什么）。
STEPS = """How to do it:
1. Go through the numbered sentences and find those where this edit can be done (see "Where it can be done"). If a sentence lacks what the edit needs, do not pick it.
2. Among them, choose the one where a real assistant could most plausibly make this mistake.
3. Find the exact words that have to change. Change only those words. Every other word of the sentence must stay exactly as it is: do not paraphrase, reorder, shorten or extend the rest.
4. Check against the source that the rewritten sentence is now wrong for the reason given above ("Why the result is an error"), and is not just another way of saying the same thing.
5. The rewritten sentence must still read naturally in the response.

Output only one JSON object, nothing else:
{{%s"idx": <number of the rewritten sentence>, "from": "<the exact words of the original sentence that you replace>", "to": "<what you put in their place>", "sentence": "<the full rewritten sentence>", "edit": "<one sentence describing exactly what you changed>"}}"""
OLD_TAIL = TAIL[TAIL.index("Choose the sentence where"):]
STYLES = {"strict": TAIL.replace(OLD_TAIL, STEPS % ""),
          "plan": TAIL.replace(OLD_TAIL, STEPS % '"plan": "<one or two sentences: which sentence fits this edit, what the source says about it, and which words you will change>", ')}


def prompt(source: str, sentences: list[str], op: str, examples: bool = False, tail: bool = False, cards: dict | None = None, style: str = "") -> str:
    """examples：说明后面附上这种操作的真实例子；tail：指令放在原文和回答之后；cards：放整张操作卡片（例句对由 examples() 给）。"""
    if cards is not None:
        sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(sentences, 1))
        return (STYLES[style] if style else TAIL).format(mech_name=op, mech_desc=card(op, cards[op]), source=source.strip(), sentences=sents)
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(sentences, 1))
    desc = OPS[op]["desc"]
    if examples:
        desc += ".\nExamples of this kind of edit, made in other texts:\n" + "\n".join("- " + e for e in OPS[op]["examples"])
    return (TAIL if tail else INSTR).format(mech_name=op, mech_desc=desc, source=source.strip(), sentences=sents)


def op_of(o: dict) -> str | None:
    """主操作。标注时清单里还没有 add_hedge，这类被标成了 other 并在说明里写了加推测词，按说明归过去。"""
    if o["op"] == "other":
        return "add_hedge" if "hedge" in o["why_other"].lower() and "add" in o["why_other"].lower() else None
    return o["op"] if o["op2"] == "none" else None


def cmd_examples(a) -> None:
    """每种操作从种子错误对（训练部分、单一操作）里挑 2 条改动最小的当例句：改对句和错句只差那一处，例子本身就示范了「只改一处」。"""
    lab = {r["id"]: r["o"] for r in read_jsonl(LABELS)}
    cand = collections.defaultdict(list)
    for p in read_jsonl(D / "pairs.jsonl"):
        op = op_of(lab[p["id"]]) if p["id"] in lab else None
        n = len(p["repaired"].split())
        if op and p["split"] == "train" and 8 <= n <= 40 and norm(p["repaired"]) != norm(p["real"]):
            cand[op].append((difflib.SequenceMatcher(None, p["repaired"].split(), p["real"].split()).ratio(), p["id"], p))
    out = {op: [{"id": i, "faithful": norm(p["repaired"]), "error": norm(p["real"]), "sim": round(v, 2)} for v, i, p in sorted(cand[op], key=lambda t: (-t[0], t[1]))[:2]] for op in OPS}
    EXAMPLES.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for op, xs in out.items():
        print(op, [x["sim"] for x in xs])


def cmd_build(a) -> None:
    lab = {r["id"]: r["o"] for r in read_jsonl(LABELS)}
    cards = examples() if a.card else None
    used = {e["id"] for xs in (cards or {}).values() for e in xs}
    desc = {r["id"]: r["desc"] for r in read_jsonl(D / a.desc) if r.get("ok")} if a.desc else {}
    out, why = collections.defaultdict(list), collections.Counter()
    for p in read_jsonl(D / "pairs.jsonl"):
        if p["id"] not in lab:
            why["no_label"] += 1
            continue
        if p["id"] in used:
            why["used_as_example"] += 1
            continue
        op = op_of(lab[p["id"]])
        if op is None:
            why["two_ops" if lab[p["id"]]["op"] != "other" else "other"] += 1
            continue
        if norm(p["repaired"]) == norm(p["real"]):
            why["no_edit"] += 1
            continue
        if OPS[op]["pre"] and not PRE[OPS[op]["pre"]](p["repaired"], p["source"]):
            why["pre_fail"] += 1
            continue
        if a.min_sim and difflib.SequenceMatcher(None, p["repaired"].split(), p["real"].split()).ratio() < a.min_sim:
            why["rewritten"] += 1   # 改对句和错句差得太远：这种样本教的是整句换说法，不是做一处指定的改动
            continue
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        pr = prompt(p["source"], sents, op, a.examples, a.tail, cards)
        comp = json.dumps({"idx": int(p["idx"]), "sentence": p["real"], "edit": desc.get(p["id"]) or describe(p["repaired"], p["real"])}, ensure_ascii=False)
        if len(pr) + len(comp) > a.max_chars:
            why["too_long"] += 1
            continue
        out[p["split"]].append({"id": p["id"], "task": "rewrite", "source_id": p["source_id"], "task_type": p["task_type"], "severity": p["severity"], "op": op, "layer": OPS[op]["layer"],
                                "mechanism": p["mechanism"], "idx": int(p["idx"]), "n_edits": len(edits(p["repaired"], p["real"]) or []), "prompt": pr, "completion": comp})
    random.Random("core|seed_ops").shuffle(out["train"])
    for sp in ("train", "holdout"):
        (D / f"seed_ops{a.tag}_{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[sp]), encoding="utf-8")
    print("SEED_OPS" + a.tag, json.dumps({"train": len(out["train"]), "holdout": len(out["holdout"]), "dropped": dict(why)}, ensure_ascii=False))
    for sp in ("train", "holdout"):
        c, cl = collections.Counter(r["op"] for r in out[sp]), collections.Counter(r["op"] for r in out[sp] if r["severity"] == "clear")
        print(sp, " ".join(f"{k}:{c[k]}({cl[k]})" for k in OPS))


def cmd_sample(a) -> None:
    """留出原文的回答 × 每种操作采样。有前提规则的操作，只派给至少有一句满足规则的回答；模型挑的那一句满不满足另记在 pre_ok。"""
    import zlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cards = examples() if a.card else None
    seen, resps = set(), []
    for r in read_jsonl(D / a.pool):
        if r["resp_id"] not in seen:
            seen.add(r["resp_id"])
            resps.append({**r, "sentences": r["sentences"] if isinstance(r["sentences"], list) else json.loads(r["sentences"])})

    def ok(op, sent, src):
        return OPS[op]["pre"] is None or PRE[OPS[op]["pre"]](sent, src)

    jobs = [(r, op) for r in resps for op in (a.ops or list(OPS)) if any(ok(op, x, r["source"]) for x in r["sentences"])]

    def one(job):
        r, op = job
        key = f"{r['resp_id']}|{op}"
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, prompt(r["source"], r["sentences"], op, a.examples, a.tail, cards, a.style)))["input_ids"], max_tokens=512,
                                    temperature=a.temperature, top_p=1.0, n=a.n, seed=zlib.crc32(key.encode()) % 100000, extra_body={"top_k": -1})
        res = []
        for j, c in enumerate(out.choices):
            rec, why = parse(c.text, r["sentences"])
            x = {"id": f"{r['resp_id']}#{op}#v{j}", "prompt_id": key, "resp_id": r["resp_id"], "source_id": r["source_id"], "task_type": r["task_type"], "op": op, "layer": OPS[op]["layer"], "why": why}
            if rec:
                ctx = list(r["sentences"])
                ctx[rec["idx"] - 1] = rec["sentence"]
                x.update({"idx": rec["idx"], "sentence": rec["sentence"], "original": rec["original"], "edit": rec["edit"], "edit_gen": rec["edit_gen"],
                          "pre_ok": ok(op, rec["original"], r["source"]), "source": r["source"], "context": ctx, "sentences_orig": r["sentences"]})
            res.append(x)
        return res

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = [x for xs in ex.map(one, jobs) for x in xs]
    (D / a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    good = [x for x in out if x["why"] == "ok"]
    print("GENOPS_SAMPLE", json.dumps({"responses": len(resps), "jobs": len(jobs), "samples": len(out), "why": dict(collections.Counter(x["why"] for x in out)),
                                       "pre_ok": {op: "%d/%d" % (sum(x["pre_ok"] for x in good if x["op"] == op), sum(x["op"] == op for x in good)) for op in OPS if OPS[op]["pre"]}}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sample", "examples"])
    ap.add_argument("--card", action="store_true", help="提示词里放整张操作卡片（做法、在什么句子上能做、为什么算错、例句对），指令在原文和回答之后")
    ap.add_argument("--examples", action="store_true", help="指令里附上这种操作的例子")
    ap.add_argument("--tail", action="store_true", help="指令放在原文和回答之后")
    ap.add_argument("--min_sim", type=float, default=0.0, help="build：只留改对句和错句词重合度不低于这个值的样本")
    ap.add_argument("--tag", default="", help="build：输出文件名后缀（seed_ops<tag>_train.jsonl）")
    ap.add_argument("--style", default="", choices=["", "strict", "plan"], help="sample（配 --card）：只靠提示词时的写法，见 STYLES")
    ap.add_argument("--pool", default="genpool_eval.jsonl")
    ap.add_argument("--ops", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", default="samples_ops_mc.jsonl")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--desc", default="", help="改动描述文件（outputs/rt/core 下，{id, desc, ok}）；没有的用规则模板")
    ap.add_argument("--max_chars", type=int, default=14000)
    a = ap.parse_args()
    {"build": cmd_build, "sample": cmd_sample, "examples": cmd_examples}[a.cmd](a)


if __name__ == "__main__":
    main()
