"""错误改写模型第二版格式（2026-11-08，用户）：
  输入：RAG 原文 + 完整回答（编号句子，都正确）+ 要造的错误机制（不指定改哪一句）
  输出：改第几句 + 一句自然语言的改动描述（edit）+ 改写后的句子；不再输出证据 / 理由 / 类型 / plan
  （用户：edits 改成自然描述，如 Replaced "xx" with "yy"；同样的描述给下游 sev，让它看清改了什么。）

动机（同日数据）：生成器声称的改法和 Claude 判的实际机制只有 2–27% 一致，无法按 critic 漏判分布（范围 / 确定性偏移 35%、跨段拼接 11% …）控制生成；
生成器的证据 / 理由只有约三成对，还会把 sev 带偏（不看解释的 sev 更准）。

训练数据：种子集错误对（outputs/rt/core/pairs.jsonl，真实模型写错的句子 + Claude 改对句 + Claude 标的机制），本地构建、不调 Claude；
edit = 改对句 → 真实错句的逐词 diff 按模板写成自然句（describe()）；采样后给 sev 的描述也由 describe(原句, 改写句) 重新生成，
不用生成器自己写的（生成器写的理由会把 sev 带偏，见 2026-11-08 dne 实验）。机制 other 不进训练。

  python -m ragtruth.gen_v2 build            # → outputs/rt/core/seed_v2_{train,holdout}.jsonl
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")

MECH = {   # 同 ragtruth/core_label.py 的机制；说明给生成器看
    "certainty_or_scope_shift": "shift the certainty or scope of a claim (possible -> certain, some -> all, usually -> always, a specific case -> a general rule, or the reverse)",
    "near_synonym_shift": "replace a word or phrase with a near-synonym that changes the meaning a little",
    "wrong_attribution": "attribute a statement, action or property to the wrong person, entity, passage or step",
    "contradicts_source": "state something the source contradicts (reverse a relation, an outcome or a fact)",
    "merged_across_passages": "join details from different parts of the source into one claim the source does not make",
    "unsupported_inference": "draw a conclusion, cause or purpose that the source does not state",
    "number_or_date": "get a number, date, duration, count or order wrong",
    "vague_evaluation_or_addon": "add a vague evaluation or add-on that the source does not support",
    "added_specific_fact": "add a specific fact or detail that the source does not contain",
}

INSTR = """You are creating training data for a model that detects unsupported or contradicted sentences in responses written from a source.
Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences. Every sentence is faithful to the source.

Rewrite exactly ONE sentence so that it contains an error of this kind:
{mech_name}: {mech_desc}

Choose the sentence where a real assistant could most plausibly make this kind of mistake. The rewritten sentence must still read naturally in the response.

<source>
{source}
</source>

<response>
{sentences}
</response>

Output only one JSON object, nothing else:
{{"idx": <number of the rewritten sentence>, "sentence": "<the full rewritten sentence>", "edit": "<one sentence describing exactly what you changed>"}}"""


def prompt(source: str, sentences: list[str], mech: str) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(sentences, 1))
    return INSTR.format(mech_name=mech, mech_desc=MECH[mech], source=source.strip(), sentences=sents)


def edits(a: str, b: str) -> list[dict] | None:
    """改对句 a → 错句 b 的逐词改动。纯插入 / 纯删除带上前一个词（没有就后一个）当锚点，保证 from 非空、可定位。"""
    wa, wb = a.split(), b.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:   # 插入：带锚点
            if i1 > 0:
                i1, j1 = i1 - 1, j1 - 1
            elif i2 < len(wa):
                i2, j2 = i2 + 1, j2 + 1
            else:
                return None
        out.append({"from": " ".join(wa[i1:i2]), "to": " ".join(wb[j1:j2])})
    return out or None


def apply(a: str, ed: list[dict]) -> str | None:
    """按顺序把每处 from 换成 to（在上一处之后找，保证同一个词出现多次时位置对）。"""
    s, pos = " ".join(a.split()), 0
    for e in ed:
        f, t = " ".join(str(e.get("from", "")).split()), " ".join(str(e.get("to", "")).split())
        if not f:
            return None
        k = s.find(f, pos)
        if k < 0:
            return None
        s = s[:k] + t + s[k + len(f):]
        pos = k + len(t)
    return re.sub(r"\s+", " ", s).strip()


def describe(a: str, b: str, max_ops: int = 6) -> str:
    """原句 a → 改写句 b 的改动，写成一句自然的英文（逐词 diff，确定、和实际改动一致）。"""
    wa, wb = a.split(), b.split()
    ops = [op for op in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return "No change."
    if len(ops) > max_ops:
        return "Rewrote most of the sentence."
    parts = []
    for tag, i1, i2, j1, j2 in ops:
        old, new = " ".join(wa[i1:i2]), " ".join(wb[j1:j2])
        if tag == "replace":
            parts.append(f'replaced "{old}" with "{new}"')
        elif tag == "delete":
            parts.append(f'removed "{old}"')
        else:
            parts.append(f'added "{new}" ' + (f'after "{wa[i1 - 1]}"' if i1 > 0 else "at the beginning"))
    txt = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    return txt[0].upper() + txt[1:] + "."


def norm(s: str) -> str:
    return " ".join(s.split())


def parse(text: str, sentences: list[str]) -> tuple[dict | None, str]:
    """采样时用：→ (rec, 原因)。改动必须能从原句推出输出的改写句。"""
    from ragtruth.synth import parse_json_obj
    obj = parse_json_obj(text)
    if not isinstance(obj, dict):
        return None, "unparsable"
    try:
        i = int(obj.get("idx"))
    except (TypeError, ValueError):
        return None, "bad_idx"
    if not 1 <= i <= len(sentences):
        return None, "bad_idx"
    sent, ed = obj.get("sentence"), obj.get("edit")
    if not isinstance(sent, str) or not sent.strip():
        return None, "bad_fields"
    orig = sentences[i - 1]
    if norm(sent) == norm(orig):
        return None, "identical"
    if any(norm(sent) == norm(x) for k, x in enumerate(sentences, 1) if k != i):
        return None, "copied_other_sentence"
    # edit_gen = 生成器自己写的描述（只留档）；edit = 由实际改动生成的描述（给下游 sev）
    return {"idx": i, "edit_gen": ed if isinstance(ed, str) else "", "edit": describe(orig, sent), "sentence": sent.strip(), "original": orig}, "ok"


def cmd_build(a) -> None:
    out, why = collections.defaultdict(list), collections.Counter()
    desc9 = {r["id"]: r["desc"] for r in read_jsonl(D / a.desc) if r.get("ok")} if a.desc else {}
    for p in read_jsonl(D / "pairs.jsonl"):
        m = p["mechanism"]
        if m not in MECH:
            why["mech_other"] += 1
            continue
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        k = int(p["idx"])
        if norm(p["repaired"]) == norm(p["real"]):
            why["no_edit"] += 1
            continue
        ed = desc9.get(p["id"])   # 9B 写的意思层面描述（通过检查的）
        if not ed:
            if a.desc_only:      # 用户：edit 只用 9B 补，不用规则模板 → 没有合格描述的去掉
                why["no_desc"] += 1
                continue
            ed = describe(p["repaired"], p["real"])
        pr = prompt(p["source"], sents, m)
        # 定版（用户 2026-11-08）：先句子后 edit。edit 只当训练里的辅助信号（去掉后原样输出 19 → 62 / 1260、真错 73% → 62%），
        # 推理时不用它，给 sev 的改动清单由 describe() 规则生成。edit 目标 = 9B 基座写的描述（--desc，Claude 核查 75 / 21 / 4）。
        comp = json.dumps({"idx": k, "sentence": p["real"], "edit": ed}, ensure_ascii=False)
        if len(pr) + len(comp) > a.max_chars:
            why["too_long"] += 1
            continue
        base = {"source_id": p["source_id"], "task_type": p["task_type"], "type": p["type"], "severity": p["severity"], "mechanism": m, "idx": k,
                "n_edits": len(edits(p["repaired"], p["real"]) or [])}
        out[p["split"]].append({"id": p["id"], "task": "rewrite", **base, "prompt": pr, "completion": comp})
        if a.with_describe and p["id"] in desc9:   # （已弃用，用户改为规则列改动）第二步描述任务
            out[p["split"]].append({"id": p["id"] + "#desc", "task": "describe", **base, "prompt": desc_prompt(p["repaired"], p["real"]), "completion": ed})
    random.Random("core|seed_v2").shuffle(out["train"])
    for sp in ("train", "holdout"):
        (D / f"seed_v2_{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[sp]), encoding="utf-8")
    tr = out["train"]
    print("SEED_V2", json.dumps({"train": len(tr), "holdout": len(out["holdout"]), "dropped": dict(why),
                                 "mechanism": dict(collections.Counter(r["mechanism"] for r in tr).most_common()),
                                 "n_edits": dict(sorted(collections.Counter(min(r["n_edits"], 6) for r in tr).items()))}, ensure_ascii=False), flush=True)


# ── 意思层面的改动描述（2026-11-08，用户：用 9B 生成自然描述，再让 Claude 抽查）─────────────────────
# 9B = Qwen3.5-9B 基座（不是评测用的 9B critic），只写描述、不做任何筛选。输入原句、改写句和模板算出的逐词改动（防看错），
# 一句话说意思上改了什么，不评价对错、不提原文。自动检查：非空、≤ 40 词、不含对错评价词；不过关的回退到模板描述。
DESC_SYSTEM = """You describe edits. Given an ORIGINAL sentence, an EDITED version of it, the exact word-level changes, and the KIND of edit that was intended, write ONE short English sentence (at most 25 words) that says what the edit changed in meaning, e.g. "Changed who released the training records from a lawyer to the police." or "Replaced the serving suggestion with a claim that the dish is quick and perfect for any night."
When the edit rewrites much of the sentence, summarize it through the intended kind of edit (for example "Shifted the claim from 'should provoke self-examination' to 'is inspiring reforms', making it more certain.").
Cover every change listed in the word-level changes (mention minor ones briefly), not only the main one.
Describe only the change itself. Do not judge whether it is right or wrong, do not mention any source, passage, error, support or accuracy. Output only the sentence."""
BAD_DESC = re.compile(r"\b(error|wrong|incorrect|unsupported|hallucinat\w*|inaccura\w*|incorrect\w*|false|faithful|source|passage|mistake\w*|contradict\w*|evidence|support\w*)\b", re.I)


def desc_user(orig: str, edited: str, mech: str | None) -> str:
    """mech=None：不给机制（采样上「要求的机制」生成器未必做到，给了会让描述替它声称造出了这种错，2026-11-08 抽查 7 条错误多为此）。"""
    u = f"ORIGINAL: {orig}\nEDITED: {edited}\nWord-level changes: {describe(orig, edited)}"
    return u + (f"\nIntended kind of edit: {mech}: {MECH[mech]}" if mech else "")


def desc_ok(d: str) -> bool:
    return bool(d) and len(d.split()) <= 40 and not BAD_DESC.search(d) and "\n" not in d.strip()


def desc_prompt(orig: str, edited: str) -> str:
    """两步生成的第二步：生成器自己写改动描述。不给机制（给了会顺着要求写），给逐词改动（没有差异清单时错误率高 3 倍）。"""
    sysmsg = DESC_SYSTEM.replace(", and the KIND of edit that was intended", "").split("\nWhen the edit rewrites")[0]
    rest = DESC_SYSTEM.split("\nCover every change")[1] if "\nCover every change" in DESC_SYSTEM else ""
    return sysmsg + "\nCover every change" + rest + "\n\n" + desc_user(orig, edited, None)


def cmd_describe(a) -> None:
    """种子集错误对（改对句 → 真实错句）让 9B 基座写意思层面的描述 → outputs/rt/core/<--out>（id, desc, template, ok）。"""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    cl = OpenAI(base_url=a.url, api_key="x", timeout=600, max_retries=3)
    if a.samples:   # 生成器采样（给新版 sev 的输入用）：原句 → 改写句，机制 = 要求的机制
        ps = [{"id": ("2b/" if "_mc" in f else "9b/" if "_9b" in f else "") + x["id"], "mechanism": x["mechanism"], "repaired": x["original"], "real": x["sentence"]}
              for f in a.samples for x in read_jsonl(D / f) if x.get("why") == "ok"]
    else:
        ps = [p for p in read_jsonl(D / "pairs.jsonl") if p["mechanism"] in MECH and norm(p["repaired"]) != norm(p["real"])]

    def one(p):
        msgs = [{"role": "system", "content": DESC_SYSTEM}, {"role": "user", "content": desc_user(p["repaired"], p["real"], None if a.no_mech else p["mechanism"])}]
        tries = []
        for k in range(1 + a.retries):   # 第一次贪心；不合格的换种子、温度 0.7 重写（用户：只用 9B 补，不用规则模板）
            r = cl.chat.completions.create(model=a.model, messages=msgs, max_tokens=100, temperature=0.0 if k == 0 else 0.7, seed=k,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            d = (r.choices[0].message.content or "").strip().strip('"')
            tries.append(d)
            if desc_ok(d):
                break
        return {"id": p["id"], "mechanism": p["mechanism"], "original": p["repaired"], "edited": p["real"], "template": describe(p["repaired"], p["real"]),
                "desc": d, "ok": desc_ok(d), "tries": len(tries)}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(one, ps))
    (D / a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("DESC9B", json.dumps({"n": len(out), "ok": sum(r["ok"] for r in out)}), flush=True)


CHECK_SYSTEM = """You check short descriptions of sentence edits. Each item gives an ORIGINAL sentence, its EDITED version, and a DESCRIPTION of the edit written by another model. Judge the description only against the two sentences:
- "accurate": it correctly says what changed in meaning, with no false or missing key change;
- "partial": broadly right but misses or blurs an important part of the change;
- "wrong": it misdescribes the change (says something changed that did not, or gets the direction wrong).
Also flag "judges": true if the description evaluates correctness (calls it an error, wrong, unsupported, etc.).
Reply with one JSON object only: {"items": [{"id": ..., "verdict": "accurate"|"partial"|"wrong", "judges": true|false, "note": "<short>"}]}, one entry per item, in order."""


def cmd_check(a) -> None:
    """Claude 抽查 9B 描述（用户批准约 5 次调用）：随机 --n 条（通过自动检查的），每次 20 条 → <--desc 同名>_check.jsonl。"""
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    rows = sorted((r for r in read_jsonl(D / a.desc) if r["ok"]), key=lambda r: r["id"])
    pick = random.Random("gen_v2|check").sample(rows, min(a.n, len(rows)))
    batches = [{"id": f"b{k // 20}", "xs": pick[k:k + 20]} for k in range(0, len(pick), 20)]
    assert len(batches) <= a.max_calls
    llm = LLM(LLMConfig(model="claude-opus-5-5", provider="claude_cli", effort="high", concurrency=4))
    raw = D / (Path(a.desc).stem + "_check_raw.jsonl")

    def run(b):
        u = "\n\n".join(f'<item id="{x["id"]}">\nORIGINAL: {x["original"]}\nEDITED: {x["edited"]}\nDESCRIPTION: {x["desc"]}\n</item>' for x in b["xs"])
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": u}]))
        ys = {str(y.get("id")): y for y in (o.get("items") or [])} if isinstance(o, dict) else {}
        return {"id": b["id"], "rows": [{**x, "claude": ys[x["id"]]} for x in b["xs"] if x["id"] in ys] or None}

    map_resumable(batches, run, raw, concurrency=4)
    res = [r for b in kept(raw) for r in (b["rows"] or [])]
    (D / (Path(a.desc).stem + "_check.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res), encoding="utf-8")
    print("DESC_CHECK", json.dumps({"n": len(res), "verdict": dict(collections.Counter(r["claude"].get("verdict") for r in res)),
                                    "judges": sum(bool(r["claude"].get("judges")) for r in res)}), flush=True)


def cmd_claude_desc(a) -> None:
    """9B 描述不准的部分交 Claude 重写（2026-11-08 用户批准约 51 次调用）：自动检查没过的 + 改动 ≥ 3 处的 + 张冠李戴 / 添加事实，
    每次 20 条，规则同 DESC_SYSTEM（输入同 9B：原句、改写句、逐词改动、机制）→ <--out>：在 --desc 基础上替换这些条目（src=claude）。"""
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    rows = list(read_jsonl(D / a.desc))
    for r in rows:
        r.setdefault("src", "9b")
        r["n_edits"] = len(edits(r["original"], r["edited"]) or [])
    ids = {l.strip() for l in open(D / a.ids, encoding="utf-8") if l.strip()} if a.ids else None
    weak = set(a.weak_mechs or ["wrong_attribution", "added_specific_fact"])
    pick = [r for r in rows if (ids is None or r["id"] in ids)
            and (not r["ok"] or r["n_edits"] >= 3 or r["mechanism"] in weak)]
    pick.sort(key=lambda r: r["id"])
    batches = [{"id": f"b{k // 20}", "xs": pick[k:k + 20]} for k in range(0, len(pick), 20)]
    assert len(batches) <= a.max_calls, f"{len(batches)} calls > {a.max_calls}"
    llm = LLM(LLMConfig(model="claude-opus-5-5", provider="claude_cli", effort="high", concurrency=a.concurrency))
    raw = D / (Path(a.out).stem + "_raw.jsonl")
    sysmsg = (DESC_SYSTEM.replace(", and the KIND of edit that was intended", "").split("\nWhen the edit rewrites")[0] + "\nDescribe only the change itself. Do not judge whether it is right or wrong, do not mention any source, passage, error, support or accuracy. Output only the sentence." if a.no_mech else DESC_SYSTEM) + '\nEach item is in its own <item id="..."> block. Reply with one JSON object only: {"items": [{"id": ..., "desc": "..."}]}, one entry per item, in order. Make sure the description covers every change listed in the word-level changes (merge minor ones briefly), not only the main one.'

    def run(b):
        u = "\n\n".join(f'<item id="{x["id"]}">\n{desc_user(x["original"], x["edited"], None if a.no_mech else x["mechanism"])}\n</item>' for x in b["xs"])
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": sysmsg}, {"role": "user", "content": u}]))
        ys = {str(y.get("id")): y.get("desc", "") for y in (o.get("items") or [])} if isinstance(o, dict) else {}
        return {"id": b["id"], "rows": [{"id": x["id"], "desc": str(ys[x["id"]]).strip().strip('"')} for x in b["xs"] if x["id"] in ys] or None}

    map_resumable(batches, run, raw, concurrency=a.concurrency)
    got = {r["id"]: r["desc"] for b in kept(raw) for r in (b["rows"] or [])}
    c = collections.Counter()
    for r in rows:
        if r["id"] in got:
            d = got[r["id"]]
            if desc_ok(d) and len(d.split()) <= 45:
                r.update({"desc": d, "ok": True, "src": "claude"}); c["claude_ok"] += 1
            else:
                c["claude_rejected"] += 1
    (D / a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("CLAUDE_DESC", json.dumps({"picked": len(pick), "returned": len(got), **c, "ok_total": sum(r["ok"] for r in rows),
                                     "src": dict(collections.Counter(r["src"] for r in rows if r["ok"]))}), flush=True)


# 类型只是占位（进 critic 前由 Claude 核实给出）：按种子集里该机制的多数类型
TYPE_OF = {"certainty_or_scope_shift": "conflict", "near_synonym_shift": "conflict", "wrong_attribution": "conflict", "contradicts_source": "conflict",
           "number_or_date": "conflict", "merged_across_passages": "conflict", "unsupported_inference": "baseless",
           "vague_evaluation_or_addon": "baseless", "added_specific_fact": "baseless"}


def cmd_sample(a) -> None:
    """--pool 里每条回答（同一 resp_id 只取一次）× --mechs 里每个机制，各采 --n 条 → --out（字段同 core sample，aid 前缀由下游加）。"""
    import zlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    seen, resps = set(), []
    for r in read_jsonl(D / a.pool):
        if r["resp_id"] in seen:
            continue
        seen.add(r["resp_id"])
        resps.append({**r, "sentences": r["sentences"] if isinstance(r["sentences"], list) else json.loads(r["sentences"])})
    mechs = a.mechs or list(MECH)
    jobs = [(r, m) for r in resps for m in mechs]
    if a.plan:   # 按计划采：每行 {"resp_id", "mechanism"}（扩增 sev 训练数据时按机制加权分配，2026-11-08）
        by = {r["resp_id"]: r for r in resps}
        jobs = [(by[q["resp_id"]], q["mechanism"]) for q in read_jsonl(D / a.plan) if q["resp_id"] in by]

    def one(job):
        r, m = job
        key = f"{r['resp_id']}|{m}"
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, prompt(r["source"], r["sentences"], m)))["input_ids"], max_tokens=512,
                                    temperature=a.temperature, top_p=1.0, n=a.n, seed=zlib.crc32(key.encode()) % 100000, extra_body={"top_k": -1})
        res = []
        recs = [parse(c.text, r["sentences"]) for c in out.choices]
        for rec, why in recs:   # （已弃用）第二步：同一个生成器看差异清单写 edit；现在 edit = describe() 规则列改动
            if rec and a.two_step:
                d = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, desc_prompt(rec["original"], rec["sentence"])))["input_ids"],
                                          max_tokens=120, temperature=0.0)
                rec["edit_gen"] = d.choices[0].text.strip().strip('"').split("\n")[0]
        for j, (rec, why) in enumerate(recs):
            x = {"id": f"{r['resp_id']}#{m}#v{j}", "prompt_id": key, "resp_id": r["resp_id"], "source_id": r["source_id"], "task_type": r["task_type"],
                 "mechanism": m, "why": why}
            if rec:
                ctx = list(r["sentences"]); ctx[rec["idx"] - 1] = rec["sentence"]
                x.update({"idx": rec["idx"], "sentence": rec["sentence"], "original": rec["original"], "edit": rec["edit"], "edit_gen": rec["edit_gen"], "type": TYPE_OF[m],
                          "evidence": "", "reason": "", "source": r["source"], "context": ctx, "sentences_orig": r["sentences"]})
            res.append(x)
        return res

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = [x for xs in ex.map(one, jobs) for x in xs]
    (D / a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print("GENV2_SAMPLE", json.dumps({"pool": a.pool, "responses": len(resps), "mechs": len(mechs), "samples": len(out),
                                      "why": dict(collections.Counter(x["why"] for x in out))}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sample", "describe", "check", "claude_desc"])
    ap.add_argument("--desc", default="", help="build：9B 描述文件（outputs/rt/core 下）；check：要抽查的描述文件")
    ap.add_argument("--max_calls", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3, help="describe：不合格的重写次数")
    ap.add_argument("--with_describe", action="store_true", help="build：另加描述任务（两步生成，已弃用）")
    ap.add_argument("--two_step", action="store_true", help="sample：第二步让生成器写描述（已弃用）")
    ap.add_argument("--desc_only", action="store_true", help="build：只用 --desc 里合格的描述，没有的样本去掉")
    ap.add_argument("--ids", default="", help="claude_desc：只在这些 id 里挑（每行一个，outputs/rt/core 下）")
    ap.add_argument("--no_mech", action="store_true", help="describe / claude_desc：不把机制给写描述的模型")
    ap.add_argument("--weak_mechs", nargs="*", default=[], help="claude_desc：这些机制不论改动多少都重写")
    ap.add_argument("--samples", nargs="*", default=[], help="describe：改为给这些采样文件（outputs/rt/core 下）的合格样本写描述")
    ap.add_argument("--max_chars", type=int, default=14000)
    ap.add_argument("--pool", default="genpool_eval.jsonl")
    ap.add_argument("--mechs", nargs="*", default=[])
    ap.add_argument("--plan", default="", help="sample：按计划文件采（resp_id × mechanism），outputs/rt/core 下")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", default="samples_v2.jsonl")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    {"build": cmd_build, "sample": cmd_sample, "describe": cmd_describe, "check": cmd_check, "claude_desc": cmd_claude_desc}[a.cmd](a)


if __name__ == "__main__":
    main()
