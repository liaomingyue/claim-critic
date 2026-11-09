"""sev 第二版：训练数据全部来自错误改写模型的真实输出 + Claude 判断（2026-11-08，用户：sev 训练集不该从种子集程序造，
要从 2B / 9B 生成器的输出里标；核心是把 sev 的效果提上去）。批准约 75 次 Claude 调用。

和 sev__core 的区别：
  - 去掉种子集程序造的 A–D（类型反转 / 证据换成别处的反例太好认，和生成器真实的解释错误不像）；
  - 标签来自 ragtruth.gen_audit 的统一审计（严重度、类型、证据、理由分开判，类型按种子集约定），
    合成 5 选 1：A 显然错 + 解释全对、B 显然错 + 解释有错、C 细微错 + 解释全对、D 细微错 + 解释有错、E 没错；
  - 系统提示词把类型约定写清楚（同 core_label：范围 / 确定性偏移、改变意思的近义词算 conflict）。

  python -m ragtruth.sev_gen items --n9 1000      # → gen_audit/items_sevnew.jsonl（2B sevpool 里没审过的 + 9B 新抽 1000）
  python -m ragtruth.gen_audit label --group sevnew --max_calls 75
  python -m ragtruth.sev_gen data                 # → outputs/rt/rm/sev__gen/{train,calib,test}.jsonl，按原文 70 / 10 / 20
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.gen_audit import OUT, from_sample, write

D = Path("outputs/rt/core")

SEV2_SYSTEM = """You audit a sentence that was deliberately edited to plant an error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the original version of that sentence, and the editor's claimed error: its type, the source evidence it points to, and its reason.

Judge the edited sentence against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source.
The explanation is right only if all three hold: the claimed type is right (conflict = the source states something incompatible with the sentence, including a changed certainty or scope, a near-synonym that changes the meaning, wrong attribution, number, date or order; baseless = the source says nothing about what the sentence adds), the claimed evidence is the part of the source the error is about, and the reason names the actual erroneous claim and why it is wrong.

Options:
A. clear error, and the explanation is right.
B. clear error, but the explanation is wrong.
C. minor error, and the explanation is right.
D. minor error, but the explanation is wrong.
E. not an error.

Answer with a single letter: A, B, C, D or E."""


def prompt(x: dict) -> list[dict]:
    from ragtruth import audit_synth
    rec = {"source": x["source"], "sentences": x["context"], "idx": x["idx"], "original": x["original"], "sentence": x["sentence"],
           "type": x["type"], "evidence": x["evidence"], "reason": x["reason"]}
    return [{"role": "system", "content": SEV2_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}]


# ── 变体（2026-11-08，用户：把改动明确告诉模型再判，应该容易很多）────────────────────────────
# diff ：现有输入 + 逐词 diff 标出的改动（Change: "旧" → "新"），5 选 1 不变
# dne  ：原文 + 回答 + 原句 / 改写句 + 改动标记，不给生成器的类型 / 证据 / 理由，只判严重程度 3 选 1（A 显然错 / B 细微错 / C 没错）
#        —— 36 条误判（sev__gen ≥ 0.8 但 Claude 判没错）都是生成器写了像样但不成立的理由，sev 跟着理由走
DNE_SYSTEM = """You check whether an edit to one sentence of a response introduced an error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL and the EDITED version of that sentence (the original is faithful to the source), and the exact changes between them.

Judge only the changes, against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source or is stated elsewhere in it.

Options:
A. clear error.
B. minor error.
C. not an error.

Answer with a single letter: A, B or C."""


def changes(a: str, b: str, max_spans: int = 4) -> str:
    """逐词 diff：每处改动一行 "旧" → "新"；删除 / 插入单独写；改动太多（超过 max_spans 处）就说整句重写。"""
    import difflib
    wa, wb = a.split(), b.split()
    ops = [op for op in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return "Changes: none"
    if len(ops) > max_spans:
        return "Changes: most of the sentence was rewritten (compare ORIGINAL and EDITED)."
    out = []
    for tag, i1, i2, j1, j2 in ops:
        old, new = " ".join(wa[i1:i2]), " ".join(wb[j1:j2])
        out.append(f'- deleted: "{old}"' if tag == "delete" else f'- inserted: "{new}"' if tag == "insert" else f'- "{old}" -> "{new}"')
    return "Changes:\n" + "\n".join(out)


def prompt_diff(x: dict) -> list[dict]:
    m = prompt(x)
    m[1]["content"] = m[1]["content"].replace("\nClaimed type:", "\n" + changes(x["original"], x["sentence"]) + "\nClaimed type:", 1)
    return m


def prompt_dne(x: dict) -> list[dict]:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["context"], 1))
    u = (f"<source>\n{x['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
         f"Edited sentence: S{x['idx']}\nORIGINAL: {x['original']}\nEDITED: {x['sentence']}\n{changes(x['original'], x['sentence'])}")
    return [{"role": "system", "content": DNE_SYSTEM}, {"role": "user", "content": u}]


# ── 第三版（2026-11-08，用户）：输入带「要求的机制」+ 改动描述（9B 基座写、不给机制）+ 完整逐词改动；判有没有造成要求的那种错 ──
V3_SYSTEM = """You check whether an edit to one sentence of a response introduced the REQUESTED kind of error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL (faithful) and EDITED versions of that sentence, the requested kind of error, a description of the edit and the exact word-level changes.

Judge the changes against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source or is stated elsewhere in it.

Options:
A. a clear error of the requested kind.
B. a minor error of the requested kind.
C. an error, but not of the requested kind.
D. not an error.

Answer with a single letter: A, B, C or D."""


def prompt_v3(x: dict) -> list[dict]:
    from ragtruth.gen_v2 import MECH, describe
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["context"], 1))
    u = (f"<source>\n{x['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
         f"Edited sentence: S{x['idx']}\nORIGINAL: {x['original']}\nEDITED: {x['sentence']}\n"
         f"Requested kind of error: {x['mechanism']}: {MECH[x['mechanism']]}\n"
         + (f"Description of the edit: {x['desc']}\n" if x.get("desc") else "")   # sev__v3 带 9B 描述；sev__v3r 只给规则改动清单
         + f"Word-level changes: {describe(x['original'], x['sentence'])}")
    return [{"role": "system", "content": V3_SYSTEM}, {"role": "user", "content": u}]


def option_v3(c: dict, req: str = "", relax: frozenset = frozenset()) -> str | None:
    """relax：这些要求的机制只要是真错就算「符合要求」（用户 2026-11-09：直接矛盾是结果不是手法，和其他细类重叠，放宽为真错即可）。"""
    s = c.get("severity")
    if s == "none":
        return "D"
    if s not in ("clear", "minor"):
        return None
    return ("A" if s == "clear" else "B") if (c.get("matches") or req in relax) else "C"


def cmd_data_v3(a) -> None:
    """gen_v2_label（964）+ gen_v2_label_r2（190）的 Claude 标注 → sev__v3，按原文 70 / 10 / 20；描述来自 --desc（9B、不给机制）。"""
    from ragtruth.core import hsplit, write_rm
    desc = {r["id"]: r["desc"] for f in (a.desc.split(",") if a.desc else []) for r in read_jsonl(D / f) if r.get("ok")}
    parts, c = collections.defaultdict(list), collections.Counter()
    relax = frozenset(a.relax.split(",")) if a.relax else frozenset()
    extra = a.extra_train.split(",") if a.extra_train else []   # 扩增样本（训练池原文）：只进训练集，校准 / 测试不变
    for sub in ["gen_v2_label", "gen_v2_label_r2"] + extra:
        it = {x["aid"]: x for x in read_jsonl(D / sub / "items.jsonl")}
        for r in read_jsonl(D / sub / "labels.jsonl"):
            x = it.get(r["aid"])
            o = option_v3(r["claude"], x["mechanism"] if x else "", relax)
            if not x or not o:
                continue
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            x = {**x, "context": ctx, "idx": int(x["idx"]), "desc": desc.get(r["aid"], "")}
            src = x["id"].split("#")[0]
            sp = "train" if sub in extra else ("train", "calib", "test")[hsplit(src, "sev_v3", (70, 80))]
            parts[sp].append({"id": "sev3/" + r["aid"], "prompt": prompt_v3(x), "completion": o, "label": o, "task_type": x["task_type"],
                              "group": "gh-" + src, "origin": f"claude_{x['group']}"})
            c[(sp, o)] += 1
            c["with_desc"] += bool(x["desc"])
    meta = write_rm(a.name or "sev__v3", parts, 4096, {"counts": {f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: v for k, v in sorted(c.items(), key=str)}})
    print("SEV_V3_DATA", json.dumps({k: meta[k] for k in ("train", "calib", "test")}, ensure_ascii=False), "with_desc", c["with_desc"], flush=True)


PROMPTS = {"base": prompt, "diff": prompt_diff, "dne": prompt_dne, "v3": prompt_v3}
NAMES = {"base": "sev__gen", "diff": "sev__gen_diff", "dne": "sev__gen_dne"}


def option(x: dict, c: dict, variant: str = "base") -> str | None:
    if variant == "dne":
        return {"clear": "A", "minor": "B", "none": "C"}.get(c.get("severity"))
    s = c.get("severity")
    if s == "none":
        return "E"
    if s not in ("clear", "minor"):
        return None
    ok = x["type"] == c.get("true_type") and c.get("evidence_ok") is True and c.get("reason_ok") is True
    return {("clear", True): "A", ("clear", False): "B", ("minor", True): "C", ("minor", False): "D"}[(s, ok)]


def cmd_items(a) -> None:
    done = {x["id"] for x in read_jsonl(OUT / "items_pair.jsonl")}
    s2 = [from_sample(x, "2b") for x in read_jsonl(D / "sevpool.jsonl") if x["id"] not in done]
    per, pool9 = collections.Counter(), []
    xs = sorted((x for x in read_jsonl(D / "samples_9b_audit.jsonl") if x.get("why") == "ok" and x["id"] not in done), key=lambda x: x["id"])
    random.Random("sev_gen|9b").shuffle(xs)
    for x in xs:
        if len(pool9) >= a.n9:
            break
        if per[x["prompt_id"]] >= 2:
            continue
        per[x["prompt_id"]] += 1
        pool9.append(from_sample(x, "9b"))
    out = s2 + pool9
    random.Random("sev_gen|mix").shuffle(out)
    write(OUT / "items_sevnew.jsonl", out)
    print("SEVGEN_ITEMS", len(out), dict(collections.Counter(x["group"] for x in out)), flush=True)


def source_of(x: dict) -> str:
    return x.get("source_id") or x["id"].split("#")[0].replace("core-", "")


def cmd_data(a) -> None:
    from ragtruth.core import hsplit, write_rm
    pf, name = PROMPTS[a.variant], NAMES[a.variant]
    rows = []
    for f in ("pair", "sevnew"):
        it = {x["aid"]: x for x in read_jsonl(OUT / f"items_{f}.jsonl")}
        for r in read_jsonl(OUT / f"labels_{f}.jsonl"):
            x = it.get(r["aid"])
            o = option(x, r["claude"], a.variant) if x else None
            if o:
                rows.append((x, o))
    parts, c = collections.defaultdict(list), collections.Counter()
    for x, o in sorted(rows, key=lambda t: t[0]["aid"]):
        src = x["id"].split("#")[0]           # prompt id 形如 core-<source>#b / #c；同一原文的两种类型落同一切分
        sp = ("train", "calib", "test")[hsplit(src, "sev_gen", (70, 80))]
        parts[sp].append({"id": "sev2/" + x["aid"], "prompt": pf(x), "completion": o, "label": o, "task_type": x["task_type"],
                          "group": "gh-" + src, "origin": f"claude_{x['group']}"})
        c[(sp, x["group"], o)] += 1
    meta = write_rm(name, parts, 4096, {"by_split_group_label": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(c.items())}})
    print("SEVGEN_DATA", json.dumps({k: meta[k] for k in ("train", "calib", "test")}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "data", "data_v3"])
    ap.add_argument("--relax", default="", help="data_v3：这些要求的机制真错即算符合（逗号分隔，如 contradicts_source）")
    ap.add_argument("--extra_train", default="", help="data_v3：只进训练集的扩增标注目录（逗号分隔，outputs/rt/core 下）")
    ap.add_argument("--name", default="", help="data_v3：数据集名（默认 sev__v3；只给规则改动清单的叫 sev__v3r）")
    ap.add_argument("--desc", default="", help="data_v3：9B 描述文件（outputs/rt/core 下，id = aid）")
    ap.add_argument("--n9", type=int, default=1000)
    ap.add_argument("--variant", default="base", choices=list(PROMPTS))
    a = ap.parse_args()
    {"items": cmd_items, "data": cmd_data, "data_v3": cmd_data_v3}[a.cmd](a)


if __name__ == "__main__":
    main()
