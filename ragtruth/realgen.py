"""用真实幻觉句把生成器拉到真实错误的分布上（技术报告 Q4 / Q5 / Q6）：真实错句 → 35B 最小修复成正确句 →
「输入正确句、输出真实错句」的 SFT 样本，格式和生成器（ragtruth.synth / gen_dpo）的 prompt、输出 JSON 完全一致。

动机：G0 / G1 / G2 只靠 13 种改法 + 示例造错，判别器 D2 给 G1 产出的平均真实性分数只有 0.028；DPO 只能在生成器
已经会写的东西里挑，起点不在真实分布上就挑不出来。先 SFT 模仿真实模型怎么出错，再 DPO 往「难」推。

步骤：
1. repair：RAGTruth 训练集（train_pool + train_sub）里人工标出的幻觉句（整条回答里没有粗粒度句的），逐句让 35B
   只改标注片段、改成原文支持的说法（conflict 换回原文说法，baseless 删掉多出来的信息），同时写出生成器输出 JSON
   的其余字段（plan、两边的 evidence / reason）和改法归类。校验沿用 synth.check（改动幅度、证据逐字可定位、说明
   与类型一致），复核沿用 synth.verify_one：修复句须判为有依据，真实错句须判为有问题（类型以人工标注为准）。
2. audit：Claude 抽查修复质量（修复句是否完全有依据、是否只改了问题片段、证据和说明是否正确）。
3. build：一条回答里所有幻觉句都修复成功才用；上下文里其他幻觉句换成修复句（生成器 prompt 声明整条回答都正确），
   prompt = synth.build_prompt（统一 prompt，3 个同任务同类型的真实幻觉示例，不带难样本示例，示例排除同一原文），
   completion = 生成器输出 JSON。按原文切 15% 留出（固定种子，排序后再打乱，重跑结果一致）。

用法：
  python -m ragtruth.realgen repair --model judge --out outputs/rt/realgen/repair.jsonl [--limit 10]
  python -m ragtruth.realgen audit --repair outputs/rt/realgen/repair.jsonl --n 30 --out outputs/rt/realgen/audit.jsonl
  python -m ragtruth.realgen build --repair outputs/rt/realgen/repair.jsonl --out_dir outputs/rt/realgen
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth.annotate import render_response, truth_text
from ragtruth.synth import (EVIDENCE_NULL, EVIDENCE_WHAT, TACTICS, TYPE_DEF, build_prompt, build_prompt_free, check,
                            example_pool, free_examples, parse_json_obj, verify_one)

NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
# 修复和复核不开思考（2026-10-28）：开思考每句约 1.9 万 token，修 2,200 句要 6 小时；当前目标是先跑通流程（用户决定）。
# 抽查发现的漏标问题靠 prompt 解决（同一句里其他没依据的内容一起修）

REPAIR_PROMPT = """You are creating training data for a model that learns how assistants make mistakes. Below are a source (the exact input an assistant received) and the assistant's response, split into numbered sentences, followed by human annotations of the problems in the response.

{input}

Human annotations:
{truth}

Sentence S{idx} has this problem:
- {kind}: {type_def}

Repair S{idx} so that it is fully supported by the source, changing as little as possible:
- fix the annotated problem: for a contradiction, replace the wrong part with what the source actually says; for added information, delete the added part;
- annotators sometimes miss problems: check every detail of S{idx} against the source and also fix anything else it states that the source contradicts or does not contain;
- change nothing else: every other word, number, list marker and citation stays exactly as it is; if a deletion leaves the sentence broken, make the smallest fix that restores it;
- the repaired sentence must still read naturally in the response. If S{idx} cannot be repaired this way (for example, the whole sentence is made up), output {{"repairable": false}}.

Then describe the mistake as if someone had introduced it into your repaired sentence:
- "plan": where and how the repaired sentence was changed to produce the original S{idx};
- "tactic": the kind of change, one of: {tactics}; if none fits, a short phrase of your own;
- "evidence": copy, word for word, the part of the source that the original S{idx} {evidence_what}. Copy one contiguous stretch, as short as possible; for a long passage copy its first few words and its last few words joined by " ... " (each part at least 3 words).{evidence_null}
- "reason": one English sentence of at most 30 words: what the source says and what the original S{idx} says, concretely (for example "The source says X, while the sentence says Y" or "The source does not mention Z").
- "original_evidence": the part of the source, word for word, that supports the repaired sentence (null only if it has no factual content).
- "original_reason": one English sentence of at most 30 words describing how the source supports the repaired sentence.
Never use the words hallucination, faithful, conflict, baseless, ground truth.

Output only one JSON object, nothing else:
{{"repaired": "...", "plan": "...", "tactic": "...", "evidence": "...", "reason": "...", "original_evidence": "...", "original_reason": "..."}}"""


def natural_responses(paths: list[str]) -> list[dict]:
    """有人工标出的幻觉句、且没有粗粒度句的回答（粗粒度句的问题片段太大，修不成最小改动）。同一回答只取一次。"""
    seen, out = set(), []
    for p in paths:
        for s in read_jsonl(p):
            if s["bad"] and not s.get("coarse") and s["id"] not in seen:
                seen.add(s["id"])
                out.append(s)
    return out


def items_of(resps: list[dict]) -> list[dict]:
    return [{"id": f"{s['id']}-S{k}", "resp_id": s["id"], "idx": int(k), "type": v}
            for s in resps for k, v in sorted(s["bad"].items(), key=lambda kv: int(kv[0]))]


def select_responses(resps: list[dict], n_target: int, seed: int) -> list[dict]:
    """按（任务类型, 生成回答的模型, 错误类型）分层，各层轮流抽一条含该层幻觉句的回答（整条回答的幻觉句都修，
    上下文才能全换成正确句），直到幻觉句总数达到 n_target。轮流抽保证每一层都有，小层抽完就跳过。
    用户要求（2026-10-28）：SFT 一千多条就够，但每种数据都要有一部分。"""
    rng = random.Random(seed)
    strata = collections.defaultdict(list)
    for s in sorted(resps, key=lambda s: s["id"]):
        for kind in sorted(set(s["bad"].values())):
            strata[(s["task_type"], s["model"], kind)].append(s)
    for k in strata:
        rng.shuffle(strata[k])
    chosen, seen, n = [], set(), 0
    while n < n_target and any(strata.values()):
        for k in sorted(strata):
            while strata[k] and strata[k][-1]["id"] in seen:
                strata[k].pop()
            if not strata[k] or n >= n_target:
                continue
            s = strata[k].pop()
            seen.add(s["id"])
            chosen.append(s)
            n += len(s["bad"])
    return chosen


def repair_prompt(s: dict, idx: int, kind: str) -> str:
    return REPAIR_PROMPT.format(input=render_response(s), truth=truth_text(s), idx=idx, kind=kind,
                                type_def=TYPE_DEF[kind], tactics="; ".join(TACTICS[kind]),
                                evidence_what=EVIDENCE_WHAT[kind], evidence_null=EVIDENCE_NULL[kind])


def repair_one(llm: LLM, s: dict, item: dict, retries: int, rng: random.Random, verify: bool = True) -> dict:
    """返回修复结果；校验 / 复核不过时带 failed（统计原因用）。"""
    idx, kind = item["idx"], item["type"]
    real = s["sentences"][idx - 1]
    msgs = [{"role": "user", "content": repair_prompt(s, idx, kind)}]
    problems = []
    for attempt in range(retries + 1):
        text = llm.chat_one(msgs, temperature=0.3 if attempt == 0 else 0.6, **NO_THINK)
        obj = parse_json_obj(text)
        if obj is not None and obj.get("repairable") is False:
            return {"failed": ["not repairable"]}
        repaired = ((obj or {}).get("repaired") or "").strip()
        # 复用生成器的校验：把修复句当「原句」、真实错句当「改后句」
        seed = {"resp": {**s, "sentences": s["sentences"][:idx - 1] + [repaired] + s["sentences"][idx:]},
                "idx": idx, "type": kind}
        # 长度、改动幅度放宽：方向和造错相反，baseless 的真实错句本来就比删掉多余信息后的修复句长
        res, problems = check({**(obj or {}), "sentence": real} if obj else None, seed, max_edit_frac=1.0,
                              max_len_ratio=3.0)
        if not repaired:
            problems = ["the repaired sentence is empty"]
        if not problems and not verify:   # 只出修复、过规则；复核另起一步（G_fix 和 35B 用同一个复核模型比）
            return {**res, "repaired": repaired, "tactic": (obj.get("tactic") or "").strip(), "attempts": attempt + 1}
        if not problems:
            v = verify_one(llm, seed, real, rng, NO_THINK)
            if v["orig"] != "none" or v["new"] == "none":   # 真实错句的类型以人工标注为准，只要求复核也认为有问题
                return {"failed": [f"verify {v['orig']}/{v['new']}"], "repaired": repaired, "verify": v}
            return {**res, "repaired": repaired, "tactic": (obj.get("tactic") or "").strip(), "verify": v,
                    "attempts": attempt + 1}
        fix = "Fix these problems and output the full JSON object again:\n- " + "\n- ".join(problems)
        if llm.cfg.provider == "claude_cli":   # claude -p 只支持单轮：上一次的回答和要修的问题拼进同一条消息
            msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\nYour previous answer was:\n" + text + "\n\n" + fix}]
        else:
            msgs = msgs[:1] + [{"role": "assistant", "content": text}, {"role": "user", "content": fix}]
    return {"failed": problems}


def cmd_repair(args) -> None:
    resps = {s["id"]: s for s in natural_responses(args.data)}
    pool = list(resps.values())
    if args.fresh:   # 评测 G_fix：从没修过、原文也没进过 G_fix 训练集的回答里另选
        used = {r["resp_id"] for f in args.exclude for r in kept(f)}
        srcs = {r.get("source_id") for f in args.exclude_sources for r in read_jsonl(f)}
        pool = [x for x in pool if x["id"] not in used and x.get("source_id") not in srcs]
    if args.ids:   # 种子生成器（ragtruth.gen_seed select）：只修指定的句子
        want = {r["id"] for r in read_jsonl(args.ids)}
        chosen = [s for s in pool if any(it["id"] in want for it in items_of([s]))]
        items = [it for it in items_of(chosen) if it["id"] in want]
    else:
        chosen = select_responses(pool, args.fresh or args.n_target, args.seed)
        items = items_of(chosen)
    if args.limit:   # 试跑 / 抽查：取选中的前 n 句（选中顺序已随机），之后全量续跑时不重复
        items = items[:args.limit]
    print(f"{len(resps)} responses, chose {len(chosen)}; {len(items)} hallucinated sentences to repair; strata:",
          json.dumps(collections.Counter(f"{s['task_type']}|{s['model']}" for s in chosen), ensure_ascii=False))
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=1024, concurrency=args.concurrency,
                        temperature=0.3, timeout=600, provider=args.provider, effort=args.effort))

    def run(item: dict) -> dict:
        s = resps[item["resp_id"]]
        out = repair_one(llm, s, item, args.retries, random.Random(item["id"]), verify=not args.no_verify)
        return {**item, "source_id": s.get("source_id"), "task_type": s["task_type"], "model": s["model"],
                "real": s["sentences"][item["idx"] - 1], **out}

    map_resumable(items, run, args.out, concurrency=args.concurrency)
    report(kept(args.out))


def report(rows: list[dict]) -> dict:
    ok = [r for r in rows if "failed" not in r]
    fails = collections.Counter(p.split(":")[0][:60] for r in rows for p in r.get("failed", []))
    out = {"n": len(rows), "ok": len(ok), "ok_rate": round(len(ok) / max(1, len(rows)), 3),
           "ok_by_type": dict(collections.Counter(r["type"] for r in ok)),
           "tactics": dict(collections.Counter(r["tactic"] for r in ok).most_common(20)),
           "failures": dict(fails.most_common(12))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------- 改错生成器 G_fix（2026-10-28）----------
# 输入同 35B 修复（REPAIR_PROMPT：原文、整条回答、错误标注、要修的句子），输出同一个 JSON。
# 训练数据：35B 修复成功的真实错句 + 反过来的合成错误（合成错句 → 原句；造错时写的分析正好是修复要写的）。
# 评测：另选没修过的真实错句，G_fix 和 35B 各修一遍（都单次生成、不重试），再用同一个 35B 复核、同一个 critic 检测修复句。

def synthetic_case(pool: dict, rec: dict) -> tuple[dict, dict] | None:
    """合成错误记录 → (带「标注」的回答, 修复目标)。标注只有句号、类型和造错时写的理由（没有人工片段）。"""
    s = pool.get(rec["response_id"])
    if s is None:
        return None
    i = rec["idx"]
    sents = s["sentences"][:i - 1] + [rec["sentence"]] + s["sentences"][i:]
    spans, pos = [], 0
    for t in sents:
        spans.append([pos, pos + len(t)])
        pos += len(t) + 1
    a, b = spans[i - 1]
    resp = {**s, "sentences": sents, "response": " ".join(sents), "sent_spans": spans, "spans": [],
            "span_notes": [[a, b, rec["reason"]]], "bad": {str(i): rec["type"]}, "coarse": []}
    tactic = norm_tactic(rec.get("tactic") or "", rec["type"])
    target = {"repaired": rec["original"], "plan": f"Introduce one {rec['type']} error into S{i}: {tactic}.",
              "tactic": tactic, "evidence": rec["evidence"], "reason": rec["reason"],
              "original_evidence": rec["original_evidence"], "original_reason": rec["original_reason"]}
    return resp, target


def cmd_fixdata(args) -> None:
    resps = {s["id"]: s for s in natural_responses(args.data)}
    pool = {s["id"]: s for s in read_jsonl(args.pool)}
    rows = []
    for r in kept(args.repair):
        if "failed" in r:
            continue
        s = resps[r["resp_id"]]
        tactic = norm_tactic(r["tactic"], r["type"])
        tgt = {"repaired": r["repaired"], "plan": f"Introduce one {r['type']} error into S{r['idx']}: {tactic}.",
               "tactic": tactic, **{k: r[k] for k in ("evidence", "reason", "original_evidence", "original_reason")}}
        rows.append({"id": r["id"], "kind": "real", "source_id": s.get("source_id"),
                     "prompt": repair_prompt(s, r["idx"], r["type"]), "completion": json.dumps(tgt, ensure_ascii=False)})
    n_real = len(rows)
    syn = []
    for gd in args.gen_dirs:
        if (Path(gd) / "valid.jsonl").exists():
            syn += [c["rec"] for x in kept(Path(gd) / "valid.jsonl") for c in x["cands"] if c["valid"]]
    rng = random.Random(args.seed)
    rng.shuffle(syn)
    seen = set()
    for rec in syn:
        key = (rec["response_id"], rec["idx"])
        if key in seen or len(rows) - n_real >= args.max_synthetic:
            continue
        seen.add(key)
        case = synthetic_case(pool, rec)
        if case:
            resp, tgt = case
            rows.append({"id": f"{rec['id']}-inv", "kind": "synthetic", "source_id": resp.get("source_id"),
                         "prompt": repair_prompt(resp, rec["idx"], rec["type"]),
                         "completion": json.dumps(tgt, ensure_ascii=False)})
    rows = [r for r in rows if len(r["prompt"]) + len(r["completion"]) <= args.max_chars]
    rng.shuffle(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft_train.jsonl", rows)
    meta = {"rows": len(rows), "by_kind": dict(collections.Counter(r["kind"] for r in rows))}
    (out / "sft_train.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


def _repair_seed(s: dict, r: dict) -> dict:
    i = r["idx"]
    return {"resp": {**s, "sentences": s["sentences"][:i - 1] + [r["repaired"]] + s["sentences"][i:]}, "idx": i,
            "type": r["type"]}


def cmd_verify(args) -> None:
    """对 repair --no_verify 的结果统一复核：修复句须有依据、真实错句须仍有问题（同 repair_one）。"""
    resps = {s["id"]: s for s in natural_responses(args.data)}
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=512, concurrency=args.concurrency,
                        temperature=0.0, timeout=600))

    def run(r: dict) -> dict:
        if "failed" in r:
            return r
        v = verify_one(llm, _repair_seed(resps[r["resp_id"]], r), r["real"], random.Random(r["id"]), NO_THINK)
        if v["orig"] != "none" or v["new"] == "none":
            return {**r, "failed": [f"verify {v['orig']}/{v['new']}"], "verify": v}
        return {**r, "verify": v}

    map_resumable(list(kept(args.repair)), run, args.out, concurrency=args.concurrency)
    report(kept(args.out))


def cmd_criticcheck(args) -> None:
    """critic 检测修复句（采样 4 次的识破率）：识破率 ≤ 1/4 才算修好（用户提出：改正确了，至少不能被 critic 检出有错）。"""
    from ragtruth.task import messages, parse_output
    resps = {s["id"]: s for s in natural_responses(args.data)}
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=512, concurrency=args.concurrency,
                        temperature=0.7, timeout=600))

    def run(r: dict) -> dict:
        if "failed" in r:
            return {**r, "detect_rate": None}
        seed = _repair_seed(resps[r["resp_id"]], r)
        claim = {"source": seed["resp"]["source"], "sentences": seed["resp"]["sentences"], "idx": r["idx"]}
        outs = llm.chat(messages(claim), n=4, **NO_THINK)
        return {**r, "detect_rate": sum(parse_output(o).verdict == "hallucination" for o in outs) / len(outs)}

    map_resumable(list(kept(args.repair)), run, args.out, concurrency=args.concurrency)
    rows = kept(args.out)
    ok = [r for r in rows if r.get("detect_rate") is not None]
    summ = {"n": len(rows), "rule_and_verify_ok": round(len(ok) / max(1, len(rows)), 3),
            "critic_pass_among_ok": round(sum(r["detect_rate"] <= 0.25 for r in ok) / max(1, len(ok)), 3),
            "all_pass": round(sum(r["detect_rate"] <= 0.25 for r in ok) / max(1, len(rows)), 3)}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


# ---------- Claude 抽查 ----------

AUDIT_PROMPT = """Below are a source (the exact input an assistant received), an assistant's response split into numbered sentences, and human annotations of the problems in it. Sentence S{idx} was annotated as a problem; another model repaired it.

{input}

Human annotations:
{truth}

[original S{idx}] {real}
[repaired S{idx}] {repaired}

The repaired sentence will be used as a correct sentence, and the original as the erroneous version of it, so check:
1. "supported": is everything in the repaired sentence supported by the source (paraphrases and simple inferences count)?
2. "minimal": does the repair change only the annotated problem and keep everything else, so that the original reads like the repaired sentence with one realistic mistake introduced?
3. "error_kept": compared with the repaired sentence, is the original still wrong with respect to the source in the annotated way?
4. "evidence_ok": the analysis below quotes the source for the original sentence; is it the right passage and is the reason accurate?
   evidence: {evidence}
   reason: {reason}

Output only JSON: {{"supported": true, "minimal": true, "error_kept": true, "evidence_ok": true, "note": "one short sentence on any problem"}}"""


def cmd_audit(args) -> None:
    rows = [r for r in kept(args.repair) if "failed" not in r]
    rows = random.Random(args.seed).sample(sorted(rows, key=lambda r: r["id"]), min(args.n, len(rows)))
    resps = {s["id"]: s for s in natural_responses(args.data)}
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))

    def run(r: dict) -> dict:
        s = resps[r["resp_id"]]
        text = llm.chat_one([{"role": "user", "content": AUDIT_PROMPT.format(
            input=render_response(s), truth=truth_text(s), idx=r["idx"], real=r["real"], repaired=r["repaired"],
            evidence=r["evidence"], reason=r["reason"])}])
        return {"id": r["id"], "type": r["type"], "tactic": r["tactic"], "real": r["real"], "repaired": r["repaired"],
                **(parse_json_obj(text) or {"parse_error": text[:300]})}

    map_resumable(rows, run, args.out, concurrency=args.concurrency)
    res = kept(args.out)
    keys = ("supported", "minimal", "error_kept", "evidence_ok")
    summ = {"n": len(res), **{k: round(sum(bool(x.get(k)) for x in res) / max(1, len(res)), 3) for k in keys},
            "all_ok": round(sum(all(x.get(k) for k in keys) for x in res) / max(1, len(res)), 3)}
    print(json.dumps(summ, ensure_ascii=False))
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")


# ---------- SFT 样本 ----------

def norm_tactic(t: str, kind: str) -> str:
    """35B 归类的改法名大小写、冠词、截断不统一（"attribute or property of something" / "negation or polarity"），
    能对上 13 种之一就换成标准名，对不上保留原文。"""
    low = (t or "").strip().lower().rstrip(".")
    strip = lambda x: x[2:] if x.startswith("a ") else x[3:] if x.startswith("an ") else x  # noqa: E731
    for c in TACTICS.get(kind, []):
        if strip(low) and (strip(c).startswith(strip(low)) or strip(low).startswith(strip(c))):
            return c
    return t.strip()


def cmd_build(args) -> None:
    """--free：输入只给修复后的整条回答（自由版 prompt），输出带真实的句号、类型、改法——让生成器学真实错误落在哪。
    --drop_failed（默认开）：一条回答里没修好的幻觉句直接从上下文删掉，其余句子照用（旧做法整条回答丢掉，
    选中的 1,357 条回答里 597 条因此作废）；删句后句号重新编号。"""
    resps = {s["id"]: s for s in natural_responses(args.data)}
    rep = {r["id"]: r for r in kept(args.repair)}
    # --insert_from（自由版 v2，2026-10-30）：修复时判为不可修（整句都是原文没有的内容）的句子，做成「插入一整句」的样本
    ins = ({r["id"] for r in kept(args.insert_from) if r.get("failed") == ["not repairable"]} if args.insert_from else set())
    ins_resp = {i.rsplit("-S", 1)[0] for i in ins}
    by_resp = collections.defaultdict(list)
    for it in items_of([resps[i] for i in sorted({r["resp_id"] for r in rep.values()} | ins_resp) if i in resps]):
        by_resp[it["resp_id"]].append(it)
    ex_pool = example_pool(list(read_jsonl(args.examples)))
    rows, skipped, too_long, dropped = [], 0, 0, 0
    for rid in sorted(by_resp):
        its = by_resp[rid]
        ok = [it for it in its if it["id"] in rep and "failed" not in rep[it["id"]]]
        if not (ok or any(it["id"] in ins for it in its)) or (len(ok) < len(its) and not args.drop_failed):
            skipped += 1
            continue
        s = resps[rid]
        bad = {it["idx"]: it for it in its}
        fixed, new_idx = [], {}
        for i, t in enumerate(s["sentences"], 1):   # 上下文：修好的换成修复句，没修好的删掉，整条回答正确
            if i in bad and bad[i] not in ok:
                dropped += 1
                continue
            fixed.append(rep[bad[i]["id"]]["repaired"] if i in bad else t)
            new_idx[i] = len(fixed)
        resp = {**s, "sentences": fixed}
        for it in ok:
            r = rep[it["id"]]
            rng = random.Random(it["id"])
            idx, tactic = new_idx[it["idx"]], norm_tactic(r["tactic"], it["type"])
            # plan 不用修复时 35B 写的：它常写成修复方向（"把 X 改回原文的 Y"），和生成器该写的造错方向相反
            plan = f"Introduce one {it['type']} error into S{idx}: {tactic}."
            out = {"plan": plan, "sentence": r["real"], "evidence": r["evidence"], "reason": r["reason"],
                   "original_evidence": r["original_evidence"], "original_reason": r["original_reason"]}
            if args.free:
                prompt = build_prompt_free(resp, free_examples(ex_pool, s["task_type"], rng, exclude_source=s["source"]),
                                           v2=args.v2)
                out = {"plan": plan, **({"op": "rewrite"} if args.v2 else {}), "idx": idx, "type": it["type"],
                       "tactic": tactic, **{k: v for k, v in out.items() if k != "plan"}}
            else:
                cands = [c for c in ex_pool.get((s["task_type"], it["type"]), []) if c["source"] != s["source"]]
                seed = {"resp": resp, "idx": idx, "type": it["type"], "tactic": tactic or "any"}
                prompt = build_prompt(seed, rng.sample(cands, min(3, len(cands))))   # 示例不用同一原文
            completion = json.dumps(out, ensure_ascii=False)
            if len(prompt) + len(completion) > args.max_chars:   # 约 4k token 以上：SFT 截断会切掉输出，直接丢
                too_long += 1
                continue
            rows.append({"id": it["id"], "source_id": s.get("source_id"), "task_type": s["task_type"],
                         "model": s["model"], "type": it["type"], "tactic": tactic, "resp_id": rid,
                         "idx": idx, "n_sentences": len(fixed), "repaired": r["repaired"], "real": r["real"], "sentences": fixed,
                         "prompt": prompt, "completion": completion})
        for it in its:   # 插入一整句：上下文里这句已删掉，新句子放回原来的位置
            if it["id"] not in ins or not args.free:
                continue
            rng = random.Random(it["id"])
            idx = sum(1 for i in new_idx if i < it["idx"]) + 1
            a, b = s["sent_spans"][it["idx"] - 1]
            note = next((n[2] for n in s.get("span_notes") or [] if n[0] < b and n[1] > a), "")
            reason = re.sub(r"^[A-Z][A-Z ]+(?=[A-Z][a-z])", "", note).strip()
            if not reason or re.search(r"Original:|Generat|\*", reason):   # 标注备注是「原文 / 生成」对照格式时用通用说法
                reason = "The source does not say anything that supports this added sentence."
            tactic = "a whole added sentence the source does not support"
            out = {"plan": f"Insert one {it['type']} error as a new S{idx}: {tactic}.", "op": "insert", "idx": idx,
                   "type": it["type"], "tactic": tactic, "sentence": s["sentences"][it["idx"] - 1], "evidence": None,
                   "reason": reason, "original_evidence": None, "original_reason": None}
            prompt = build_prompt_free(resp, free_examples(ex_pool, s["task_type"], rng, exclude_source=s["source"]), v2=True)
            completion = json.dumps(out, ensure_ascii=False)
            if len(prompt) + len(completion) > args.max_chars:
                too_long += 1
                continue
            rows.append({"id": it["id"], "source_id": s.get("source_id"), "task_type": s["task_type"], "model": s["model"],
                         "type": it["type"], "tactic": tactic, "resp_id": rid, "idx": idx, "op": "insert",
                         "n_sentences": len(fixed), "repaired": None, "real": s["sentences"][it["idx"] - 1],
                         "sentences": fixed, "prompt": prompt, "completion": completion})
    sources = sorted({r["source_id"] for r in rows})
    random.Random(args.seed).shuffle(sources)
    held = set(sources[:int(len(sources) * args.heldout_frac)])
    train = [r for r in rows if r["source_id"] not in held]
    test = [r for r in rows if r["source_id"] in held]
    if args.n_train:   # 样本效率对照：只用前 n 条人工数据（按 id 排序后固定抽）
        train = random.Random(args.seed).sample(sorted(train, key=lambda r: r["id"]), min(args.n_train, len(train)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"sft_train{args.suffix}.jsonl", train)
    write_jsonl(out / f"sft_heldout{args.suffix}.jsonl", test)
    meta = {"responses_used": len({r["resp_id"] for r in rows}), "responses_skipped": skipped, "sentences_dropped": dropped, "too_long": too_long,
            "free": args.free, "rows": len(rows),
            "train": len(train), "heldout": len(test), "by_type": dict(collections.Counter(r["type"] for r in train)),
            "by_task": dict(collections.Counter(r["task_type"] for r in train)),
            "by_model": dict(collections.Counter(r["model"] for r in train))}
    (out / f"sft_build{args.suffix}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


# ---------- 错误分布和真实分布的距离 ----------

def _dist(rows: list[dict]) -> dict[str, collections.Counter]:
    """位置（句号 / 句数，分 4 段）、类型、改法三个维度的分布。"""
    out = {"position": collections.Counter(), "type": collections.Counter(), "tactic": collections.Counter()}
    for r in rows:
        out["position"][min(3, int(4 * (r["idx"] - 1) / max(1, r["n"])))] += 1
        out["type"][r["type"]] += 1
        out["tactic"][norm_tactic(r.get("tactic") or "", r["type"])] += 1
    return out


def _tv(a: collections.Counter, b: collections.Counter) -> float:
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return round(0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b)), 3)


def cmd_dist(args) -> None:
    """生成器有效产出（gen_dpo valid 的 rec）vs 真实错误（修复成功的幻觉句）：各维度的总变差距离（0 = 一样，1 = 完全不重叠）。
    指定版的位置和改法是随机指定的，这个指标主要看自由版有没有学到真实模型「错在哪、怎么错」。"""
    resps = {s["id"]: s for s in natural_responses(args.data)}
    real = [{"idx": r["idx"], "n": len(resps[r["resp_id"]]["sentences"]), "type": r["type"], "tactic": r["tactic"]}
            for r in kept(args.repair) if "failed" not in r]
    ref = _dist(real)
    rep = {"real": {"n": len(real), **{k: dict(v.most_common(8)) for k, v in ref.items()}}}
    for d in args.dirs:
        gen = [{"idx": c["rec"]["idx"], "n": len(c["rec"]["sentences"]), "type": c["rec"]["type"],
                "tactic": c["rec"].get("tactic")} for r in kept(Path(d) / "valid.jsonl") for c in r["cands"] if c["valid"]]
        g = _dist(gen)
        rep[d] = {"n": len(gen), "tv": {k: _tv(g[k], ref[k]) for k in ref}, **{k: dict(v.most_common(8)) for k, v in g.items()}}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = ["outputs/external/ragtruth/train_pool.jsonl", "outputs/external/ragtruth/train_sub.jsonl"]
    r = sub.add_parser("repair")
    r.add_argument("--data", nargs="+", default=data)
    r.add_argument("--model", default="judge")
    r.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--n_target", type=int, default=2200, help="分层选回答，幻觉句总数到这么多为止")
    r.add_argument("--limit", type=int, default=0, help="只修选中的前 n 句（试跑 / 抽查）")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--fresh", type=int, default=0, help="评测用：从没修过的回答里另选这么多句")
    r.add_argument("--exclude", nargs="*", default=[], help="--fresh：这些修复结果里的回答不选")
    r.add_argument("--exclude_sources", nargs="*", default=[], help="--fresh：这些文件里出现的原文不选")
    r.add_argument("--no_verify", action="store_true", help="只修复 + 规则校验，复核另用 verify")
    r.add_argument("--ids", default="", help="只修这个 jsonl 里列出的句子 id（ragtruth.gen_seed select 的输出）")
    r.add_argument("--provider", default="openai", help="openai（本地 vLLM）| claude_cli（用户 2026-10-30：种子数据全部用 Claude 修）")
    r.add_argument("--effort", default="high")
    f = sub.add_parser("fixdata")
    f.add_argument("--data", nargs="+", default=data)
    f.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
    f.add_argument("--repair", required=True)
    f.add_argument("--gen_dirs", nargs="*", default=[])
    f.add_argument("--max_synthetic", type=int, default=1500)
    f.add_argument("--max_chars", type=int, default=14000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out_dir", required=True)
    for name in ("verify", "criticcheck"):
        v = sub.add_parser(name)
        v.add_argument("--data", nargs="+", default=data)
        v.add_argument("--repair", required=True)
        v.add_argument("--out", required=True)
        v.add_argument("--model", default="judge" if name == "verify" else "critic")
        v.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
        v.add_argument("--concurrency", type=int, default=64)
    a = sub.add_parser("audit")
    a.add_argument("--data", nargs="+", default=data)
    a.add_argument("--repair", required=True)
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--model", default="claude-opus-5-5")
    a.add_argument("--effort", default="high")
    a.add_argument("--concurrency", type=int, default=4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", nargs="+", default=data)
    b.add_argument("--repair", required=True)
    b.add_argument("--examples", default="outputs/rt/annotated/train_sub35.jsonl")
    b.add_argument("--heldout_frac", type=float, default=0.15)
    b.add_argument("--max_chars", type=int, default=14000)
    b.add_argument("--free", action="store_true", help="自由版 prompt：不指定句子、类型、改法，输出里带真实的句号")
    b.add_argument("--v2", action="store_true", help="自由版 v2 prompt：可改写（大小不限）或插入一整句，输出带 op")
    b.add_argument("--insert_from", default="", help="--v2：修复结果文件，其中不可修的整句错误做成插入样本")
    b.add_argument("--no_drop_failed", dest="drop_failed", action="store_false",
                   help="旧做法：回答里有幻觉句没修好就整条丢掉")
    b.add_argument("--n_train", type=int, default=0)
    b.add_argument("--suffix", default="")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out_dir", required=True)
    t = sub.add_parser("dist")
    t.add_argument("--data", nargs="+", default=data)
    t.add_argument("--repair", required=True)
    t.add_argument("--dirs", nargs="+", required=True)
    t.add_argument("--out", default="")
    p = sub.add_parser("report")
    p.add_argument("--repair", required=True)
    args = ap.parse_args()
    {"repair": cmd_repair, "audit": cmd_audit, "build": cmd_build, "dist": cmd_dist, "fixdata": cmd_fixdata,
     "verify": cmd_verify, "criticcheck": cmd_criticcheck,
     "report": lambda a: report(kept(a.repair))}[args.cmd](args)


if __name__ == "__main__":
    main()
