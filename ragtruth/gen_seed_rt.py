"""真实错误种子（RAG 训练数据里 Claude 修过的那批可以当错误生成器的起始种子样本；critic 也单独用这批训一个，
拿去优化 RAG 生成的 GRPO 看效果）。

来源：S1 的 Claude 修正（outputs/rt/minicpm/s1/fix_raw.jsonl / fix_single_raw.jsonl）—— 831 个 GRPO 训练 prompt 的参考回答
（RAGTruth 原模型写的），Claude 按严格边界逐句改对。每处句级修改 = 一对（改对句 → 原错句）：真实模型在上下文里犯的真错。
整段重写（rewrite）、删句（改成空串，对应生成器的 insert）、改对句拆成多句的不用。
上下文：所有修改都应用后的句子列表（整条回答干净，和生成器 prompt「每句都被原文支持」一致），目标位置换回原错句。

类型 / 改法 / 证据 / 理由 Claude 没给。补法（只用 2B，不调 Claude、不用 9B）：
  对两种类型各做一次：现有生成器（gen-seed5-mc）在同一个生成器 prompt（指定类型）上强制前缀续写——
  先给 plan（说明要把 S<i> 改成这句真错句）续写出 tactic，再把 sentence 固定成真错句续写 evidence / reason / original_*；
  严重程度 RM（sev__mc，6 选 1）判两种说法，取「解释对」（A + C）大的那个类型。同时记 rm1__mcgold_ref P(判出)、RM2 P(自然)，只作分析。
不按任何分数筛（Claude 判的就是错）。

  python -m ragtruth.gen_seed_rt fill --url http://127.0.0.1:8000/v1 --base <2B 目录> --sev_temp T --rm1_temp T --rm2_temp T
  python -m ragtruth.gen_seed_rt build      # → seedrt/sft_train.jsonl（新种子 + 第五版 151 条）、harvest_rt.jsonl（critic_harvest 格式）
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl

G = Path("outputs/rt/gentarget")
D = G / "seedrt"
FIX = Path("outputs/rt/minicpm/s1")
TYPES = ("conflict", "baseless")


def candidates() -> tuple[list[dict], dict]:
    from common.llm import kept
    from ragtruth.convert import split_sentences
    from ragtruth.policy_sft import rows_of
    fixes = {}
    for f in ("fix_raw.jsonl", "fix_single_raw.jsonl"):
        if (FIX / f).exists():
            for b in kept(FIX / f):
                for r in b["rows"]:
                    fixes[r["id"]] = r
    out, why = [], collections.Counter()
    for r in sorted(rows_of(), key=lambda r: str(r["id"])):
        f = fixes.get(r["id"])
        if not f or not r.get("ref_text"):
            continue
        if f.get("rewrite"):
            why["rewrite"] += 1
            continue
        t = r["ref_text"].strip()
        sents = [t[a:b] for a, b in split_sentences(t)]
        new = {int(x["idx"]): (x.get("new") or "").strip() for x in f.get("fixes") or []}
        fixed, pos = [], {}
        for k, s in enumerate(sents, 1):
            if k in new and not new[k]:
                continue   # 删句
            pos[k] = len(fixed) + 1
            fixed.append(new.get(k, s))
        for k in sorted(new):
            if not 1 <= k <= len(sents):
                why["bad_idx"] += 1
            elif not new[k]:
                why["deleted"] += 1
            elif len(split_sentences(new[k])) != 1:
                why["multi_sentence_fix"] += 1
            elif " ".join(new[k].split()) == " ".join(sents[k - 1].split()):
                why["identical"] += 1
            elif len(fixed) < 2:
                why["one_sentence"] += 1
            else:
                out.append({"id": f"rt-{r['id']}-S{k}", "resp_id": str(r["id"]), "source_id": str(r["source_id"]), "task_type": r["task_type"],
                            "source": r["source"], "sentences": fixed, "idx": pos[k], "real": sents[k - 1], "repaired": new[k]})
    return out, dict(why)


def gen_prompt(c: dict, pool: dict, want: str | None) -> str:
    from ragtruth.synth import build_prompt_free, free_examples
    resp = {"source": c["source"], "sentences": c["sentences"], "task_type": c["task_type"]}
    return build_prompt_free(resp, free_examples(pool, c["task_type"], random.Random(c["id"]), exclude_source=c["source"]), focus=True,
                             free_size=True, ctx=True, want_type=want)


def cmd_fill(a) -> None:
    from openai import OpenAI
    from transformers import AutoTokenizer
    from ragtruth import audit_synth, discrim
    from ragtruth.rm_data import RM1_SYSTEM, RM2G_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.sev_rm import SEV_SYSTEM
    from ragtruth.synth import example_pool
    from ragtruth.task import render_input
    from train.gen_grpo import chat_prompt
    from train.gen_reward import NO_THINK, context, parse
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.base)
    pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
    cands, why = candidates()
    print("CANDIDATES", len(cands), why, flush=True)

    def opt(model, system, user, opts, t):
        r = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], opts)
        return probs(lp, t) if lp else None

    def complete(prompt: str, prefix: str, max_tokens: int, stop=None) -> str:
        out = cl.completions.create(model=a.model, prompt=chat_prompt(tok, prompt) + prefix, max_tokens=max_tokens, temperature=0.0, seed=0,
                                    stop=stop, extra_body={"top_k": -1})
        return out.choices[0].text

    def one(c: dict) -> dict:
        res = {**c, "fill": {}}
        for kind in TYPES:
            p = gen_prompt(c, pool, kind)
            real = json.dumps(c["real"], ensure_ascii=False)
            head = '{"plan": "Introduce one ' + kind + ' error into S' + str(c["idx"]) + ' by rewriting it as: ' + real[1:-1] + \
                   '", "idx": ' + str(c["idx"]) + ', "type": "' + kind + '", "tactic": "'
            tactic = complete(p, head, 60, stop=['"']).strip()
            pre = head + json.dumps(tactic, ensure_ascii=False)[1:-1] + '", "sentence": ' + real + ', "evidence": "'
            text = pre + complete(p, pre, 500)
            rec, w = parse(text, {"source": c["source"], "sentences": c["sentences"], "task_type": c["task_type"], "want_type": kind})
            if rec is None:
                res["fill"][kind] = {"why": w}
                continue
            obj = json.loads(text[:text.rindex("}") + 1]) if "}" in text else {}
            ctx = context(rec)
            sev = opt(a.sev, SEV_SYSTEM, audit_synth.render({**rec, "sentences": ctx}), "ABCDEF", a.sev_temp)
            res["fill"][kind] = {"why": "ok", "tactic": tactic, "evidence": rec["evidence"], "reason": rec["reason"],
                                 "original_evidence": str(obj.get("original_evidence") or ""), "original_reason": str(obj.get("original_reason") or ""), "sev": sev}
        ok = {k: v for k, v in res["fill"].items() if v.get("sev")}
        if ok:
            kind = max(sorted(ok), key=lambda k: ok[k]["sev"]["A"] + ok[k]["sev"]["C"])
            ctx = c["sentences"][:c["idx"] - 1] + [c["real"]] + c["sentences"][c["idx"]:]
            p1 = opt(a.rm1, RM1_SYSTEM, render_input({"source": c["source"], "sentences": ctx, "idx": c["idx"]}), "ABC", a.rm1_temp)
            p2 = opt(a.rm2, RM2G_SYSTEM, discrim.render(c["source"], ctx, c["idx"]), "AB", a.rm2_temp)
            res.update({"type": kind, "p_halluc": (p1["B"] + p1["C"]) if p1 else None, "p_natural": p2["A"] if p2 else None})
        return res

    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(one, cands))
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "fill.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("FILL", len(rows), "typed", sum("type" in x for x in rows), collections.Counter(x.get("type") for x in rows),
          collections.Counter(v["why"] for x in rows for v in x["fill"].values()), flush=True)


def cmd_build(a) -> None:
    from ragtruth.synth import FOCUS_TACTICS, example_pool
    from train.gen_reward import OTHER_TACTIC, focus_tactic
    pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
    rows = [x for x in read_jsonl(D / "fill.jsonl") if x.get("type")]
    seeds, hv, why = [], [], collections.Counter()
    for x in rows:
        kind, fl = x["type"], x["fill"][x["type"]]
        ctx = x["sentences"][:x["idx"] - 1] + [x["real"]] + x["sentences"][x["idx"]:]
        s = fl["sev"]
        hv.append({"id": x["id"], "prompt_id": x["id"], "resp_id": x["resp_id"], "source_id": x["source_id"], "task_type": x["task_type"], "why": "ok",
                   "type": kind, "idx": x["idx"], "sentence": x["real"], "original": x["repaired"], "evidence": fl["evidence"], "reason": fl["reason"],
                   "p_correct": 1.0, "orig_p": 0.0, "p_sev": s, "p_sev_clear": s["A"] + s["B"], "p_halluc": x["p_halluc"] if x["p_halluc"] is not None else 0.5,
                   "p_natural": x["p_natural"], "source": x["source"], "context": ctx, "sentences_orig": x["sentences"], "origin": "claude_s1_fix"})
        tactic = focus_tactic(fl["tactic"], kind)
        tactic = (fl["tactic"].strip() or OTHER_TACTIC) if tactic == OTHER_TACTIC else tactic
        out = {"plan": f"Introduce one {kind} error into S{x['idx']}: {tactic}.", "idx": x["idx"], "type": kind, "tactic": tactic, "sentence": x["real"],
               "evidence": fl["evidence"], "reason": fl["reason"], "original_evidence": fl["original_evidence"], "original_reason": fl["original_reason"]}
        want = kind if zlib.crc32(x["id"].encode()) % 2 else None   # 一半指定类型（同第五版种子 --hint 1）
        prompt = gen_prompt(x, pool, want)
        comp = json.dumps(out, ensure_ascii=False)
        if len(prompt) + len(comp) > a.max_chars:
            why["too_long"] += 1
            continue
        seeds.append({"id": x["id"], "round": "rt", "source_id": x["source_id"], "task_type": x["task_type"], "model": "ragtruth_ref", "type": kind,
                      "severity": "clear" if s["A"] + s["B"] >= 0.5 else ("minor" if s["C"] + s["D"] >= 0.5 else "other"), "tactic": tactic,
                      "focus": tactic in FOCUS_TACTICS[kind], "resp_id": x["resp_id"], "idx": x["idx"], "n_sentences": len(x["sentences"]),
                      "repaired": x["repaired"], "real": x["real"], "sentences": x["sentences"], "want_type": want, "prompt": prompt, "completion": comp})
    old = list(read_jsonl(G / "seed2b" / a.old)) if a.old else []
    allr = seeds + old
    random.Random(a.seed).shuffle(allr)
    with open(D / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(D / "harvest_rt.jsonl", "w", encoding="utf-8") as f:
        for r in hv:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bins = ((0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01))
    meta = {"typed": len(rows), "seeds_new": len(seeds), "seeds_old": len(old), "sft_rows": len(allr), "dropped": dict(why),
            "by_type": dict(collections.Counter(s["type"] for s in seeds)), "by_task": dict(collections.Counter(s["task_type"] for s in seeds)),
            "by_severity_sev": dict(collections.Counter(s["severity"] for s in seeds)), "focus_tactic_share": round(sum(s["focus"] for s in seeds) / max(1, len(seeds)), 3),
            "critic_ref_p_bins": {f"{lo}-{hi}": sum(lo <= h["p_halluc"] < hi for h in hv) for lo, hi in bins},
            "natural>0.5": round(sum((h["p_natural"] or 0) > 0.5 for h in hv) / max(1, len(hv)), 3),
            "mean_chars": round(sum(len(s["prompt"]) + len(s["completion"]) for s in seeds) / max(1, len(seeds)))}
    (D / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SEEDRT_BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "fill", "build"])
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--base", default="/root/autodl-tmp/models/MiniCPM5-2B")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--sev", default="sev")
    ap.add_argument("--rm1", default="rm1")
    ap.add_argument("--rm2", default="rm2")
    ap.add_argument("--sev_temp", type=float, default=1.0)
    ap.add_argument("--rm1_temp", type=float, default=1.0)
    ap.add_argument("--rm2_temp", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--old", default="sft_train_v5.jsonl", help="合并的旧种子（seed2b 下）；空 = 只用新种子")
    ap.add_argument("--max_chars", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "cands":
        c, w = candidates()
        print("CANDIDATES", len(c), w, collections.Counter(x["task_type"] for x in c))
        return
    {"fill": cmd_fill, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
