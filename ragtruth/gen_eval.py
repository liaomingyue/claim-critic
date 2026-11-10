"""种子生成器：SFT 数据的 RM1 复核 + 多次采样评测（生成器只要种子级启动，看多次采样能不能造出多样的合格错误）。

  python -m ragtruth.gen_eval rm1check --model rm1-9b      # SFT 行里整条（修复后）回答逐句打 P(有幻觉)，P > 0.5 的行丢掉
  python -m ragtruth.gen_eval sample --model gen --tag gseed  # 评测种子回答（gen_seed select 选的 198 条）每条采样 K 次
  python -m ragtruth.gen_eval score --tag gseed --model rm3-9b             # 规则检查 + RM3 P(合格)
  python -m ragtruth.gen_eval score --tag gseed --model rm1-9b --which rm1  # 再补 RM1 P(有幻觉)
  python -m ragtruth.gen_eval report                          # 各生成器：不同合格错误数@K、合格率、同一回答内的多样性、难度分布

合格 = 规则检查通过且 RM3 P(合格) ≥ 0.5；「不同」按（句号, 类型, 改法归类）去重。概率都用各 RM 报告里的全局温度。
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth import audit_synth
from ragtruth.gen_dpo import NO_THINK, _rec
from ragtruth.realgen import norm_tactic
from ragtruth.rm_data import RM1_SYSTEM, RM3_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.synth import build_prompt_free, check_free, example_pool, free_examples, parse_json_obj
from ragtruth.task import render_input

D = Path("outputs/rt/realgen/gen_eval")
POOL = ["outputs/external/ragtruth/train_pool.jsonl", "outputs/external/ragtruth/train_sub.jsonl"]
T = lambda rm: json.loads(Path(f"outputs/rt/rm/{rm}/eval/report.json").read_text(encoding="utf-8"))["temperature"]


def client(model: str, base_url: str, conc: int, max_tokens: int = 8) -> LLM:
    return LLM(LLMConfig(model=model, base_url=base_url, max_tokens=max_tokens, concurrency=conc, timeout=900))


def option_p(llm: LLM, model: str, system: str, user: str, opts: str, t: float) -> dict | None:
    resp = llm.client.chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                              {"role": "user", "content": user}], max_tokens=8, temperature=0.0,
                                              logprobs=True, top_logprobs=20, **NO_THINK)
    lp = option_logprobs(resp.choices[0], opts)
    return probs(lp, t) if lp else None


def cmd_rm1check(args) -> None:
    t, llm = T("rm1"), client(args.model, args.base_url, args.concurrency)
    srcs = {s["id"]: s["source"] for p in POOL for s in read_jsonl(p)}
    for split in ("train", "heldout"):
        rows = list(read_jsonl(f"outputs/rt/realgen/sft_{split}{args.suffix}.jsonl"))

        def run(r: dict) -> dict:
            ps = []
            for i in range(1, len(r["sentences"]) + 1):
                p = option_p(llm, args.model, RM1_SYSTEM,
                             render_input({"source": srcs[r["resp_id"]], "sentences": r["sentences"], "idx": i}), "ABC", t)
                ps.append(None if p is None else round(p["B"] + p["C"], 4))
            return {"id": r["id"], "p_sentences": ps, "idx": r["idx"]}

        out = Path(f"outputs/rt/realgen/sft_{split}{args.suffix}.rm1.jsonl")
        map_resumable(rows, run, out, concurrency=args.concurrency)
        sc = {r["id"]: r for r in kept(out)}
        # 修复句本身 P > 0.5：整行丢。其他句 P > 0.5：只把那几句从上下文删掉、重新编号，按原来的示例重拼 prompt
        # （第一版整行丢，896 行丢了 450 行：RAGTruth 回答里普遍有漏标的句子，整行丢会砍掉一半覆盖面）
        pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
        resp_of = {s["id"]: s for p in POOL for s in read_jsonl(p)}
        keep, drop = [], collections.Counter()
        for r in rows:
            ps = sc[r["id"]]["p_sentences"]
            insert = r.get("op") == "insert"   # 插入样本：目标句不在上下文里，上下文的句子都要查
            if not insert and (ps[r["idx"] - 1] or 0) > 0.5:
                drop["repaired_sentence"] += 1
                continue
            bad = {k + 1 for k, p in enumerate(ps) if (insert or k + 1 != r["idx"]) and (p or 0) > 0.5}
            if bad:
                sents = [t for k, t in enumerate(r["sentences"], 1) if k not in bad]
                idx = r["idx"] - sum(1 for k in bad if k < r["idx"])
                s = resp_of[r["resp_id"]]
                rng = random.Random(r["id"])
                comp = {**json.loads(r["completion"]), "idx": idx}
                comp["plan"] = (f"Insert one {r['type']} error as a new S{idx}: {r['tactic']}." if insert
                                else f"Introduce one {r['type']} error into S{idx}: {r['tactic']}.")
                r = {**r, "sentences": sents, "idx": idx, "n_sentences": len(sents), "context_dropped": len(bad),
                     "prompt": build_prompt_free({**s, "sentences": sents},
                                                 free_examples(pool, s["task_type"], rng, exclude_source=s["source"]),
                                                 v2="op" in comp),
                     "completion": json.dumps(comp, ensure_ascii=False)}
                drop["context_sentences_removed"] += len(bad)
            keep.append(r)
        with open(f"outputs/rt/realgen/sft_{split}{args.suffix}_f.jsonl", "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, {"rows": len(rows), "kept": len(keep), **drop})


def eval_resps(path: str = "outputs/rt/realgen/seed_eval_resps.jsonl") -> list[dict]:
    want = {r["resp_id"] for r in read_jsonl(path)}
    seen, out = set(), []
    for p in POOL:
        for s in read_jsonl(p):
            if s["id"] in want and s["id"] not in seen:
                seen.add(s["id"])
                out.append(s)
    return sorted(out, key=lambda s: s["id"])


def cmd_sample(args) -> None:
    pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
    llm = client(args.model, args.base_url, args.concurrency, 1024)
    D.mkdir(parents=True, exist_ok=True)

    def run(s: dict) -> dict:   # 同一回答、同一组示例采 K 次（只看生成器自身的多样性，示例不变）
        prompt = build_prompt_free(s, free_examples(pool, s["task_type"], random.Random(s["id"]), exclude_source=s["source"]),
                                   v2=args.v2)
        resp = llm.client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": prompt}], n=args.k,
                                                  max_tokens=1024, temperature=args.temperature, **NO_THINK)
        return {"id": s["id"], "outputs": [c.message.content or "" for c in resp.choices]}

    map_resumable(eval_resps(args.resps), run, D / f"samples_{args.tag}.jsonl", concurrency=args.concurrency)


def score_rm2(args) -> None:
    """给 RM1 那一遍的结果再补 RM2 P(自然)（rm2v2 的全局温度），写 scored3_。"""
    from ragtruth import discrim
    from ragtruth.rm_data import RM2_SYSTEM
    t2, rm2 = T("rm2v2"), client(args.model, args.base_url, args.concurrency)

    def run2(it: dict) -> dict:
        if not it.get("rec") or (it.get("p_correct") or 0) < 0.5:
            return it
        rec, i = it["rec"], it["idx"]
        sents = rec["sentences"][:i - 1] + [rec["sentence"]] + (rec["sentences"][i - 1:] if it.get("op") == "insert"
                                                               else rec["sentences"][i:])
        p2 = option_p(rm2, args.model, RM2_SYSTEM, discrim.render(rec["source"], sents, i), "AB", t2)
        return {**it, "p_natural": p2 and p2["A"]}

    sfx = "_norules" if args.no_rules else "_relaxed" if args.relaxed else ""
    map_resumable(list(kept(D / f"scored2_{args.tag}{sfx}.jsonl")), run2, D / f"scored3_{args.tag}{sfx}.jsonl",
                  concurrency=args.concurrency)


def score_rm1(args) -> None:
    """给 score 第一遍的结果补 RM1 P(有幻觉)。"""
    t1, rm1 = T("rm1"), client(args.model, args.base_url, args.concurrency)

    def run1(it: dict) -> dict:
        if not it.get("rec"):
            return it
        rec, i = it["rec"], it["idx"]
        sents = rec["sentences"][:i - 1] + [rec["sentence"]] + rec["sentences"][i:]
        p1 = option_p(rm1, args.model, RM1_SYSTEM, render_input({"source": rec["source"], "sentences": sents, "idx": i}),
                      "ABC", t1)
        return {**it, "p_halluc": p1 and p1["B"] + p1["C"]}

    sfx = "_norules" if args.no_rules else "_relaxed" if args.relaxed else ""
    map_resumable(list(kept(D / f"scored_{args.tag}{sfx}.jsonl")), run1, D / f"scored2_{args.tag}{sfx}.jsonl",
                  concurrency=args.concurrency)


def cmd_score(args) -> None:
    if args.which == "rm2":
        return score_rm2(args)
    if args.which == "rm1":   # 第二遍：给第一遍的结果补 RM1（一张卡放不下两个 9B 时分两遍起服务；多 LoRA 服务时同一个地址）
        return score_rm1(args)
    resps = {s["id"]: s for s in eval_resps(args.resps)}
    rows = []
    for r in kept(D / f"samples_{args.tag}.jsonl"):
        for j, text in enumerate(r["outputs"]):
            try:
                kw = {"max_edit_frac": 1.0, "max_len_ratio": 3.0} if args.relaxed else {}   # 放宽：同种子 SFT 数据的修复校验
                fs, res, problems = check_free(parse_json_obj(text), resps[r["id"]], **kw)
            except (AttributeError, TypeError, ValueError, KeyError) as e:   # 字段写成列表 / 字典等：算规则不过
                fs, res, problems = None, {}, [f"malformed output: {type(e).__name__}"]
            item = {"id": f"{r['id']}#{j}", "resp_id": r["id"], "j": j, "rule_ok": not problems, "problems": problems[:2]}
            if args.no_rules:   # 不在意是否遵守规则（格式问题后面靠 DPO / GRPO 调），能读出句号和改写句就交给 RM3
                obj, s = parse_json_obj(text), resps[r["id"]]
                try:
                    i = int((obj or {}).get("idx"))
                    sent = str(obj.get("sentence") or "").strip()
                except (TypeError, ValueError, AttributeError):
                    i, sent = 0, ""
                item["rule_ok_strict"], item["rule_ok"] = item["rule_ok"], bool(sent) and 1 <= i <= len(s["sentences"])
                insert = str((obj or {}).get("op") or "").strip().lower() == "insert"
                if insert:   # 插入：idx 可以到句数 + 1
                    item["rule_ok"] = bool(sent) and 1 <= i <= len(s["sentences"]) + 1
                item["op"] = "insert" if insert else "rewrite"
                if item["rule_ok"]:
                    kind = str(obj.get("type") or "").strip().lower()
                    item["rec"] = {"source": s["source"], "sentences": s["sentences"], "idx": i,
                                   "original": "(none: this is a newly inserted sentence)" if insert else s["sentences"][i - 1],
                                   "sentence": sent, "type": kind if kind in ("conflict", "baseless") else kind or "unknown",
                                   "tactic": str(obj.get("tactic") or ""), "evidence": str(obj.get("evidence") or ""),
                                   "reason": str(obj.get("reason") or ""), "task_type": s["task_type"]}
                rows.append(item)
                continue
            if fs and not problems:
                rec = _rec({**fs, "id": item["id"]}, res)
                item.update(rec={k: rec[k] for k in ("source", "sentences", "idx", "original", "sentence", "type", "tactic",
                                                     "evidence", "reason", "task_type")})
            rows.append(item)
    t3, rm3 = T("rm3"), client(args.model, args.base_url, args.concurrency)
    out = D / f"scored_{args.tag}{'_norules' if args.no_rules else '_relaxed' if args.relaxed else ''}.jsonl"
    old = ({r["id"]: r for r in kept(D / f"scored_{args.tag}.jsonl")}
           if (args.relaxed or args.no_rules) and (D / f"scored_{args.tag}.jsonl").exists() else {})

    def run(it: dict) -> dict:
        if not it.get("rec"):
            return it
        rec = it["rec"]
        if old.get(it["id"], {}).get("p_correct") is not None and old[it["id"]]["sentence"] == rec["sentence"]:   # 已打过 RM3 的复用
            o = old[it["id"]]
            return {**it, **{k: o[k] for k in ("idx", "type", "tactic", "original", "sentence", "p_correct")}}
        p3 = option_p(rm3, args.model, RM3_SYSTEM, audit_synth.render(rec), "ABCD", t3)
        return {**it, "idx": rec["idx"], "type": rec["type"], "tactic": norm_tactic(rec.get("tactic") or "", rec["type"]),
                "original": rec["original"], "sentence": rec["sentence"], "p_correct": p3 and p3["A"]}

    map_resumable(rows, run, out, concurrency=args.concurrency)


def cmd_report(args) -> None:
    rep = {}
    for f in sorted(D.glob("scored_*.jsonl")):   # 种子生成器评测不用 RM1，有 scored2_（补了 RM1）就用它，没有就只看 RM3
        tag = f.stem[len("scored_"):]
        if tag.startswith("dpo_"):   # DPO 训练用的采样，不是评测
            continue
        for pre in ("scored3_", "scored2_"):   # 补了 RM2 / RM1 的版本优先
            if (D / f"{pre}{tag}.jsonl").exists():
                f = D / f"{pre}{tag}.jsonl"
                break
        by = collections.defaultdict(list)
        for r in kept(f):
            by[r["resp_id"]].append(r)
        ks = [1, 2, 4, 8, 16]
        distinct = {k: [] for k in ks}
        n_ok = n_all = n_rule = 0
        idx_div, tac_div, sims, hard, ph, pnat = [], [], [], 0, [], []
        tactics, types = collections.Counter(), collections.Counter()
        for rid, rs in by.items():
            rs = sorted(rs, key=lambda r: r["j"])
            q = [r for r in rs if r["rule_ok"] and (r.get("p_correct") or 0) >= 0.5]
            n_all += len(rs)
            n_rule += sum(r["rule_ok"] for r in rs)
            n_ok += len(q)
            for k in ks:
                distinct[k].append(len({(r.get("op", "rewrite"), r["idx"], r["type"], r["tactic"]) for r in q if r["j"] < k}))
            idx_div.append(len({r["idx"] for r in q}))
            tac_div.append(len({r["tactic"] for r in q}))
            ss = [r["sentence"] for r in q]
            for a in range(len(ss)):
                for b in range(a + 1, len(ss)):
                    sims.append(difflib.SequenceMatcher(None, ss[a], ss[b]).ratio())
            for r in q:
                tactics[r["tactic"]] += 1
                types[r["type"]] += 1
                if r.get("p_natural") is not None:
                    pnat.append(r["p_natural"])
                if r.get("p_halluc") is not None:
                    ph.append(r["p_halluc"])
                    hard += r["p_halluc"] < 0.5
        n = len(by)
        ent = -sum(c / sum(tactics.values()) * math.log(c / sum(tactics.values())) for c in tactics.values()) if tactics else 0
        rep[tag] = {"responses": n, "samples": n_all, "rule_ok_rate": round(n_rule / max(1, n_all), 3),
                    "qualified_rate": round(n_ok / max(1, n_all), 3),
                    "distinct_qualified@K": {k: round(sum(v) / max(1, n), 2) for k, v in distinct.items()},
                    "responses_with_any_qualified": round(sum(1 for v in distinct[16] if v) / max(1, n), 3),
                    "distinct_sentences_per_resp": round(sum(idx_div) / max(1, n), 2),
                    "distinct_tactics_per_resp": round(sum(tac_div) / max(1, n), 2),
                    "mean_pairwise_similarity": round(sum(sims) / max(1, len(sims)), 3),
                    "tactic_entropy": round(ent, 3), "tactics_used": len(tactics), "types": dict(types),
                    "rm2_p_natural_mean": round(sum(pnat) / max(1, len(pnat)), 3) if pnat else None,
                    "rm2_natural_share(>0.5)": round(sum(p > 0.5 for p in pnat) / max(1, len(pnat)), 3) if pnat else None,
                    "rm1_p_halluc_mean": round(sum(ph) / max(1, len(ph)), 3), "hard_share(rm1<0.5)": round(hard / max(1, len(ph)), 3),
                    "rm1_p_halluc_deciles": [round(sorted(ph)[int(len(ph) * q / 10)], 3) for q in range(10)] if ph else []}
    (D / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rm1check", "sample", "score", "report"])
    ap.add_argument("--model", default="gen")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--resps", default="outputs/rt/realgen/seed_eval_resps.jsonl", help="sample / score：用哪些回答（resp_id 列表）")
    ap.add_argument("--suffix", default="_seed", help="rm1check：SFT 文件后缀（_seed / _seed2）")
    ap.add_argument("--v2", action="store_true", help="sample：用自由版 v2 prompt（可改写或插入）")
    ap.add_argument("--no_rules", action="store_true", help="score：不做规则检查，能读出句号和改写句就打 RM3，写 scored_<tag>_norules")
    ap.add_argument("--relaxed", action="store_true", help="score：改动上限放宽到和种子 SFT 数据一致，结果写 scored_<tag>_relaxed")
    ap.add_argument("--which", choices=["rm3", "rm1", "rm2"], default="rm3", help="score：先 rm3（服务 = rm3-9b）再 rm1（服务 = rm1-9b）")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    {"rm1check": cmd_rm1check, "sample": cmd_sample, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
