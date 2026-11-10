"""错误合成模型（生成器）的 DPO：让它更多地写出「有效且难」的错误句（技术报告 Q4）。

迭代实验表明，固定的 35B + 13 种改法在 2 轮后就挖不出新的难样本（难样本率 20% → 7–10% → 5.5%），
瓶颈在生成器。这里用偏好优化直接训练生成器。

流程（每步可续跑，产出都在 --dir 下）：
  sample  ：生成器对每个种子按指定改法采样 k 条改写；--free 时种子是整条回答，句子、类型、改法由生成器自己选
            （ragtruth.synth.FREE_PROMPT），后续各步不变——valid 按生成器报的句号和类型组成种子再校验（单次生成、不重试，看它本来的水平）→ samples.jsonl；
            --hard_shots N 时每条 prompt 另加 N 个难样本示例（被较强 critic 漏检、过了全文核对的合成样本，不同改法）
  valid   ：规则校验（ragtruth.synth.check）→ 35B 复核（verify_one）→ 35B 全文核对（ragtruth.fulltext），三道都过才算有效
  hard    ：目标 critic 对有效改写采样 4 次，算识破率（ragtruth.mine 同口径）
  realism ：真实性判别器（ragtruth.discrim）给有效改写打 p(自然错误)
  pairs   ：--mode hard：同一种子内按 有效且难（识破率 < 0.5）> 有效但易 > 无效 组偏好对；
            --mode hard_real：难度和真实性两维帕累托占优组对 → pairs_<mode>.jsonl（TRL DPO 的 prompt/chosen/rejected）
  report  ：生成器的有效率、难样本率、有效且难的比例、改法分布（DPO 前后各跑一遍对比）
无效样本永远只做 rejected：只奖励「骗过 critic」时，生成器最省事的办法是写一句根本没错的话（HSP 2026 已证实这种奖励投机）。

用法（服务器上，vLLM 依次起生成器 gen / 35B judge / critic）：
  python -m ragtruth.gen_dpo sample --dir outputs/rt/dpo/base --n_seeds 2000 --k 8 --model gen
  python -m ragtruth.gen_dpo valid  --dir outputs/rt/dpo/base --model judge
  python -m ragtruth.gen_dpo hard   --dir outputs/rt/dpo/base --model critic
  python -m ragtruth.gen_dpo pairs  --dir outputs/rt/dpo/base
  python -m ragtruth.gen_dpo report --dir outputs/rt/dpo/base
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth import fulltext
from ragtruth.synth import (build_prompt, build_prompt_free, check, check_free, example_pool, free_examples, parse_json_obj,
                            pick_responses, pick_seeds, to_claims, verify_one, verify_ok)
from ragtruth.task import messages, parse_output

NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
TIER = {"hard": 2, "easy": 1, "invalid": 0}


def llm(model: str, base_url: str, concurrency: int, max_tokens: int = 1024, temperature: float = 1.0) -> LLM:
    return LLM(LLMConfig(model=model, base_url=base_url, max_tokens=max_tokens, concurrency=concurrency,
                         temperature=temperature, timeout=600))


def seeds_for(args) -> list[dict]:
    """种子：不重复合成池 r1–r3 用过的句子；held-out 评测用另一个随机种子再排除训练种子。"""
    used = {(r["response_id"], r["idx"]) for f in args.exclude for r in read_jsonl(f)
            if "response_id" in r and "idx" in r}
    for f in args.exclude_samples:   # 其他 gen_dpo 目录用过的种子（留出评测不和 DPO 训练种子重叠）
        if Path(f).exists():
            # 自由版样本没有指定句号（seed 里只有 free）：跳过。它们的回答和同目录指定版评测用的是同一批，已经排除过
            used |= {(r["resp_id"], r["seed"]["idx"]) for r in kept(f) if "idx" in r["seed"]}
    pool = list(read_jsonl(args.pool))
    return pick_seeds(pool, args.n_seeds, args.seed, used)


def hard_pool(synth: list[str], mined: list[str], threshold: float) -> dict[str, list[dict]]:
    """难样本示例池：过了全文核对的合成记录里，被（较强的）critic 判为难样本的，按类型分组。"""
    rate = {}
    for f in mined:
        for r in kept(f):
            rate[r["id"]] = min(rate.get(r["id"], 1.0), r["detect_rate"])
    pool = collections.defaultdict(list)
    for f in synth:
        for r in read_jsonl(f):
            if rate.get(r["id"], 1.0) < threshold:
                pool[r["type"]].append(r)
    return pool


def pick_hard(pool: dict[str, list[dict]], seed: dict, n: int, rng: random.Random) -> list[dict]:
    """同类型、不同改法（彼此不同，也不同于本条指定的改法）的难样本示例。"""
    cands = [r for r in pool.get(seed["type"], []) if r.get("tactic") != seed["tactic"]]
    rng.shuffle(cands)
    out, seen = [], set()
    for r in cands:
        if r.get("tactic") not in seen:
            out.append(r)
            seen.add(r.get("tactic"))
        if len(out) == n:
            break
    return out


def free_seeds(args) -> list[dict]:
    """自由版种子。--resp_from 时用另一个目录 samples.jsonl 里的回答（和指定版在同一批回答上对比）。"""
    pool = list(read_jsonl(args.pool))
    if args.resp_from:
        want = sorted({r["resp_id"] for r in kept(args.resp_from)})
        by_id = {s["id"]: s for s in pool}
        return [{"id": f"{i}-free", "resp": by_id[i], "free": True} for i in want]
    used = {r["resp_id"] for f in args.exclude_samples if Path(f).exists() for r in kept(f)}
    return pick_responses(pool, args.n_seeds, args.seed, used)


def cmd_sample(args) -> None:
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    seeds = free_seeds(args) if args.free else seeds_for(args)
    ex_pool = example_pool(list(read_jsonl(args.examples)))
    hpool = hard_pool(args.hard_synth, args.hard_mined, args.threshold) if args.hard_shots else {}
    if args.hard_shots:
        print("hard-shot pool:", {k: len(v) for k, v in hpool.items()})
    gen = llm(args.model, args.base_url, args.concurrency, temperature=args.temperature)

    def run_free(seed: dict) -> dict:
        rng = random.Random(seed["id"])
        prompts, outputs, variant = [], [], []
        per = max(1, args.k // args.variants)
        for v in range(args.variants):
            prompt = build_prompt_free(seed["resp"], free_examples(ex_pool, seed["resp"]["task_type"], rng), focus=args.focus)
            outs = gen.chat([{"role": "user", "content": prompt}], n=per, **NO_THINK)
            prompts.append(prompt)
            outputs += outs
            variant += [v] * len(outs)
        return {"prompts": prompts, "outputs": outputs, "variant": variant, "seed": {"free": True},
                "resp_id": seed["resp"]["id"]}

    def run(seed: dict) -> dict:
        if seed.get("free"):
            return run_free(seed)
        # 示例每组都换：k 条采样分成 variants 组，每组重新抽 3 个真实幻觉示例 + 难样本示例；
        # 偏好对只在同一组（同一个 prompt）内配，DPO 要求 chosen / rejected 的 prompt 相同
        rng = random.Random(seed["id"])
        cands = ex_pool.get((seed["resp"]["task_type"], seed["type"])) or []
        prompts, outputs, variant = [], [], []
        per = max(1, args.k // args.variants)
        for v in range(args.variants):
            hard = pick_hard(hpool, seed, args.hard_shots, rng) if args.hard_shots else None
            prompt = build_prompt(seed, rng.sample(cands, min(3, len(cands))), hard)
            outs = gen.chat([{"role": "user", "content": prompt}], n=per, **NO_THINK)
            prompts.append(prompt)
            outputs += outs
            variant += [v] * len(outs)
        return {"prompts": prompts, "outputs": outputs, "variant": variant,
                "seed": {k: seed[k] for k in ("idx", "type", "tactic")}, "resp_id": seed["resp"]["id"]}

    map_resumable(seeds, run, d / "samples.jsonl", concurrency=args.concurrency)
    print(f"sampled {len(kept(d / 'samples.jsonl'))} seeds x {args.k}")


def _seed(row: dict, pool: dict) -> dict:
    return {"id": row["id"], "resp": pool[row["resp_id"]], **row["seed"]}


def _rec(seed: dict, res: dict) -> dict:
    """改写结果 → ragtruth.synth 的合成记录格式（to_claims / fulltext / mine 都用这个格式）。"""
    r = seed["resp"]
    return {"id": seed["id"], "response_id": r["id"], "idx": seed["idx"], "type": seed["type"], "tactic": seed["tactic"],
            "task_type": r["task_type"], "model": r["model"], "source": r["source"], "sentences": r["sentences"],
            "original": r["sentences"][seed["idx"] - 1], **res}


def cmd_valid(args) -> None:
    d = Path(args.dir)
    pool = {r["id"]: r for r in read_jsonl(args.pool)}
    judge = llm(args.model, args.base_url, args.concurrency, max_tokens=512, temperature=0.0)

    def run(row: dict) -> dict:
        seed = _seed(row, pool)
        rng = random.Random(row["id"])
        out = []
        for j, text in enumerate(row["outputs"]):
            if seed.get("free"):   # 句号、类型、改法取生成器自己报的
                fs, res, problems = check_free(parse_json_obj(text), seed["resp"])
                if fs:
                    seed = {**fs, "id": row["id"]}
            else:
                res, problems = check(parse_json_obj(text), seed)
            c = {"j": j, "rule_ok": not problems, "problems": problems[:3]}
            if not problems:
                v = verify_one(judge, seed, res["sentence"], rng, NO_THINK)
                c["verify"] = v
                if verify_ok(v["orig"], v["new"], seed["type"]):
                    ft = fulltext.check_one(judge, _rec(seed, res), NO_THINK)
                    c["fulltext"] = ft
                c["rec"] = _rec(seed, res)
            seed = _seed(row, pool)
            c["valid"] = bool(c.get("fulltext") and c["fulltext"]["supported"] is False)
            out.append(c)
        return {"cands": out}

    map_resumable(list(kept(d / "samples.jsonl")), run, d / "valid.jsonl", concurrency=args.concurrency)
    rows = kept(d / "valid.jsonl")
    n = sum(len(r["cands"]) for r in rows)
    print(f"valid {sum(c['valid'] for r in rows for c in r['cands'])}/{n}; "
          f"rule_ok {sum(c['rule_ok'] for r in rows for c in r['cands'])}/{n}")


def disputed(c: dict) -> bool:
    v = c.get("verify") or {}
    return bool(c.get("rule_ok")) and v.get("orig") == "none" and v.get("new") == "none"


def type_prob(choice) -> float | None:
    """critic 一次输出里，<type> 标签处「conflict / baseless」（有幻觉）相对「none」的概率。critic 的格式是
    evidence → reason → type → verdict，判定在 type 这一步就定了；比 verdict 投票的 0/1 细得多。"""
    import math
    import re
    toks = (choice.logprobs.content if choice.logprobs else None) or []
    norm = lambda x: re.sub(r"^[^a-z]+", "", x.strip().lower())  # noqa: E731
    is_h = lambda x: bool(x) and any(w.startswith(x) or x.startswith(w) for w in ("conflict", "baseless"))  # noqa: E731
    is_n = lambda x: bool(x) and ("none".startswith(x) or x.startswith("none"))  # noqa: E731
    seen = ""
    for t in toks:
        tok = norm(t.token)
        if "<type>" in (seen + t.token).lower() and (is_h(tok) or is_n(tok)):
            ph = sum(math.exp(c.logprob) for c in t.top_logprobs if is_h(norm(c.token)))
            pn = sum(math.exp(c.logprob) for c in t.top_logprobs if is_n(norm(c.token)))
            chosen = math.exp(t.logprob)   # 另一边不在前 20 名时，用 1 − 选中概率估计（模型很自信时也有连续值）
            if ph + pn > 0:
                if ph == 0:
                    ph = max(1e-6, 1 - chosen)
                if pn == 0:
                    pn = max(1e-6, 1 - chosen)
                return ph / (ph + pn)
        seen += t.token
    return None


def cmd_hard(args) -> None:
    d = Path(args.dir)
    critic = llm(args.model, args.base_url, args.concurrency, max_tokens=512, temperature=0.7)
    items = [{"id": f"{r['id']}#{c['j']}", "rec": c["rec"]} for r in kept(d / "valid.jsonl") for c in r["cands"]
             if c["valid"] or (args.include_disputed and disputed(c))]

    def run(it: dict) -> dict:
        bad, _ = to_claims(it["rec"])
        if not args.logprob:
            outs = critic.chat(messages(bad), n=args.n, **NO_THINK)
            return {"detect_rate": sum(parse_output(o).verdict == "hallucination" for o in outs) / len(outs)}
        # 连续难度：识破率只有 n+1 档，72% 的有效样本是 1，DPO 配不出「更难」的对
        resp = critic.client.chat.completions.create(
            model=args.model, messages=messages(bad), n=args.n, max_tokens=512, temperature=0.7, logprobs=True,
            top_logprobs=20, **NO_THINK)
        ps = [type_prob(ch) for ch in resp.choices]
        votes = [parse_output(ch.message.content or "").verdict == "hallucination" for ch in resp.choices]
        ps = [p if p is not None else float(v) for p, v in zip(ps, votes)]
        return {"detect_rate": sum(votes) / len(votes), "p_halluc": sum(ps) / len(ps)}

    out = d / f"hard_{args.tag}.jsonl"
    map_resumable(items, run, out, concurrency=args.concurrency)
    rates = [r["detect_rate"] for r in kept(out)]
    print(f"{args.tag}: {len(rates)} valid scored; hard (detect < 0.5) {sum(x < 0.5 for x in rates)}")


def cmd_arbitrate(args) -> None:
    """「35B 说没错、critic 也没检出」的样本：35B 开思考重新复核（只这一小批，思考慢但量小）。
    判为与原文矛盾 / 原文没有 → 记为错误（隐蔽错误，有效性按 1 算）；仍判为有依据 → 记为没错。"""
    from ragtruth.synth import verify_one as v1
    d = Path(args.dir)
    pool = {r["id"]: r for r in read_jsonl(args.pool)}
    hard = {r["id"]: r for r in kept(d / f"hard_{args.tag}.jsonl")} if (d / f"hard_{args.tag}.jsonl").exists() else {}
    samples = {r["id"]: r for r in kept(d / "samples.jsonl")}   # resp_id 在采样记录里，valid.jsonl 的行没有
    judge = llm(args.model, args.base_url, args.concurrency, max_tokens=8192, temperature=0.6)
    items = []
    for r in kept(d / "valid.jsonl"):
        for c in r["cands"]:
            key = f"{r['id']}#{c['j']}"
            h = hard.get(key)
            if disputed(c) and (args.all or (h and h.get("p_halluc", h["detect_rate"]) < 0.5)):
                items.append({"id": key, "row": r, "c": c})

    def run(it: dict) -> dict:
        c, r = it["c"], it["row"]
        seed = {**_seed(samples[r["id"]], pool), **{k: c["rec"][k] for k in ("idx", "type")}}
        v = v1(judge, seed, c["rec"]["sentence"], random.Random(it["id"]), THINK)
        return {"verify_think": v, "verdict": "error" if v["orig"] == "none" and v["new"] in ("conflict", "baseless")
                else "none"}

    out = d / f"arbitrate_{args.tag}.jsonl"
    map_resumable([{"id": x["id"], "row": x["row"], "c": x["c"]} for x in items], run, out, concurrency=args.concurrency)
    res = kept(out)
    print(f"arbitrated {len(res)} disputed samples (35B none, critic missed): "
          f"{sum(x['verdict'] == 'error' for x in res)} confirmed as subtle errors")


def cmd_realism(args) -> None:
    """真实性判别器（ragtruth.discrim）给有效改写打 p(自然错误)：采样 4 次，判为 natural 的比例。"""
    from ragtruth import discrim
    d = Path(args.dir)
    disc = llm(args.model, args.base_url, args.concurrency, max_tokens=16, temperature=1.0)
    arb = ({x["id"] for x in kept(d / f"arbitrate_{args.arb_tag}.jsonl") if x["verdict"] == "error"}
           if args.arb_tag and (d / f"arbitrate_{args.arb_tag}.jsonl").exists() else set())
    items = []
    for r in kept(d / "valid.jsonl"):
        for c in r["cands"]:
            if c["valid"] or f"{r['id']}#{c['j']}" in arb:
                rec = c["rec"]
                i = rec["idx"]
                items.append({"id": f"{r['id']}#{c['j']}", "source": rec["source"], "idx": i, "unified": args.unified,
                              "sentences": rec["sentences"][:i - 1] + [rec["sentence"]] + rec["sentences"][i:]})

    def run(it: dict) -> dict:   # 标签词 logprob 的连续分数（ragtruth.discrim.logprob_score）
        return discrim.logprob_score(disc.client, args.model, it)

    out = d / f"realism_{args.tag}.jsonl"
    map_resumable(items, run, out, concurrency=args.concurrency)
    ps = [r["p_natural"] for r in kept(out)]
    print(f"realism {args.tag}: {len(ps)} scored; mean p_natural {sum(ps) / max(1, len(ps)):.3f}")


def scored(d: Path, tag: str, rtag: str | None) -> dict[str, list[dict]]:
    """种子 id → [{text, valid, h（1 − 识破率）, r（p 自然，未打分为 None）}, …]。"""
    rate = {r["id"]: r["detect_rate"] for r in kept(d / f"hard_{tag}.jsonl")} if (d / f"hard_{tag}.jsonl").exists() else {}
    real = ({r["id"]: r["p_natural"] for r in kept(d / f"realism_{rtag}.jsonl")}
            if rtag and (d / f"realism_{rtag}.jsonl").exists() else {})
    samples = {r["id"]: r for r in kept(d / "samples.jsonl")}
    out = {}
    for r in kept(d / "valid.jsonl"):
        sm = samples[r["id"]]
        out[r["id"]] = [{"text": sm["outputs"][c["j"]], "valid": c["valid"], "variant": variant_of(sm, c["j"]),
                         "h": 1 - rate.get(f"{r['id']}#{c['j']}", 1.0), "r": real.get(f"{r['id']}#{c['j']}")}
                        for c in r["cands"]]
    return out


def variant_of(sample: dict, j: int) -> int:
    return sample["variant"][j] if "variant" in sample else 0


def prompt_of(sample: dict, v: int) -> str:
    return sample["prompts"][v] if "prompts" in sample else sample["prompt"]


def canonical_prompt(sample: dict) -> str:
    """统一训练 prompt：第 0 组 prompt 去掉难样本示例段，只留真实幻觉示例。
    跨示例组配偏好对时 chosen / rejected 都放在它下面，满足 DPO「同一 prompt」的要求；
    相当于把难样本示例的引导效果蒸馏进参数，DPO 后的生成器评测时也不带难样本示例。"""
    p = prompt_of(sample, 0)
    i, j = p.find("\nEarlier rewrites (on other sources)"), p.find("\nThen analyse both versions")
    return p[:i] + p[j:] if 0 <= i < j else p


def tiers(d: Path, tag: str, threshold: float) -> dict[str, list[tuple[int, str, int]]]:
    """种子 id → [(档次, 原始输出文本, 示例组号), …]。"""
    rate = {r["id"]: r["detect_rate"] for r in kept(d / f"hard_{tag}.jsonl")} if (d / f"hard_{tag}.jsonl").exists() else {}
    samples = {r["id"]: r for r in kept(d / "samples.jsonl")}
    out = {}
    for r in kept(d / "valid.jsonl"):
        lst = []
        for c in r["cands"]:
            key = f"{r['id']}#{c['j']}"
            t = ("invalid" if not c["valid"] else "hard" if rate.get(key, 1.0) < threshold else "easy")
            lst.append((TIER[t], samples[r["id"]]["outputs"][c["j"]], variant_of(samples[r["id"]], c["j"])))
        out[r["id"]] = lst
    return out


def pareto_pairs(cands: list[dict], margin: float) -> list[tuple[dict, dict, str]]:
    """难度 h 和真实性 r 两个维度：chosen 在两维上都不差于 rejected，且两维提升之和 ≥ margin（帕累托占优）。
    取两维之和最高的有效改写作 chosen，配一个被它占优的有效改写；另配一个无效改写（有效 > 无效）。"""
    valid = [c for c in cands if c["valid"] and c["r"] is not None]
    out = []
    if valid:
        best = max(valid, key=lambda c: c["h"] + c["r"])
        dom = [c for c in valid if c is not best and best["h"] >= c["h"] and best["r"] >= c["r"]
               and (best["h"] - c["h"]) + (best["r"] - c["r"]) >= margin]
        if dom:
            out.append((best, min(dom, key=lambda c: c["h"] + c["r"]), "pareto"))
        bad = [c for c in cands if not c["valid"]]
        if bad:
            out.append((best, bad[0], "valid>invalid"))
    return out


def pct_rank(vals: dict[str, float]) -> dict[str, float]:
    """按大小换成 0–1 的百分位：难度和真实性尺度不同（概率 / log-odds），换成百分位后权重才有可比的含义。"""
    order = sorted(vals, key=lambda k: vals[k])
    n = max(1, len(order) - 1)
    out, i = {}, 0
    while i < len(order):   # 并列的取平均名次：分数相同的不能被随意排出先后（否则能否当 chosen 取决于排序噪声）
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            out[k] = (i + j) / 2 / n
        i = j + 1
    return out


def composite(d: Path, args) -> dict[str, dict]:
    """每条采样的有效性 V、难度 h、真实性 r（百分位）和综合分 = V × (a + b·h + c·r)。
    V：规则 / 复核硬失败 0；35B 认为没错（未仲裁或仲裁仍为没错）0.3；有效或仲裁确认是隐蔽错误 1。"""
    hard = {r["id"]: r for r in kept(d / f"hard_{args.tag}.jsonl")}
    real = {r["id"]: r for r in kept(d / f"realism_{args.rtag}.jsonl")}
    arb = {x["id"]: x["verdict"] for x in kept(d / f"arbitrate_{args.tag}.jsonl")} \
        if (d / f"arbitrate_{args.tag}.jsonl").exists() else {}
    hp = pct_rank({k: 1 - v.get("p_halluc", v["detect_rate"]) for k, v in hard.items()})
    rp = pct_rank({k: v.get("score", v["p_natural"]) for k, v in real.items()})
    out = {}
    for r in kept(d / "valid.jsonl"):
        for c in r["cands"]:
            k = f"{r['id']}#{c['j']}"
            if c["valid"] or arb.get(k) == "error":
                V = 1.0
            elif disputed(c):
                V = 0.3
            else:
                V = 0.0
            h, rr = hp.get(k), rp.get(k)
            score = V * (args.w_base + args.w_hard * (h or 0) + args.w_real * (rr or 0)) if V == 1.0 else V * args.w_base
            out[k] = {"sid": r["id"], "j": c["j"], "V": V, "h": h, "r": rr, "score": score,
                      "eligible": V == 1.0 and h is not None and rr is not None and h >= args.min_hard and rr >= args.min_real}
    return out


def cmd_pairs(args) -> None:
    d = Path(args.dir)
    samples = {r["id"]: r for r in kept(d / "samples.jsonl")}
    rng = random.Random(0)
    pairs = []
    if args.mode == "score":
        # 有效、难、真加权成一个分数再配对；chosen 必须同时满足有效、难度和真实性门槛
        sc = composite(d, args)
        by = collections.defaultdict(list)
        for k, x in sc.items():
            by[x["sid"]].append(x)
        kinds = collections.Counter()
        for sid, xs in sorted(by.items()):
            prompt = [{"role": "user", "content": canonical_prompt(samples[sid])}]
            outs = samples[sid]["outputs"]
            cand = sorted(((ch, rj) for ch in xs if ch["eligible"] for rj in xs
                           if ch["score"] - rj["score"] >= args.gap), key=lambda p: p[1]["score"] - p[0]["score"])
            used = set()
            for ch, rj in cand:
                if len(used) >= args.max_pairs or rj["j"] in used:
                    continue
                used.add(rj["j"])
                kinds["rej_V=%s" % rj["V"]] += 1
                pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": outs[ch["j"]]}],
                              "rejected": [{"role": "assistant", "content": outs[rj["j"]]}],
                              "gap": round(ch["score"] - rj["score"], 3)})
        rng.shuffle(pairs)
        out = d / "pairs_score.jsonl"
        write_jsonl(out, pairs)
        print(json.dumps({"mode": "score", "pairs": len(pairs), "rejected_by_validity": dict(kinds),
                          "eligible_chosen": sum(x["eligible"] for x in sc.values()), "samples": len(sc),
                          "mean_gap": round(sum(p["gap"] for p in pairs) / max(1, len(pairs)), 3)}))
        return
    if args.mode == "hard_real":   # 两个打分维度：critic 难度 + 判别器真实性
        for sid, cands in scored(d, args.tag, args.rtag).items():
            groups = ([(canonical_prompt(samples[sid]), cands)] if args.canonical else
                      [(prompt_of(samples[sid], v), [c for c in cands if c["variant"] == v])
                       for v in sorted({c["variant"] for c in cands})])
            for ptext, group in groups:
                prompt = [{"role": "user", "content": ptext}]
                for ch, rj, kind in pareto_pairs(group, args.margin):
                    pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": ch["text"]}],
                                  "rejected": [{"role": "assistant", "content": rj["text"]}], "kind": kind})
        rng.shuffle(pairs)
        out = d / f"pairs_{args.mode}{'_canon' if args.canonical else ''}.jsonl"
        write_jsonl(out, pairs)
        print(json.dumps({"mode": args.mode, "pairs": len(pairs), "by_kind": dict(collections.Counter(p["kind"] for p in pairs))}))
        return
    for sid, lst in tiers(d, args.tag, args.threshold).items():
        groups = [None] if args.canonical else sorted({x[2] for x in lst})
        for v in groups:
            by = collections.defaultdict(list)
            for t, text, vv in lst:
                if v is None or vv == v:
                    by[t].append(text)
            prompt = [{"role": "user", "content": canonical_prompt(samples[sid]) if v is None else prompt_of(samples[sid], v)}]
            # 每组最多两对：难 vs 易（教它写更难的），有效 vs 无效（教它别写假错误）
            if by[2] and by[1]:
                pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": rng.choice(by[2])}],
                              "rejected": [{"role": "assistant", "content": rng.choice(by[1])}], "kind": "hard>easy"})
            good = by[2] or by[1]
            if good and by[0]:
                pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": rng.choice(good)}],
                              "rejected": [{"role": "assistant", "content": rng.choice(by[0])}], "kind": "valid>invalid"})
    rng.shuffle(pairs)
    write_jsonl(d / f"pairs_{args.mode}{'_canon' if args.canonical else ''}.jsonl", pairs)
    print(json.dumps({"mode": args.mode, "canonical": args.canonical, "pairs": len(pairs),
                      "by_kind": dict(collections.Counter(p["kind"] for p in pairs))}))


def cmd_select(args) -> None:
    """给 critic 的训练样本：本轮有效改写里按 难度 + 真实性 之和取前 n 条（每个种子最多 1 条，保持多样），
    输出 ragtruth.synth 的合成记录格式，交给 ragtruth.iterate build 成对加入训练集。"""
    d = Path(args.dir)
    clean = None
    if args.neg_decisions:   # 只用原句经 negcheck 复核为无幻觉的改写，合成对两侧标签都可信
        clean = {x["id"] for x in read_jsonl(args.neg_decisions) if x["src"] == "seed" and x["decision"] == "faithful"}
    ok_orig = lambda rec: clean is None or f"{rec['response_id']}-S{rec['idx']}" in clean  # noqa: E731
    if args.hard_mix:   # 反馈 critic 选样：每个种子最多 1 条；critic 识破率 < threshold 的难样本和易样本各一半，难样本不够时用易样本补
        rng = random.Random(args.seed)
        rate = {r["id"]: r["detect_rate"] for r in kept(d / f"hard_{args.tag}.jsonl")}
        hard, easy = [], []
        for r in sorted(kept(d / "valid.jsonl"), key=lambda r: r["id"]):
            ok = [{**c["rec"], "id": f"{r['id']}#{c['j']}"} for c in r["cands"]
                  if c["valid"] and ok_orig(c["rec"]) and f"{r['id']}#{c['j']}" in rate]
            h = [c for c in ok if rate[c["id"]] < args.threshold]
            if h:
                hard.append(rng.choice(h))
            elif ok:
                easy.append(rng.choice(ok))
        rng.shuffle(hard), rng.shuffle(easy)
        h = hard[:args.n // 2]
        chosen = h + easy[:args.n - len(h)]
        rng.shuffle(chosen)
        for c in chosen:
            c["detect_rate"] = rate[c["id"]]
        write_jsonl(args.out, chosen)
        print(json.dumps({"selected": len(chosen), "hard": len(h), "hard_available": len(hard), "easy_available": len(easy),
                          "types": dict(collections.Counter(c["type"] for c in chosen))}))
        return
    if args.random:   # 闭环验证：每个种子随机取一条有效改写，不按难度 / 真实性挑——比较的只是生成器本身的分布
        rng = random.Random(args.seed)
        picks = []
        arb = ({x["id"] for x in kept(d / f"arbitrate_{args.arb_tag}.jsonl") if x["verdict"] == "error"}
               if args.arb_tag else set())
        for r in sorted(kept(d / "valid.jsonl"), key=lambda r: r["id"]):
            ok = [c for c in r["cands"] if (c["valid"] or f"{r['id']}#{c['j']}" in arb) and ok_orig(c["rec"])]
            if ok:
                c = rng.choice(ok)
                picks.append({**c["rec"], "id": f"{r['id']}#{c['j']}"})
        rng.shuffle(picks)
        chosen = picks[:args.n]
        write_jsonl(args.out, chosen)
        print(json.dumps({"selected": len(chosen), "seeds_with_valid": len(picks),
                          "types": dict(collections.Counter(c["type"] for c in chosen))}))
        return
    rate = {r["id"]: r["detect_rate"] for r in kept(d / f"hard_{args.tag}.jsonl")}
    real = {r["id"]: r["p_natural"] for r in kept(d / f"realism_{args.rtag}.jsonl")}
    if args.pct:   # 和 DPO v2 同口径：连续难度 / 真实性先换成百分位，各占一半（难度和真实性同样重要）
        hp = pct_rank({r["id"]: 1 - r.get("p_halluc", r["detect_rate"]) for r in kept(d / f"hard_{args.tag}.jsonl")})
        rp = pct_rank({r["id"]: r.get("score", r["p_natural"]) for r in kept(d / f"realism_{args.rtag}.jsonl")})
        rate = {k: 1 - v for k, v in hp.items() if k in rp}
        real = {k: rp[k] for k in rate}
    best = {}
    for r in kept(d / "valid.jsonl"):
        for c in r["cands"]:
            key = f"{r['id']}#{c['j']}"
            if c["valid"] and key in rate and key in real:
                sc = (1 - rate[key]) + real[key]
                if sc > best.get(r["id"], (-1, None))[0]:
                    best[r["id"]] = (sc, {**c["rec"], "id": key, "score_h": 1 - rate[key], "score_r": real[key]})
    chosen = [rec for _, rec in sorted(best.values(), key=lambda x: -x[0])[:args.n]]
    write_jsonl(args.out, chosen)
    print(json.dumps({"selected": len(chosen), "mean_h": round(sum(c["score_h"] for c in chosen) / max(1, len(chosen)), 3),
                      "mean_r": round(sum(c["score_r"] for c in chosen) / max(1, len(chosen)), 3),
                      "types": dict(collections.Counter(c["type"] for c in chosen))}))


def cmd_report(args) -> None:
    d = Path(args.dir)
    tr = tiers(d, args.tag, args.threshold)
    n = sum(len(v) for v in tr.values())
    cnt = collections.Counter(t for v in tr.values() for t, *_ in v)
    rows = kept(d / "valid.jsonl")
    fails = collections.Counter()
    for r in rows:
        for c in r["cands"]:
            if not c["rule_ok"]:
                fails["rule"] += 1
            elif "fulltext" not in c:
                fails["verify"] += 1
            elif c["fulltext"]["supported"] is not False:
                fails["fulltext"] += 1
    rep = {"dir": str(d), "critic": args.tag, "candidates": n,
           "valid_rate": round((cnt[2] + cnt[1]) / n, 3), "hard_valid_rate": round(cnt[2] / n, 3),
           "hard_among_valid": round(cnt[2] / max(1, cnt[2] + cnt[1]), 3), "invalid_by_stage": dict(fails),
           "seeds_with_any_hard_valid": sum(any(t == 2 for t, *_ in v) for v in tr.values()), "seeds": len(tr)}
    (d / f"report_{args.tag}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "valid", "hard", "arbitrate", "realism", "pairs", "select", "report"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
        p.add_argument("--model", default="gen")
        p.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
        p.add_argument("--concurrency", type=int, default=64)
        p.add_argument("--tag", default="target", help="hard 的输出名：用哪个 critic 打的分（target / heldout）")
        if name == "hard":
            p.add_argument("--n", type=int, default=4)
            p.add_argument("--logprob", action="store_true", help="另记 critic 在 <type> 处判为有幻觉的概率（连续难度）")
            p.add_argument("--include_disputed", action="store_true", help="仲裁后确认是错误的样本也打难度")
        p.add_argument("--threshold", type=float, default=0.5)
        if name == "sample":
            p.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
            p.add_argument("--examples", default="outputs/rt/annotated/train_sub35.jsonl")
            p.add_argument("--exclude", nargs="*", default=["outputs/rt/synth/r1.raw.jsonl", "outputs/rt/synth/r2.raw.jsonl",
                                                             "outputs/rt/synth/r3.raw.jsonl"])
            p.add_argument("--exclude_samples", nargs="*", default=[], help="其他 gen_dpo 目录的 samples.jsonl")
            p.add_argument("--n_seeds", type=int, default=2000)
            p.add_argument("--free", action="store_true", help="自由版：种子是整条回答，句子、类型、改法由生成器自己选")
            p.add_argument("--resp_from", default="", help="自由版：用这个 samples.jsonl 里的回答作种子（和指定版对比）")
            p.add_argument("--focus", action="store_true", help="自由版：改法清单换成定向版（synth.FOCUS_TACTICS）")
            p.add_argument("--k", type=int, default=8)
            p.add_argument("--variants", type=int, default=4, help="k 条采样分几组，每组重新抽示例（同组内才配偏好对）")
            p.add_argument("--seed", type=int, default=10)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--hard_shots", type=int, default=0, help="每条 prompt 加几个难样本示例；0 = 不加（原 prompt）")
            p.add_argument("--hard_synth", nargs="*", default=["outputs/rt/synth/r1_ft.jsonl", "outputs/rt/synth/r2_ft.jsonl",
                                                                "outputs/rt/synth/r3_ft.jsonl"])
            p.add_argument("--hard_mined", nargs="*", default=["outputs/rt/iter/random_r2_mined.jsonl",
                                                                "outputs/rt/iter/random_r3_mined.jsonl",
                                                                "outputs/rt/iter/critic_r2_mined.jsonl",
                                                                "outputs/rt/iter/critic_r3_mined.jsonl"],
                           help="较强 critic（迭代第 1、2 轮训出的模型）对 r2 / r3 池的识破率")
        if name == "pairs":
            p.add_argument("--mode", choices=["hard", "hard_real", "score"], default="hard",
                           help="hard：只按 critic 难度分档；hard_real：难度 + 真实性两维帕累托；score：综合分（推荐）")
            p.add_argument("--w_base", type=float, default=0.2, help="score：有效的基础分")
            p.add_argument("--w_hard", type=float, default=0.4, help="score：难度权重（和真实性同样重要）")
            p.add_argument("--w_real", type=float, default=0.4, help="score：真实性权重")
            p.add_argument("--min_hard", type=float, default=0.5, help="score：chosen 的难度百分位下限")
            p.add_argument("--min_real", type=float, default=0.5, help="score：chosen 的真实性百分位下限")
            p.add_argument("--gap", type=float, default=0.2, help="score：chosen 与 rejected 综合分之差下限")
            p.add_argument("--max_pairs", type=int, default=3, help="score：每个 prompt 最多几对")
            p.add_argument("--rtag", default="discrim", help="realism 打分的输出名")
            p.add_argument("--margin", type=float, default=0.5)
            p.add_argument("--canonical", action="store_true",
                           help="跨示例组配对，chosen / rejected 都放在统一 prompt（去掉难样本示例）下")
        if name == "realism":
            p.add_argument("--arb_tag", default="", help="仲裁结果的 tag：确认是隐蔽错误的样本也打真实性")
            p.add_argument("--unified", action="store_true", help="判别器是统一 D（中性 prompt，正确句错句一起判）")
        if name == "arbitrate":
            p.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
            p.add_argument("--all", action="store_true", help="仲裁全部争议样本（默认只仲裁 critic 也没检出的）")
        if name == "select":
            p.add_argument("--rtag", default="discrim")
            p.add_argument("--n", type=int, default=600)
            p.add_argument("--out", required=True)
            p.add_argument("--random", action="store_true", help="每个种子随机取一条有效改写（闭环验证用）")
            p.add_argument("--hard_mix", action="store_true", help="按 hard_<tag> 的识破率：难 : 易 = 1 : 1，每个种子最多 1 条")
            p.add_argument("--neg_decisions", default="", help="ragtruth.negcheck 的 decisions.jsonl：只用原句复核为无幻觉的改写")
            p.add_argument("--arb_tag", default="", help="--random：仲裁确认的隐蔽错误也算有效")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--pct", action="store_true", help="按连续难度、真实性的百分位各占一半排序（同 DPO v2 综合分）")
        if name == "valid":
            p.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
    args = ap.parse_args()
    {"sample": cmd_sample, "valid": cmd_valid, "hard": cmd_hard, "arbitrate": cmd_arbitrate, "realism": cmd_realism,
     "pairs": cmd_pairs,
     "select": cmd_select, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
