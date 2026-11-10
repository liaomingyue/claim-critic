"""生成器 v2 的 GRPO（批量生产 + GRPO 优化，核心目标是 sev 奖励，看能不能增加合格样本量）。

  prompts  训练池（genpool_train 657 回答）× 9 机制 → outputs/rt/gen_v2_grpo/prompts.jsonl（gen_grpo 的行格式 + source / sentences / mechanism）
  items    评测采样（genpool_eval 留出原文）→ <name>/items.jsonl（aid = 2b/ + id，score_samples 用）
  report   各组（SFT 起点 / GRPO A / GRPO B）在留出原文上的产出率：按机制，sev（带 9B 描述）/ 自然 / critic 区间各级门槛
  python -m ragtruth.gen_v2_grpo prompts
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.gen_v2 import MECH, prompt

D = Path("outputs/rt/core")
O = Path("outputs/rt/gen_v2_grpo")


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_prompts(a) -> None:
    seen, rows = set(), []
    for r in read_jsonl(D / "genpool_train.jsonl"):
        if r["resp_id"] in seen:
            continue
        seen.add(r["resp_id"])
        sents = r["sentences"] if isinstance(r["sentences"], list) else json.loads(r["sentences"])
        for m in MECH:
            rows.append({"id": f"{r['resp_id']}|{m}", "prompt": prompt(r["source"], sents, m), "source": r["source"], "sentences": sents,
                         "mechanism": m, "resp_id": r["resp_id"], "task_type": r["task_type"]})
    random.Random("gen_v2_grpo|prompts").shuffle(rows)
    write(O / "prompts.jsonl", rows)
    print("GEN_V2_GRPO_PROMPTS", len(rows), "responses", len(seen), flush=True)


def cmd_items(a) -> None:
    xs = [x for x in read_jsonl(D / a.samples) if x.get("why") == "ok"]
    write(O / a.name / "items.jsonl", [{**x, "aid": "2b/" + x["id"], "group": a.name} for x in xs])
    n = sum(1 for _ in read_jsonl(D / a.samples))
    print("GEN_V2_GRPO_ITEMS", a.name, len(xs), "of", n, flush=True)


def cmd_report(a) -> None:
    res = {}
    for name in a.names:
        n_all = sum(1 for _ in read_jsonl(D / f"samples_grpo_{name}_mc.jsonl"))
        it = {x["aid"]: x for x in read_jsonl(O / name / "items.jsonl")}
        sc = {r["aid"]: r for r in read_jsonl(O / name / "scores.jsonl")}
        per = collections.defaultdict(collections.Counter)
        for k, x in it.items():
            s, m = sc.get(k) or {}, x["mechanism"]
            c = per[m]
            c["ok"] += 1
            if (s.get("v3_p_req") or 0) >= a.sev:
                c["sev"] += 1
                if (s.get("rm2v3_p_natural") or 0) >= 0.5:
                    c["sev+nat"] += 1
                    if a.lo <= (s.get("rm1_p_halluc") if s.get("rm1_p_halluc") is not None else -1) < a.hi:
                        c["sev+nat+band"] += 1
        tot = sum(per.values(), collections.Counter())
        res[name] = {"samples": n_all, **{k: f"{v} ({100 * v / max(n_all, 1):.1f}%)" for k, v in tot.items()},
                     "per_mech_sev%": {m: round(100 * per[m]["sev"] / max(n_all / len(MECH), 1), 1) for m in MECH}}
    print("GEN_V2_GRPO_REPORT", json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    (O / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_probe(a) -> None:
    """训练前试采（担心大量组没有可选的正样本，跑之前先测）：训练提示里取 --n_prompts 个，
    每个采 --n 条，用训练同款奖励（V2SevRewarder hard）打分，统计每组正样本（过门槛且难度 > 0）个数的分布。"""
    import zlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    from train.gen_reward import load_temps
    from train.gen_reward_v2 import V2SevRewarder
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rw = V2SevRewarder(cl, load_temps(a.temps), shape="hard", err_gate=a.err_gate, nat_gate=0.5)   # 统计用默认形状；门槛通过率 / critic 分布与形状无关
    rows = list(read_jsonl(O / "prompts.jsonl"))[-a.n_prompts:]   # 取末尾，训练 100 步只用到开头的提示

    def one(r):
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=512, temperature=1.0,
                                    n=a.n, seed=zlib.crc32(r["id"].encode()) % 100000, extra_body={"top_k": -1})
        with ThreadPoolExecutor(32) as ex:   # 并发打分
            ss = list(ex.map(lambda c: rw.score(c.text, r), out.choices))
        keep = ("why", "idx", "original", "sentence", "p_req", "p_err", "p_natural", "p_halluc", "gate", "hardness", "reward", "qualified")
        return {"id": r["id"], "mechanism": r["mechanism"], "pos": sum(bool(x["qualified"]) for x in ss), "gate": sum(bool(x.get("gate")) for x in ss),
                "rewards": [round(x["reward"], 3) for x in ss], "samples": [{k: x.get(k) for k in keep} for x in ss]}

    with ThreadPoolExecutor(a.concurrency) as ex:   # 提示并发：vLLM 自己合批
        res = list(ex.map(one, rows))
    pos = [r["pos"] for r in res]
    flat = sum(len(set(r["rewards"])) == 1 for r in res)
    summ = {"model": a.model, "prompts": len(res), "n": a.n, "pos_rate": round(sum(pos) / (len(res) * a.n), 3),
            "groups_zero_pos": round(sum(p == 0 for p in pos) / len(res), 3), "groups_1_2_pos": round(sum(1 <= p <= 2 for p in pos) / len(res), 3),
            "groups_flat_reward": round(flat / len(res), 3), "pos_hist": dict(sorted(collections.Counter(min(p, 10) for p in pos).items())),
            "zero_by_mech": {m: sum(r["pos"] == 0 for r in res if r["mechanism"] == m) for m in MECH}}
    sm = [x for r in res for x in r["samples"]]
    ok = [x for x in sm if x["why"] == "ok"]
    hist = lambda xs, b: dict(zip([f"<{b[1]}"] + [f"{lo}-{hi}" for lo, hi in zip(b[1:-2], b[2:-1])] + [f">={b[-2]}"],
                                  [int(v) for v in np.histogram(xs, b)[0]]))
    summ.update({"samples": len(sm), "valid": round(len(ok) / len(sm), 3), "why": dict(collections.Counter(x["why"] for x in sm)),
                 "pass_sev0.9": round(sum(x["p_err"] >= a.err_gate for x in ok) / len(sm), 3),
                 "pass_nat": round(sum(x["p_natural"] >= 0.5 for x in ok) / len(sm), 3),
                 "pass_gate": round(sum(bool(x["gate"]) for x in ok) / len(sm), 3),
                 "mean_reward": round(sum(x["reward"] for x in sm) / len(sm), 4),
                 "reward_hist": hist([x["reward"] for x in sm], [0, 1e-9, 0.1001, 0.4, 0.7, 1.01]),
                 "critic_hist_gated": hist([x["p_halluc"] for x in ok if x["gate"]], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]),
                 "group_reward_std_mean": round(float(np.mean([np.std(r["rewards"]) for r in res])), 4),
                 "pos_rate_by_mech": {m: round(sum(r["pos"] for r in res if r["mechanism"] == m) / max(1, a.n * sum(r["mechanism"] == m for r in res)), 3) for m in MECH}})
    write(O / f"probe_{a.model}.jsonl", res)
    print("GEN_V2_GRPO_PROBE", json.dumps(summ, ensure_ascii=False), flush=True)


def cmd_harvest(a) -> None:
    """方向②：现有生成器在训练提示（去掉末尾 64 个试采提示）上大批量采样，用训练同款奖励打分，
    挑「过门槛（sev 有错 ≥ 0.9、自然 ≥ 0.5）且 critic ∈ [0.1, 0.7)」的难样本（每提示最多 --per_prompt 条），
    和原 SFT 数据（seed_v2_train）拼成热启动 SFT 数据 → O/sft_hard.jsonl。目标 = 生成器自己写的 JSON（idx / sentence / edit）。"""
    import zlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    from train.gen_reward import load_temps
    from train.gen_reward_v2 import V2SevRewarder
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rw = V2SevRewarder(cl, load_temps(a.temps), shape="hard", err_gate=a.err_gate, nat_gate=0.5)
    # --prompts：别的提示池（如 rag_pool/prompts.jsonl，更多问答对、每个提示温度 0 采 1 条）；默认 = 训练提示去掉末尾 64 个试采提示
    rows = list(read_jsonl(O / a.prompts)) if a.prompts else list(read_jsonl(O / "prompts.jsonl"))[:-a.n_prompts]

    def one(r):
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=512, temperature=a.temperature,
                                    n=a.n, seed=zlib.crc32(("harvest|" + r["id"]).encode()) % 100000, extra_body={"top_k": -1})
        with ThreadPoolExecutor(32) as ex:
            ss = list(ex.map(lambda c: (c.text, rw.score(c.text, r)), out.choices))
        clean = set(r.get("clean_idx") or [])   # 提示池给了干净句（critic 判没错）就只认改这些句的样本
        dirty = sum(bool(clean) and x.get("why") == "ok" and x["idx"] not in clean for _, x in ss)
        hard = [(t, x) for t, x in ss if x.get("gate") and a.lo <= x["p_halluc"] < a.hi and x["p_req"] >= a.min_req and (not clean or x["idx"] in clean)]
        hard.sort(key=lambda tx: abs(tx[1]["p_halluc"] - 0.4))   # 优先 0.4 附近
        keep, seen = [], set()
        for t, x in hard:
            if x["sentence"] in seen or len(keep) >= a.per_prompt:
                continue
            seen.add(x["sentence"])
            keep.append({"id": r["id"], "mechanism": r["mechanism"], "prompt": r["prompt"], "idx": x["idx"], "sentence": x["sentence"],
                         "original": x["original"], "edit": x.get("edit_gen") or "", "p_err": x["p_err"], "p_req": x["p_req"],
                         "p_natural": x["p_natural"], "p_halluc": x["p_halluc"]})
        return {"n": len(ss), "ok": sum(x["why"] == "ok" for _, x in ss), "gate": sum(bool(x.get("gate")) for _, x in ss), "hard": len(hard), "dirty": dirty, "keep": keep}

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, rows))
    hard = [k for r in res for k in r["keep"]]
    write(O / f"harvest_hard{a.tag}.jsonl", hard)
    if a.prev_hard:   # 上一轮的难样本里同样满足本轮条件的也带上（同一提示同一句去重）
        have = {(h["id"], h["sentence"]) for h in hard}
        prev = [h for h in read_jsonl(O / a.prev_hard) if a.lo <= h["p_halluc"] < a.hi and h["p_req"] >= a.min_req and (h["id"], h["sentence"]) not in have]
        hard += prev
        print("PREV_HARD", len(prev), flush=True)
    seed = list(read_jsonl(D / "seed_v2_train.jsonl"))
    sft = [{"id": f"seed/{x['id']}", "prompt": x["prompt"], "completion": x["completion"]} for x in seed]
    sft += [{"id": f"hard/{h['id']}/{k}", "prompt": h["prompt"],
             "completion": json.dumps({"idx": h["idx"], "sentence": h["sentence"], "edit": h["edit"]}, ensure_ascii=False)} for k, h in enumerate(hard)]
    random.Random("gen_v2_grpo|sft_hard").shuffle(sft)
    write(O / f"sft_hard{a.tag}.jsonl", sft)
    n = sum(r["n"] for r in res)
    print("GEN_V2_GRPO_HARVEST", json.dumps({"prompts": len(res), "samples": n, "ok": round(sum(r["ok"] for r in res) / n, 3),
                                             "gate": round(sum(r["gate"] for r in res) / n, 3), "hard_0.1-0.7": round(sum(r["hard"] for r in res) / n, 4),
                                             "dirty_idx": round(sum(r.get("dirty", 0) for r in res) / n, 4), "kept_with_prev": len(hard), "lo": a.lo, "hi": a.hi, "min_req": a.min_req, "prompts_with_hard": sum(bool(r["keep"]) for r in res),
                                             "kept_by_mech": dict(collections.Counter(h["mechanism"] for h in hard)), "sft_rows": len(sft)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "items", "report", "probe", "harvest"])
    ap.add_argument("--per_prompt", type=int, default=2, help="harvest：每个提示最多留几条难样本")
    ap.add_argument("--min_req", type=float, default=0.0, help="harvest：sev P(要求的那种) 下限（类型一致）")
    ap.add_argument("--tag", default="", help="harvest：输出文件名后缀")
    ap.add_argument("--prompts", default="", help="harvest：提示池文件（O 下），默认 prompts.jsonl 去掉试采提示")
    ap.add_argument("--temperature", type=float, default=1.0, help="harvest：采样温度（0 = 确定性，配 --n 1）")
    ap.add_argument("--prev_hard", default="", help="harvest：上一轮难样本文件（O 下），满足本轮条件的并进来")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="policy_v0")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--temps", default="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=rm1__core_r2")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=32, help="probe：同时在跑的提示数")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--err_gate", type=float, default=0.9)
    ap.add_argument("--samples", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--names", nargs="*", default=["v0", "ga", "gb"])
    ap.add_argument("--sev", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.7)
    a = ap.parse_args()
    {"prompts": cmd_prompts, "items": cmd_items, "report": cmd_report, "probe": cmd_probe, "harvest": cmd_harvest}[a.cmd](a)


if __name__ == "__main__":
    main()
