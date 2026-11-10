"""奖励设计对比的独立裁判：奖励里用了 rm2v2__mcgen3 和 2B critic，只用它们评测等于让考官自己判卷。
快评样本（{run}/quick_samples_{prefix}s{step}.jsonl）里合格的（P(真错) ≥ gate、原句干净），再用两个奖励外的模型打分：
  9B 金标 critic（rm1__rm1_gold_9b，训练外调用用 9B）：P(判出) < 0.5 算骗过；
  第一版 RM2（rm2v2__mcgen，第四版奖励没用它）：P(自然) > 0.5 算自然。
同时记 2B critic / 奖励里的 RM2 的分数（快评时已打），方便看两边是否背离。

  python -m ragtruth.gen_indep_eval --run outputs/rt/gentarget/grpo4 --prefix t4_ --steps 0,50 \
      --url9b http://127.0.0.1:8001/v1 --url2b http://127.0.0.1:8002/v1
写 {run}/indep_eval.json（每步一行汇总）和 {run}/indep_s{step}.jsonl（逐条）。
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl
from ragtruth import discrim
from ragtruth.gen_target_data import _clients, _pick
from ragtruth.rm_data import RM1_SYSTEM, RM2G_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.task import render_input
from train.gen_reward import NO_THINK, context, parse


def temp(rm: str) -> float:
    return json.load(open(f"outputs/rt/rm/{rm}/eval/report.json", encoding="utf-8"))["temperature"]


def opt(cl, model: str, system: str, user: str, opts: str, t: float) -> dict | None:
    resp = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
    lp = option_logprobs(resp.choices[0], opts)
    return probs(lp, t) if lp else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--steps", default="0,50")
    ap.add_argument("--eval_prompts", default="outputs/rt/gentarget/eval_prompts.jsonl")
    ap.add_argument("--url9b", required=True)
    ap.add_argument("--url2b", required=True)
    ap.add_argument("--critic9b", default="critic", help="9B 服务上金标 critic 的名字")
    ap.add_argument("--rm2", default="rm2v1", help="2B 服务上第一版 RM2 的名字")
    ap.add_argument("--gate", type=float, default=0.7)
    a = ap.parse_args()
    rows = {r["id"]: r for r in read_jsonl(a.eval_prompts)}
    c9, c2 = _clients(a.url9b), _clients(a.url2b)
    t9, t2 = temp("rm1__rm1_gold_9b"), temp("rm2v2__mcgen")
    summ = []
    for step in [int(s) for s in a.steps.split(",")]:
        f = Path(a.run) / f"quick_samples_{a.prefix}s{step}.jsonl"
        if not f.exists():
            print("missing", f, flush=True)
            continue
        jobs = []
        for l in read_jsonl(f):
            row = rows[l["id"]]
            for text, s in zip(l["outputs"], l["scores"]):
                if s.get("why") == "ok" and s.get("p_correct", 0) >= a.gate and s.get("p_orig_halluc", 0.0) <= 0.5 and "p_halluc" in s:
                    rec, _ = parse(text, row)
                    if rec is not None:
                        jobs.append((rec, s))

        def run(job):
            rec, s = job
            ctx = context(rec)
            p9 = opt(_pick(c9, rec["sentence"]), a.critic9b, RM1_SYSTEM, render_input({"source": rec["source"], "sentences": ctx, "idx": rec["idx"]}), "ABC", t9)
            p2 = opt(_pick(c2, rec["sentence"]), a.rm2, RM2G_SYSTEM, discrim.render(rec["source"], ctx, rec["idx"]), "AB", t2)
            return {"type": rec["type"], "sentence": rec["sentence"], "original": rec["original"], "p_correct": s["p_correct"], "p_a": s.get("p_a"),
                    "p_halluc_2b": s["p_halluc"], "p_natural_rw": s["p_natural"],
                    "p_halluc_9b": (p9["B"] + p9["C"]) if p9 else None, "p_natural_v1": p2["A"] if p2 else None}

        with ThreadPoolExecutor(32) as ex:
            out = [r for r in ex.map(run, jobs) if r["p_halluc_9b"] is not None and r["p_natural_v1"] is not None]
        (Path(a.run) / f"indep_s{step}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        n = max(1, len(out))
        rec = {"run": a.run, "step": step, "qualified": len(out),
               "fool_2b": round(sum(r["p_halluc_2b"] < 0.5 for r in out) / n, 4), "fool_9b": round(sum(r["p_halluc_9b"] < 0.5 for r in out) / n, 4),
               "natural_rw": round(sum(r["p_natural_rw"] > 0.5 for r in out) / n, 4), "natural_v1": round(sum(r["p_natural_v1"] > 0.5 for r in out) / n, 4),
               "fool_9b_and_natural_v1": round(sum(r["p_halluc_9b"] < 0.5 and r["p_natural_v1"] > 0.5 for r in out) / n, 4),
               "baseless_share": round(sum(r["type"] == "baseless" for r in out) / n, 4),
               "explanation_wrong_share": round(sum((r["p_a"] or 0) < a.gate for r in out) / n, 4)}
        summ.append(rec)
        print("INDEP_EVAL", json.dumps(rec, ensure_ascii=False), flush=True)
    with open(Path(a.run) / "indep_eval.json", "a", encoding="utf-8") as fo:
        for r in summ:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
