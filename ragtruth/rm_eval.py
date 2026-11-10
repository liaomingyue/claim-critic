"""选项 RM 的打分、校准和评测：读答案位置上各选项的 logprob，在校准集上拟合一个全局温度，测试集上报指标。

  python -m ragtruth.rm_eval --rm rm1 --model rm1 --base_url http://127.0.0.1:8000/v1

打分：关思考、贪心、只生成几个 token，取第一个内容 token 处 top-20 logprob 里各选项字母的概率（不在前 20 的记 −30）。
校准：p_i ∝ exp(logp_i / T)，T 在校准集上按多类负对数似然网格搜索，全局一个（不要多套参数）。
指标都按二分类目标算：RM1 P(有幻觉) = P(B) + P(C)；RM2 P(自然) = P(A)；RM3 P(合格) = P(A)。
报 AUC、ECE（10 等宽箱）、Brier、阈值 0.5 的召回 / 精确率；RM1 另按 ++ 严格、有效（去掉 Claude 判 mild / nitpick / meta 的）、
Claude 审过的一批（以 Claude 结论为准）分别算。输出 outputs/rt/rm/<rm>/eval/{calib,test}_scores.jsonl、report.json。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable

OPTS = {"rm1": "ABC", "rm2": "AB", "rm3": "ABCD", "rm2v2": "AB", "sat_pair": "AB", "sat_abs": "AB", "refuse": "AB", "sev": "ABCDEF"}
POS = {"rm1": "BC", "rm2": "A", "rm3": "A", "rm2v2": "A", "sat_pair": "A", "sat_abs": "A", "refuse": "B", "sev": "AB"}   # 二分类目标的正类选项
NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def option_logprobs(choice, opts: str) -> dict[str, float] | None:
    toks = (choice.logprobs.content if choice.logprobs else None) or []
    for t in toks:   # 跳过思考标签、空白
        if t.token.strip() in ("", "<think>", "</think>"):
            continue
        lp = {o: -30.0 for o in opts}
        for c in t.top_logprobs:
            h = c.token.strip().rstrip(".").upper()
            if h in lp:
                lp[h] = max(lp[h], c.logprob)
        return lp
    return None


def probs(lp: dict[str, float], T: float) -> dict[str, float]:
    z = max(lp.values())
    e = {k: math.exp((v - z) / T) for k, v in lp.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_T(rows: list[dict]) -> float:
    def nll(T):
        return -sum(math.log(max(1e-12, probs(r["lp"], T)[r["label"]])) for r in rows) / len(rows)
    grid = [round(0.2 + 0.05 * i, 2) for i in range(197)]   # 0.2 – 10
    return min(grid, key=nll)


def auc(ys: list[int], ps: list[float]) -> float | None:
    pos = [p for y, p in zip(ys, ps) if y]
    neg = [p for y, p in zip(ys, ps) if not y]
    if not pos or not neg:
        return None
    order = sorted(ps)
    rank = {}
    i = 0
    while i < len(order):   # 平均秩处理并列
        j = i
        while j < len(order) and order[j] == order[i]:
            j += 1
        rank[order[i]] = (i + j + 1) / 2
        i = j
    return (sum(rank[p] for p in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(ys: list[int], ps: list[float]) -> dict:
    n = len(ys)
    if not n:
        return {"n": 0}
    bins = [[] for _ in range(10)]
    for y, p in zip(ys, ps):
        bins[min(9, int(p * 10))].append((y, p))
    ece = sum(len(b) / n * abs(sum(y for y, _ in b) / len(b) - sum(p for _, p in b) / len(b)) for b in bins if b)
    tp = sum(1 for y, p in zip(ys, ps) if y and p >= 0.5)
    fp = sum(1 for y, p in zip(ys, ps) if not y and p >= 0.5)
    P = sum(ys)
    a = auc(ys, ps)
    return {"n": n, "pos": P, "auc": round(a, 4) if a is not None else None, "ece": round(ece, 4),
            "brier": round(sum((p - y) ** 2 for y, p in zip(ys, ps)) / n, 4), "mean_p": round(sum(ps) / n, 4),
            "pos_rate": round(P / n, 4), "recall@0.5": round(tp / P, 4) if P else None,
            "precision@0.5": round(tp / (tp + fp), 4) if tp + fp else None,
            "reliability": [{"bin": f"{k / 10:.1f}", "n": len(b), "mean_p": round(sum(p for _, p in b) / len(b), 3),
                             "frac_pos": round(sum(y for y, _ in b) / len(b), 3)} for k, b in enumerate(bins) if b]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, help="rm1 / rm2 / rm3 / rm2v2，或数据对比的 rm1__<组>")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    d = Path("outputs/rt/rm") / args.rm
    (d / "eval").mkdir(parents=True, exist_ok=True)
    kind = args.rm.split("__")[0]
    opts, pos = OPTS[kind], POS[kind]
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=8, concurrency=args.concurrency, timeout=600))

    def run(r: dict) -> dict:
        resp = llm.client.chat.completions.create(model=args.model, messages=r["prompt"], max_tokens=8, temperature=0.0,
                                                  logprobs=True, top_logprobs=20, **NO_THINK)
        keep = {k: v for k, v in r.items() if k not in ("prompt", "completion")}
        return {**keep, "lp": option_logprobs(resp.choices[0], opts), "text": resp.choices[0].message.content}

    scored = {}
    for sp in ("calib", "test"):
        items = [json.loads(l) for l in open(d / f"{sp}.jsonl", encoding="utf-8") if l.strip()]
        map_resumable(items, run, d / "eval" / f"{sp}_scores.jsonl", concurrency=args.concurrency)
        scored[sp] = [r for r in kept(d / "eval" / f"{sp}_scores.jsonl") if r.get("lp")]
    T = fit_T(scored["calib"])
    report = {"rm": args.rm, "temperature": T, "unparsed": {sp: sum(1 for r in kept(d / "eval" / f"{sp}_scores.jsonl")
                                                                    if not r.get("lp")) for sp in scored}}
    for sp, rs in scored.items():
        for name, t in (("raw", 1.0), ("calibrated", T)):
            ps = [sum(probs(r["lp"], t)[o] for o in pos) for r in rs]
            ys = [int(r["label"] in pos) for r in rs]
            report[f"{sp}_{name}"] = metrics(ys, ps)
            report[f"{sp}_{name}"]["option_acc"] = round(sum(max(probs(r["lp"], t), key=probs(r["lp"], t).get) == r["label"]
                                                              for r in rs) / len(rs), 4)
            if kind == "rm1" and sp == "test":   # ++ 严格、有效、Claude 审过的一批
                from ragtruth.pp_valid import load as pp_labels
                lab = pp_labels()
                valid = [(y, p) for r, y, p in zip(rs, ys, ps) if not (y and lab.get(r["id"], "real") != "real")]
                rev = [(int(r["claude"] in ("hallucination", "real", "mild")), p)   # 严格口径：mild 算幻觉，nitpick / meta 不算
                       for r, p in zip(rs, ps) if r.get("batch") == "reviewed" and r.get("claude")]
                report[f"{sp}_{name}"]["valid"] = metrics([y for y, _ in valid], [p for _, p in valid])
                report[f"{sp}_{name}"]["reviewed_claude"] = metrics([y for y, _ in rev], [p for _, p in rev])
    (d / "eval" / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "reliability"})
                      for k, v in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
