"""9B 评测器和 Claude 金标的一致性。RAG 评测里所有"含显然错 / 含任意错"的比例都由 9B 金标 critic 打分，这里检验它和 Claude（代替人工）判的是不是同一件事。

  python -m scripts.analysis.eval9b_agreement                 # 默认：四组模型的验收回答 + 三套 2B GRPO 验收回答
  python -m scripts.analysis.eval9b_agreement --runs 名=目录:分数文件 ...

每组：目录下 final.jsonl 是 Claude 逐句金标（severity clear / minor / none），分数文件是 9B 的逐句 P(有错)
（[{"id", "p"}] 的 .json，或带选项 logprob 的 .jsonl，按 9B critic 的温度换成 P = B + C）。
- 句子级：显然错 vs 没错、任意错 vs 没错的 AUC；9B ≥ 0.9 的句子里 Claude 怎么判。
- 回答级（和 pol9b_cmp 的口径一致）：9B 含显然错 = 有一句 P ≥ 0.9，含任意错 = 有一句 P ≥ 0.5；
  Claude 含显然错 = 有一句 clear，含任意错 = 有一句 clear 或 minor。报两边的比例、一致率、Cohen's κ。
- 模型间比较（--pairs 甲>乙）：同一批原文上，9B 和 Claude 各自算出的"乙 − 甲"是否同向、幅度是否接近。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

T9B = 1.3   # rm1__rm1_gold_9b 的校准温度（outputs/rt/rm/rm1__rm1_gold_9b/eval/report.json）
RUNS = {
    "9B 原始": "outputs/rt/base9b:holdout_rm1__rm1_gold_9b.jsonl",
    "9B GRPO": "outputs/rt/ansgrpo/accept_sat:critic_scores.json",
    "2B 基线": "outputs/rt/minicpm/sys:critic_scores.json",
    "2B GRPO（验收 a）": "outputs/rt/minicpm/grpo300:critic_scores.json",
    "2B GRPO（验收 b）": "outputs/rt/minicpm/grpo300b:critic_scores.json",
    "2B GRPO（验收 c）": "outputs/rt/minicpm/grpo300c:critic_scores.json",
}
PAIRS = ["9B 原始>9B GRPO", "2B 基线>2B GRPO（验收 a）", "9B GRPO>2B GRPO（验收 a）"]


def load_scores(p: Path) -> dict[str, float]:
    if p.suffix == ".jsonl":
        from ragtruth.rm_eval import probs
        out = {}
        for l in open(p, encoding="utf-8"):
            x = json.loads(l)
            q = probs(x["lp"], T9B)
            out[x["id"]] = q["B"] + q["C"]
        return out
    return {r["id"]: float(r["p"]) for r in json.load(open(p, encoding="utf-8"))}


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def load_run(spec: str):
    d, f = spec.split(":", 1)
    sp = Path(d) / f
    if not sp.exists():
        return None
    sc = load_scores(sp)
    sents = []
    for l in open(Path(d) / "final.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in sc and r.get("severity") in ("clear", "minor", "none"):
            sents.append({"resp": r["resp_id"], "src": r["resp_id"].split("-", 1)[-1], "sev": r["severity"], "p": sc[r["id"]]})   # 原文-种子（去掉组前缀），用于跨组配对
    return sents


def answer_rows(sents):
    by = {}
    for x in sents:
        by.setdefault(x["resp"], []).append(x)
    return {k: {"src": xs[0]["src"], "9b_clear": any(x["p"] >= 0.9 for x in xs), "9b_any": any(x["p"] >= 0.5 for x in xs),
                "cl_clear": any(x["sev"] == "clear" for x in xs), "cl_any": any(x["sev"] != "none" for x in xs)} for k, xs in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="名=目录:分数文件（不给 = 默认四组 + 三套验收）")
    ap.add_argument("--pairs", nargs="*", default=None, help="甲>乙：比较 9B 和 Claude 对两组差值的判断")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs) if a.runs else RUNS
    pairs = a.pairs if a.pairs is not None else (PAIRS if not a.runs else [])
    rep = {"runs": {}, "pooled": {}, "pairs": {}}
    data = {}
    print(f"{'组':18s} {'句':>5s} {'AUC显然':>8s} {'AUC任意':>8s} | {'含显然错 9B/Claude':>18s} {'一致':>6s} {'κ':>5s} | {'含任意错 9B/Claude':>18s} {'一致':>6s} {'κ':>5s} | 9B≥0.9 里 Claude 显然/轻微/没错")
    for name, spec in runs.items():
        s = load_run(spec)
        if not s:
            print(f"{name:18s} 缺分数文件，跳过"); continue
        data[name] = s
        rows = list(answer_rows(s).values())
        cvn = [(x["p"], x["sev"] == "clear") for x in s if x["sev"] != "minor"]
        r = {"n_sent": len(s), "n_answers": len(rows), "auc_clear": auc(*zip(*cvn)),
             "auc_any": auc([x["p"] for x in s], [x["sev"] != "none" for x in s])}
        for m in ("clear", "any"):
            g9 = [x[f"9b_{m}"] for x in rows]; gc = [x[f"cl_{m}"] for x in rows]
            r[m] = {"rate_9b": float(np.mean(g9)), "rate_claude": float(np.mean(gc)),
                    "agreement": float(np.mean(np.array(g9) == np.array(gc))), "kappa": kappa(g9, gc)}
        hi = [x["sev"] for x in s if x["p"] >= 0.9]
        r["claude_on_9b_ge_0.9"] = {k: hi.count(k) for k in ("clear", "minor", "none")}
        rep["runs"][name] = r
        h = r["claude_on_9b_ge_0.9"]
        print(f"{name:18s} {len(s):5d} {r['auc_clear']:8.3f} {r['auc_any']:8.3f} | {r['clear']['rate_9b']:8.1%}/{r['clear']['rate_claude']:6.1%}   "
              f"{r['clear']['agreement']:6.1%} {r['clear']['kappa']:5.2f} | {r['any']['rate_9b']:8.1%}/{r['any']['rate_claude']:6.1%}   "
              f"{r['any']['agreement']:6.1%} {r['any']['kappa']:5.2f} | {h['clear']}/{h['minor']}/{h['none']}")
    # 合并
    alls = [x for s in data.values() for x in s]
    rows = [x for s in data.values() for x in answer_rows(s).values()]
    if alls:
        pooled = {"n_sent": len(alls), "n_answers": len(rows),
                  "auc_clear": auc(*zip(*[(x["p"], x["sev"] == "clear") for x in alls if x["sev"] != "minor"])),
                  "auc_any": auc([x["p"] for x in alls], [x["sev"] != "none" for x in alls])}
        for m in ("clear", "any"):
            g9 = [x[f"9b_{m}"] for x in rows]; gc = [x[f"cl_{m}"] for x in rows]
            pooled[m] = {"rate_9b": float(np.mean(g9)), "rate_claude": float(np.mean(gc)),
                         "agreement": float(np.mean(np.array(g9) == np.array(gc))), "kappa": kappa(g9, gc)}
        rep["pooled"] = pooled
        print(f"{'合并':18s} {len(alls):5d} {pooled['auc_clear']:8.3f} {pooled['auc_any']:8.3f} | {pooled['clear']['rate_9b']:8.1%}/{pooled['clear']['rate_claude']:6.1%}   "
              f"{pooled['clear']['agreement']:6.1%} {pooled['clear']['kappa']:5.2f} | {pooled['any']['rate_9b']:8.1%}/{pooled['any']['rate_claude']:6.1%}   "
              f"{pooled['any']['agreement']:6.1%} {pooled['any']['kappa']:5.2f}")
    # 模型间比较：同一批原文上，9B 和 Claude 各自算出的差值
    if pairs:
        print("\n模型间比较（乙 − 甲，同一批原文配对；括号是 9B / Claude 各自的值）")
    for pr in pairs:
        x, y = pr.split(">")
        if x not in data or y not in data:
            continue
        ra = {v["src"]: v for v in answer_rows(data[x]).values()}
        rb = {v["src"]: v for v in answer_rows(data[y]).values()}
        common = sorted(set(ra) & set(rb))
        out = {"n": len(common)}
        line = f"  {y} − {x}（{len(common)} 个原文-种子）："
        for m, lab in (("clear", "含显然错"), ("any", "含任意错")):
            d9 = np.mean([rb[k][f"9b_{m}"] for k in common]) - np.mean([ra[k][f"9b_{m}"] for k in common])
            dc = np.mean([rb[k][f"cl_{m}"] for k in common]) - np.mean([ra[k][f"cl_{m}"] for k in common])
            out[m] = {"diff_9b": float(d9), "diff_claude": float(dc), "same_sign": bool(np.sign(d9) == np.sign(dc))}
            line += f" {lab} 9B {d9 * 100:+.1f} / Claude {dc * 100:+.1f}{'（同向）' if np.sign(d9) == np.sign(dc) else '（反向）'}；"
        rep["pairs"][pr] = out
        print(line)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
