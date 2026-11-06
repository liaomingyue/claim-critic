"""critic 能力评测（2026-11-05，用户：这次评测主要看这两个能力是不是变强）：
  ① 显然错误的排序能力：clear vs 没有错 的 AUC；
  ② 固定误报率下的召回：在「没有错」的句子里取 5% / 10% 误报的阈值，看显然错误召回多少；
  另外看细节多的正确句子（数字、专名多）上的误报（h2 的误报主要出在这里）。
两套数据：
  claude —— 200 条 2B GRPO 验收回答（grpo300 + grpo300b，Claude 逐句标，final.jsonl；critic 分在同目录 critic_<名>.json）；
  proxy  —— 回答 GRPO 策略自己的回答（cmp_h2 的 old/new × a/b），9B 金标 critic 只当评测标签（不筛训练数据）：P ≥ 0.9 记显然错、< 0.1 记没错。

  python -m ragtruth.critic_cap --critics rm1__mcgold rm1__mcgold_ref rm1__mcgold_h3 --out outputs/rt/critic_cap_h3.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common.schema import read_jsonl

CLAUDE_DIRS = ["outputs/rt/minicpm/grpo300", "outputs/rt/minicpm/grpo300b"]
PROXY_DIR = "outputs/rt/minicpm/cmp_h2"
PROXY_SETS = ["old_a", "old_b", "new_a", "new_b"]
DETAIL_RE = re.compile(r"\d[\d,.:%]*|(?<!^)(?<![.!?] )\b[A-Z][a-z]+")


def detail_count(s: str) -> int:
    return len(DETAIL_RE.findall(s))


def auc(ys: list[int], ps: list[float]) -> float | None:
    pos = [p for y, p in zip(ys, ps) if y]
    neg = [p for y, p in zip(ys, ps) if not y]
    if not pos or not neg:
        return None
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(s / (len(pos) * len(neg)), 4)


def thr_at_fp(neg: list[float], fp: float) -> float:
    xs = sorted(neg, reverse=True)
    k = int(fp * len(xs))
    return xs[k] if k < len(xs) else 0.0


def metrics(pos: list[float], neg: list[float], detail_neg: list[float]) -> dict:
    out = {"n_clear": len(pos), "n_ok": len(neg), "clear_auc": auc([1] * len(pos) + [0] * len(neg), pos + neg)}
    for fp in (0.05, 0.10):
        t = thr_at_fp(neg, fp)
        out[f"recall@fp{int(fp * 100)}"] = round(sum(p > t for p in pos) / max(1, len(pos)), 3)
        if detail_neg:
            out[f"detail_fp@thr_fp{int(fp * 100)}"] = round(sum(p > t for p in detail_neg) / len(detail_neg), 3)
    out["recall@0.5"] = round(sum(p >= 0.5 for p in pos) / max(1, len(pos)), 3)
    out["fp@0.5"] = round(sum(p >= 0.5 for p in neg) / max(1, len(neg)), 3)
    if detail_neg:
        out["n_detail_ok"] = len(detail_neg)
        out["detail_fp@0.5"] = round(sum(p >= 0.5 for p in detail_neg) / len(detail_neg), 3)
    return out


def scores(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return {r["id"]: float(r["p"]) for r in json.load(open(path, encoding="utf-8"))}


def critic_file(d: str, name: str) -> Path:
    return Path(d) / ("critic2b_scores.json" if name == "rm1__mcgold" else f"critic_{name}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critics", nargs="+", required=True)
    ap.add_argument("--detail_min", type=int, default=0, help="细节多的阈值（数字 + 句中大写词数）；0 = 取没错句子里的前三分之一")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [x for d in CLAUDE_DIRS for x in read_jsonl(Path(d) / "final.jsonl")]
    ok = [x for x in rows if not x["is_error"]]
    dmin = a.detail_min or sorted(detail_count(x["sentence"]) for x in ok)[int(len(ok) * 2 / 3)]
    rep = {"claude": {"detail_min": dmin}, "proxy": {}}
    for c in a.critics:
        sc = {}
        for d in CLAUDE_DIRS:
            sc.update(scores(critic_file(d, c)) or {})
        have = [x for x in rows if x["id"] in sc]
        if not have:
            rep["claude"][c] = None
            continue
        pos = [sc[x["id"]] for x in have if x["is_error"] and x["severity"] == "clear"]
        neg = [sc[x["id"]] for x in have if not x["is_error"]]
        dneg = [sc[x["id"]] for x in have if not x["is_error"] and detail_count(x["sentence"]) >= dmin]
        m = metrics(pos, neg, dneg)
        m["all_error_auc"] = auc([int(x["is_error"]) for x in have], [sc[x["id"]] for x in have])
        rep["claude"][c] = m
        # 策略自己的回答（9B 当评测标签）
        pp, nn = [], []
        for s in PROXY_SETS:
            g = scores(Path(PROXY_DIR) / f"{s}_9b.json")
            mine = scores(Path(PROXY_DIR) / f"{s}_{c}.json")
            if not g or not mine:
                continue
            for k, p9 in g.items():
                if k in mine:
                    (pp if p9 >= 0.9 else nn if p9 < 0.1 else []).append(mine[k])
        rep["proxy"][c] = metrics(pp, nn, []) if pp and nn else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for part in ("claude", "proxy"):
        for c in a.critics:
            m = rep[part].get(c)
            if m:
                print("CRITIC_CAP", part, c, json.dumps({k: m.get(k) for k in ("n_clear", "clear_auc", "recall@fp5", "recall@fp10", "detail_fp@0.5", "detail_fp@thr_fp10", "fp@0.5")}), flush=True)


if __name__ == "__main__":
    main()
