"""critic 循环的迭代实验：每轮从新合成池里选样本、累积进训练集，重新训练 critic，再用新 critic 挑下一轮。

两条序列共用同一批合成池（每轮新句子，经 ragtruth.fulltext 全文核对过滤），只有「怎么选」不同：
  random：从本轮池里随机取 n 对；
  critic：本轮 critic（上一轮训出来的）识破率 < threshold 的为难样本，难 : 易 = 1 : 1 各取 n/2 对；
          难样本不够时全取，用易样本补足 n 对（记录实际比例）。
训练集 = 基础数据 + 本序列第 1..r 轮选中的全部样本（成对加入，累积而不替换），每轮从基座重新训练。

用法：
  python -m ragtruth.iterate select --series critic --pool outputs/rt/synth/r2_ft.jsonl \
      --mined outputs/rt/iter/critic_r2_mined.jsonl --n 600 --out outputs/rt/iter/critic_r2_sel.jsonl
  python -m ragtruth.iterate build --base outputs/rt/claims/train_sub35_r3.jsonl \
      --sel outputs/rt/iter/critic_r1_sel.jsonl outputs/rt/iter/critic_r2_sel.jsonl --out outputs/rt/claims/iter_critic_r2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from common.llm import kept
from common.schema import read_jsonl, write_jsonl
from ragtruth.synth import to_claims


def select(pool: list[dict], rates: dict[str, float], series: str, n: int, threshold: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if series == "random":
        return rng.sample(pool, min(n, len(pool)))
    hard = [r for r in pool if r["id"] in rates and rates[r["id"]] < threshold]
    easy = [r for r in pool if r["id"] in rates and rates[r["id"]] >= threshold]
    h = rng.sample(hard, min(n // 2, len(hard)))
    e = rng.sample(easy, min(n - len(h), len(easy)))
    return h + e


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--series", choices=["random", "critic"], required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--mined", required=True, help="本序列当前 critic 对本轮池的识破率（ragtruth.mine mine 的输出）")
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--sel", nargs="+", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "select":
        pool = list(read_jsonl(args.pool))
        rates = {r["id"]: r["detect_rate"] for r in kept(args.mined)}
        chosen = select(pool, rates, args.series, args.n, args.threshold, args.seed)
        write_jsonl(args.out, chosen)
        hard_all = sum(rates.get(r["id"], 1) < args.threshold for r in pool)
        meta = {"series": args.series, "pool": len(pool), "pool_hard": hard_all,
                "pool_hard_rate": round(hard_all / len(pool), 3) if pool else None, "chosen": len(chosen),
                "chosen_hard": sum(rates.get(r["id"], 1) < args.threshold for r in chosen),
                "types": dict(collections.Counter(r["type"] for r in chosen)),
                "mean_detect_rate_pool": round(sum(rates.values()) / len(rates), 3) if rates else None}
        with open(args.out.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta))
    else:
        base = list(read_jsonl(args.base))
        added = [c for p in args.sel for r in read_jsonl(p) for c in to_claims(r)]
        out = base + added
        random.Random(args.seed).shuffle(out)
        write_jsonl(args.out, out)
        print(json.dumps({"base": len(base), "added_claims": len(added),
                          "hallucination_share": round(sum(c["label"] == "hallucination" for c in out) / len(out), 3)}))


if __name__ == "__main__":
    main()
