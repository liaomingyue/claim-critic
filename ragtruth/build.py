"""补写过 evidence / reason 的回答样本 → 逐句训练样本。

- 按句取舍：证据先做模糊修正（与原文只差标点、个别词的换成原文片段），仍不合格的句子丢掉；
- 大片段标出的句子（coarse）、补写模型判为标签存疑（disputed）的句子不进训练；
- 无问题的句子按 1:faithful_per_bad 下采样，默认 1:3（1:2 时误报偏多；比例是一个要试的超参，自然比例约 1:6–1:9）。

用法：python -m ragtruth.build --data outputs/rt/annotated/train_sub.jsonl --out outputs/rt/claims/train_sub.jsonl
"""
from __future__ import annotations

import argparse
import collections

from common.schema import read_jsonl, write_jsonl
from ragtruth.task import balance, expand_claims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/rt/annotated/train_sub.jsonl")
    ap.add_argument("--out", default="outputs/rt/claims/train_sub.jsonl")
    ap.add_argument("--faithful_per_bad", type=float, default=3.0,
                    help="幻觉句:无问题句 = 1:faithful_per_bad（默认 1:3，幻觉句约占 25%）；传 0 表示保持自然比例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    resp = list(read_jsonl(args.data))
    claims = [c for s in resp for c in expand_claims(s, require_analysis=True)]
    out = balance(claims, args.faithful_per_bad or None, args.seed)
    n_sent = sum(len(s["sentences"]) for s in resp if s.get("analysis"))
    print(f"{len(claims)}/{n_sent} sentences have a valid target; "
          f"kept {write_jsonl(args.out, out)} ({collections.Counter(c['type'] for c in out)})")


if __name__ == "__main__":
    main()
