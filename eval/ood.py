"""分布外评测：把 LLM-AggreFact 的指定子集转成本项目的测试集格式，再用 eval/run_eval.py 跑。
只报结果，不调参。该数据集没有错误类型和证据，所以只看 F1 和 accuracy。

用法：
  python -m eval.ood --subsets RAGTruth AggreFact-CNN --out outputs/ood/test.jsonl
  python -m eval.run_eval --test outputs/ood/test.jsonl --model critic-3 --out outputs/eval/ood-critic-3
"""
from __future__ import annotations

import argparse

from common.schema import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["RAGTruth"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_doc_chars", type=int, default=12000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("lytang/LLM-AggreFact", split=args.split)
    rows = []
    for i, ex in enumerate(ds):
        if ex["dataset"] not in args.subsets:
            continue
        faithful = int(ex["label"]) == 1
        rows.append({
            "id": f"ood-{ex['dataset']}-{i}",
            "split": "test",
            "source_group": ex["dataset"],
            "refs": {"ref-1": ex["doc"][: args.max_doc_chars]},
            "claim": ex["claim"],
            "label": "faithful" if faithful else "hallucination",
            # 原数据无类型；占一个合法值以通过 Sample 校验，按类型召回对 OOD 不报
            "type": "none" if faithful else "unsupported_inference",
            "evidence": "",
            "origin": "human",
        })
    print(f"wrote {write_jsonl(args.out, rows)} OOD samples from {args.subsets}")


if __name__ == "__main__":
    main()
