"""把审核台的演示数据（review.demo_data）整理成冒烟测试用的冷启动集和训练集 v1。

样本只有十几条，重复几遍凑够步数；只为验证训练流程，不看效果。
用法：python -m scripts.smoke_data
"""
from __future__ import annotations

from pathlib import Path

from common.schema import Sample, read_jsonl, write_jsonl

DEMO = Path("outputs/demo")
OUT = Path("outputs/smoke")


def load(pattern: str) -> list[dict]:
    return [s for p in sorted(DEMO.glob(pattern)) for s in read_jsonl(p)]


def main() -> None:
    if not (DEMO / "positives.jsonl").exists():
        from review import demo_data
        demo_data.main()
    positives, synth, hard = load("positives.jsonl"), load("synth_batch?.jsonl"), load("hard_batch?.jsonl")
    for s in positives + synth + hard:
        Sample.from_dict(s)

    coldstart = (positives + synth) * 4
    print(f"coldstart: {write_jsonl(OUT / 'coldstart.jsonl', coldstart)} rows")

    train = [dict(s, split="train") for s in positives + hard] * 2
    val = [dict(s, split="val") for s in positives[:1] + hard[:1]]
    write_jsonl(OUT / "trainset_v1/train.jsonl", train)
    write_jsonl(OUT / "trainset_v1/val.jsonl", val)
    print(f"trainset_v1: train {len(train)}, val {len(val)}")


if __name__ == "__main__":
    main()
