"""从 TRL / transformers 的训练日志里取出每次 logging 的 dict，画 loss、token 准确率和学习率曲线。

用法：python -m scripts.plot_loss outputs/rt/logs/sft-0.6b.log --out outputs/rt/logs/sft-0.6b-loss.png
"""
from __future__ import annotations

import argparse
import ast
import re


def parse_log(path: str) -> list[dict]:
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    rows = []
    for m in re.finditer(r"\{'loss':[^{}]*\}", text):
        d = ast.literal_eval(m.group(0))
        rows.append({k: float(v) for k, v in d.items()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    rows = parse_log(args.log)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [r["loss"] for r in rows], marker=".")
    axes[0].set_yscale("log")
    axes[0].set_title("train loss (log scale)")
    axes[1].plot(ep, [r["mean_token_accuracy"] for r in rows], marker=".", color="tab:green")
    axes[1].set_title("mean token accuracy (completion)")
    axes[2].plot(ep, [r["learning_rate"] for r in rows], color="tab:orange")
    axes[2].set_title("learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(args.title or args.log)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"{len(rows)} points -> {args.out}")


if __name__ == "__main__":
    main()
