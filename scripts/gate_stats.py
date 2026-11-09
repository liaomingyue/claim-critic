"""生成器 GRPO 训练中每道门槛的通过率（2026-11-09 用户：中途不做 Claude 抽查，但要看这些指标）。
读 <run>/samples/step*.jsonl.gz（训练时每步的原始输出和打分），每 --every 步一行：
格式合格 / sev 有错 ≥ 0.9 / 自然 ≥ 0.5 / 两道都过 / 正样本（过门槛且难度 > 0）/ 过门槛样本的 critic 分布。
  python scripts/gate_stats.py outputs/rt/gen_v2_grpo/h1 --every 10
"""
import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--err_gate", type=float, default=0.9)
    a = ap.parse_args()
    bins = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    for run in a.runs:
        files = sorted(Path(run, "samples").glob("step*.jsonl.gz"), key=lambda p: int(p.name[4:].split(".")[0]))
        print(f"\n== {run}")
        print("步 | 合格 | sev≥0.9 | 自然 | 两道都过 | 正样本 | 平均奖励 | 过门槛的 critic 分布 " + " ".join(f"{lo}-{hi}" for lo, hi in bins))
        for k in range(0, len(files), a.every):
            xs = [json.loads(l) for f in files[k:k + a.every] for l in gzip.open(f, "rt", encoding="utf-8")]
            n = len(xs)
            ok = [x for x in xs if x.get("why") == "ok"]
            g = [x for x in ok if x.get("gate")]
            pc = lambda v: f"{100 * v / max(n, 1):.1f}%"
            hist = [sum(lo <= x["p_halluc"] < hi for x in g) / max(len(g), 1) for lo, hi in bins]
            s0, s1 = files[k].name[4:].split(".")[0], files[min(k + a.every, len(files)) - 1].name[4:].split(".")[0]
            print(f"{s0}-{s1} | {pc(len(ok))} | {pc(sum(x['p_err'] >= a.err_gate for x in ok))} | {pc(sum(x['p_natural'] >= 0.5 for x in ok))} | "
                  f"{pc(len(g))} | {pc(sum(bool(x.get('qualified')) for x in xs))} | {sum(x['reward'] for x in xs) / max(n, 1):.3f} | "
                  + " ".join(f"{100 * h:.0f}%" for h in hist))


if __name__ == "__main__":
    main()
