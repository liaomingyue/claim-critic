"""按 Claude 标的错误类型看 critic 的抓取率（2026-11-03，用户：critic 自己能不能判出这些错误）。
验收目录里 final.jsonl（Claude 逐句标注，含 severity / mechanism）+ critic 分数文件（[{"id": 句子 id, "p": P(有错)}]），P ≥ 0.5 算判出。

  python -m ragtruth.critic_types     # 写 outputs/rt/critic_types.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from common.schema import read_jsonl

RUNS = {   # (组, critic): (验收目录, 分数文件)
    ("9b_base", "2b"): ("outputs/rt/base9b", "critic2b_scores.json"),
    ("9b_base", "9b"): ("outputs/rt/base9b", "holdout_rm1__rm1_gold_9b.jsonl"),   # 选项 logprob，按温度 1.3 换成 P
    ("9b_grpo", "2b"): ("outputs/rt/ansgrpo/accept_sat", "critic2b_scores.json"),
    ("9b_grpo", "9b"): ("outputs/rt/ansgrpo/accept_sat", "critic_scores.json"),
    ("2b_base", "2b"): ("outputs/rt/minicpm/sys", "critic2b_scores.json"),
    ("2b_base", "9b"): ("outputs/rt/minicpm/sys", "critic_scores.json"),
    ("2b_grpo", "2b"): ("outputs/rt/minicpm/grpo300", "critic2b_scores.json"),
    ("2b_grpo", "9b"): ("outputs/rt/minicpm/grpo300", "critic_scores.json"),
}


def bucket(r: dict) -> str:
    if not r["is_error"]:
        return "not_error"
    return f"{'clear' if r['severity'] == 'clear' else 'minor'}:{r['mechanism']}"


def rates(d: str, f: str) -> dict | None:
    p = Path(d) / f
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        from ragtruth.rm_eval import probs
        sc = {x["id"]: (lambda q: q["B"] + q["C"])(probs(x["lp"], 1.3)) for x in read_jsonl(p)}
    else:
        sc = {x["id"]: x["p"] for x in json.load(open(p, encoding="utf-8"))}
    by = collections.defaultdict(list)
    for r in read_jsonl(Path(d) / "final.jsonl"):
        if r["id"] in sc:
            hit = sc[r["id"]] >= .5
            by[bucket(r)].append(hit)
            if r["is_error"]:
                by["clear:ALL" if r["severity"] == "clear" else "minor:ALL"].append(hit)
    return {k: {"n": len(v), "flagged": round(sum(v) / len(v), 3)} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="组|critic=目录:分数文件（不给 = RUNS）；重训 critic 后对比用")
    ap.add_argument("--out", default="outputs/rt/critic_types.json")
    a = ap.parse_args()
    runs = {tuple(k.split("|", 1)): tuple(v.split(":", 1)) for k, v in (x.split("=", 1) for x in a.runs)} if a.runs else RUNS
    rep = {f"{g}|{c}": r for (g, c), (d, f) in runs.items() if (r := rates(d, f)) is not None}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    keys = sorted({k for r in rep.values() for k in r}, key=lambda k: (k != "not_error", not k.endswith("ALL"), k))
    print(f"{'type':38s}" + "".join(f"{c:>16s}" for c in rep))
    for k in keys:
        print(f"{k:38s}" + "".join(f"{(str(round(100 * r[k]['flagged'])) + '% (' + str(r[k]['n']) + ')') if k in r else '-':>16s}" for r in rep.values()))


if __name__ == "__main__":
    main()
