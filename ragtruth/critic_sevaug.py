"""在 critic 上实际验证 sev 过滤（sev 训好后要在 critic 上实际验证）。

9B 错误改写模型（gen-core-9b）在 D4 训练池提示上的合格样本（outputs/rt/core/samples_9b_audit.jsonl）按某个过滤规则挑错句，
每条错句配同一位置的原句（改对后的回答，忠实 A），加进基线 critic 的训练集；校准 / 测试（RAGTruth++）沿用基线。
标签用生成器声称的类型（B 矛盾 / C 无依据）；--claude_type 时有 Claude 审计结论的用 Claude 的 true_type。

过滤规则（--filter）：
  sev2   新 sev__gen 的 P(A–D) ≥ --thr（gen_audit/sev2_scores.jsonl）
  sev1   旧 sev__core 的 P(A–D) ≥ --thr（gen_audit/sev_scores.jsonl）
  none   不过滤，随机取（数量用 --n_match 对齐到某一组）
  claude 只取 Claude 审计判为真错的（gen_audit labels_pair / labels_sevnew 里 9B 的部分）：上限参照

  python -m ragtruth.critic_sevaug --filter sev2 --thr 0.9 --name rm1__core_s2
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.critic_harvest import row
from ragtruth.rm_data import OUT, RM1_OPT

D = Path("outputs/rt/core")
G = D / "gen_audit"


def p_err(p: dict | None) -> float:
    return sum((p or {}).get(k, 0.0) for k in "ABCD")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="rm1__core")
    ap.add_argument("--name", required=True)
    ap.add_argument("--filter", required=True, choices=["sev2", "sev1", "none", "claude"])
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--n_match", type=int, default=0, help="最多加几条错句（0 = 不限）；none 组用它和过滤组对齐数量")
    ap.add_argument("--per_prompt", type=int, default=2)
    ap.add_argument("--claude_type", action="store_true")
    ap.add_argument("--exclude_labeled", action="store_true",
                    help="候选池去掉有 Claude 审计结论的 9B 样本（其中约 860 条进了 sev__gen 训练集，新 sev 在它们上等于记住了答案）")
    ap.add_argument("--rank", action="store_true", help="sev1 / sev2：不用阈值，按分数从高到低取 --n_match 条（和别的组对齐数量）")
    a = ap.parse_args()
    xs = {f"9b/{x['id']}": x for x in read_jsonl(D / "samples_9b_audit.jsonl") if x.get("why") == "ok"}
    claude = {}
    for f in ("pair", "sevnew"):
        if (G / f"labels_{f}.jsonl").exists():
            for r in read_jsonl(G / f"labels_{f}.jsonl"):
                claude[r["aid"]] = r["claude"]
    pool = [k for k in xs if not (a.exclude_labeled and k in claude)]
    rng = random.Random(f"critic_sevaug|{a.name}")
    if a.filter in ("sev2", "sev1"):
        sc = {r["aid"]: r["p_sev"] for r in read_jsonl(G / ("sev2_scores.jsonl" if a.filter == "sev2" else "sev_scores.jsonl"))}
        if a.rank:
            ok = sorted(pool, key=lambda k: (-p_err(sc.get(k)), k))
        else:
            ok = sorted(k for k in pool if p_err(sc.get(k)) >= a.thr)
            rng.shuffle(ok)
    elif a.filter == "claude":
        ok = sorted(k for k in pool if claude.get(k, {}).get("severity") in ("clear", "minor"))
        rng.shuffle(ok)
    else:
        ok = sorted(pool)
        rng.shuffle(ok)
    keep, per, seen = [], collections.Counter(), set()
    for k in ok:
        x = xs[k]
        if x["sentence"] in seen or per[x["prompt_id"]] >= a.per_prompt:
            continue
        seen.add(x["sentence"])
        per[x["prompt_id"]] += 1
        keep.append(k)
        if a.n_match and len(keep) >= a.n_match:
            break
    add, types = [], collections.Counter()
    for k in keep:
        x = xs[k]
        t = x["type"]
        if a.claude_type and claude.get(k, {}).get("true_type") in ("conflict", "baseless"):
            t = claude[k]["true_type"]
        types[t] += 1
        ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
        orig = x["sentences_orig"] if isinstance(x["sentences_orig"], list) else json.loads(x["sentences_orig"])
        g = "gh-" + str(x["source_id"])
        add.append(row(x["id"] + "-err", x["source"], ctx, int(x["idx"]), RM1_OPT[t], x["task_type"], g, "sevaug_error"))
        add.append(row(x["id"] + "-orig", x["source"], orig, int(x["idx"]), RM1_OPT["faithful"], x["task_type"], g, "sevaug_original"))
    # 已知真伪的（有 Claude 结论的）里，加进去的错句实际有多少是真错：过滤精度的旁证
    lab = [claude[k].get("severity") in ("clear", "minor") for k in keep if k in claude]
    base, d = OUT / a.base, OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    train = list(read_jsonl(base / "train.jsonl")) + add
    random.Random(f"{a.name}|shuffle").shuffle(train)
    with open(d / "train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "filter": a.filter, "thr": a.thr, "n_match": a.n_match,
            "rank": a.rank, "exclude_labeled": a.exclude_labeled, "pool": len(pool), "passed": len(ok), "added_errors": len(keep), "added_by_type": dict(types),
            "added_with_claude": len(lab), "added_claude_real": sum(lab),
            "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CRITIC_SEVAUG", json.dumps({k: meta[k] for k in ("filter", "thr", "pool", "passed", "added_errors", "added_by_type", "added_with_claude", "added_claude_real", "train")},
                                      ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
