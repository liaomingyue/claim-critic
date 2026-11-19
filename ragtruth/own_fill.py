"""用 RAG 模型自己写的、逐句核实过的句子补 critic（合成样本之外的另一个来源）。

验收原文按任务类型对半分成两份（configs/rt/accept_split.json）：fit 这一半的句子可以进 critic 训练，final 这一半只做验收。
加进训练的两种句子（都来自 fit 这一半）：
  错句：核实为错的句子，标签按冲突 / 无依据；
  难负例：核实为对、但基线 critic 给分不低（默认 ≥ 0.3）的句子，标签为忠实。
两组：e = 只加错句；en = 错句 + 难负例。

  python -m ragtruth.own_fill split                                  # 写 configs/rt/accept_split.json（已有就不动）
  python -m ragtruth.own_fill rows  --arms cont8 cont9               # 本地：从标注和回答拼出要加的句子 → outputs/rt/own_fill/<round>/add_{e,en}.jsonl
  python -m ragtruth.own_fill build --base rm1__core_r2_bfcl         # 远程：基线训练集 + 要加的句子 → outputs/rt/rm/<name>{e,en}
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
from pathlib import Path

from common.schema import read_jsonl

SPLIT = Path("configs/rt/accept_split.json")
CORE = Path("outputs/rt/core")
O = Path("outputs/rt/own_fill")
CID = re.compile(r"^[a-z0-9]+-([abc])(.+)-S(\d+)$")   # 标注 id：<版本>-<套><原文>-<种子>-S<句号>


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_split(tasks: dict[str, str], seed: str = "accept_split|v1") -> dict[str, list[str]]:
    """tasks：原文 id → 任务类型。每种任务类型各自打乱后对半分，前一半 fit、后一半 final。"""
    out = {"fit": [], "final": []}
    for t in sorted(set(tasks.values())):
        ids = sorted(k for k, v in tasks.items() if v == t)
        random.Random(f"{seed}|{t}").shuffle(ids)
        out["fit"] += ids[: len(ids) // 2]
        out["final"] += ids[len(ids) // 2:]
    return {k: sorted(v) for k, v in out.items()}


def load_split() -> dict[str, set[str]]:
    return {k: set(v) for k, v in json.load(open(SPLIT, encoding="utf-8")).items()}


def labelled(arm: str, ver: str, critic: str) -> list[dict]:
    """一组、一个版本：标注 + 回答（原文、全部句子）+ 基线 critic 的分数。ver：v0 / s100。"""
    lab = list(read_jsonl(CORE / f"claude_check_{arm}" / f"final_{'start' if ver == 'v0' else ver}.jsonl"))
    ans, p = {}, {}
    for s in "abc":
        for r in read_jsonl(CORE / f"loop2_{arm}" / "eval" / f"{ver}_{s}.jsonl"):
            ans[(s, r["id"].split("-", 1)[1])] = r
        f = CORE / f"loop2_{arm}" / "eval" / f"{ver}_{s}_critic_{critic}.json"
        if f.exists():
            for r in json.load(open(f, encoding="utf-8")):
                p[(s, r["id"].split("-", 1)[1])] = r["p"]
    out = []
    for r in lab:
        m = CID.match(r["id"])
        if not m or (m.group(1), m.group(2)) not in ans:
            continue
        a = ans[(m.group(1), m.group(2))]
        i = int(m.group(3))
        if i > len(a["sentences"]) or a["sentences"][i - 1].strip() != r["sentence"].strip():
            continue
        out.append({**r, "source": a["source"], "sentences": a["sentences"], "p_base": p.get((m.group(1), f"{m.group(2)}-S{i}"))})
    return out


def cmd_split(a) -> None:
    if SPLIT.exists():
        sp = load_split()
    else:
        tasks = {}
        for arm in a.arms:
            for r in read_jsonl(CORE / f"claude_check_{arm}" / "final_s100.jsonl"):
                tasks[str(r["source_id"])] = r["task_type"]
        sp = make_split(tasks)
        SPLIT.write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
        sp = {k: set(v) for k, v in sp.items()}
    print("OWN_FILL_SPLIT", json.dumps({k: len(v) for k, v in sp.items()}))


def cmd_rows(a) -> None:
    from ragtruth.critic_harvest import row
    from ragtruth.rm_data import RM1_OPT
    fit = load_split()["fit"]
    seen, err, neg, stat = set(), [], [], collections.Counter()
    for arm in a.arms:
        for ver in ("v0", "s100"):
            for r in labelled(arm, ver, a.critic):
                key = (str(r["source_id"]), tuple(r["sentences"]), r["idx"])
                if str(r["source_id"]) not in fit or key in seen or "unable to answer" in r["sentence"].lower():
                    continue
                seen.add(key)
                rid = f"own-{arm}-{r['id']}"
                if r["is_error"] and r["type"] in ("conflict", "baseless"):
                    err.append(row(rid + "-err", r["source"], r["sentences"], r["idx"], RM1_OPT[r["type"]], r["task_type"], "own-" + str(r["source_id"]), "own_fill_error"))
                    stat[f"err_{r['severity']}"] += 1
                    stat[f"err_base_p<{a.neg_p}"] += r["p_base"] is not None and r["p_base"] < a.neg_p
                elif not r["is_error"] and r["p_base"] is not None and r["p_base"] >= a.neg_p:
                    neg.append(row(rid + "-ok", r["source"], r["sentences"], r["idx"], RM1_OPT["faithful"], r["task_type"], "own-" + str(r["source_id"]), "own_fill_hard_negative"))
                    stat["neg"] += 1
                    stat["neg_p>=0.5"] += r["p_base"] >= 0.5
    B = O / a.round
    write(B / "add_e.jsonl", err)
    write(B / "add_en.jsonl", err + neg)
    print("OWN_FILL_ROWS", json.dumps({"fit_sources": len(fit), "sentences_seen": len(seen), "errors": len(err), "hard_negatives": len(neg), **dict(sorted(stat.items()))}))


def cmd_build(a) -> None:
    from ragtruth.rm_data import OUT
    base = OUT / a.base
    held = {r["group"] for sp in ("calib.jsonl", "test.jsonl") for r in read_jsonl(base / sp) if "group" in r}
    trained = {str(r["group"]) for r in read_jsonl(base / "train.jsonl") if "group" in r}
    final = load_split()["final"]
    stat = {}
    for arm in a.arms:
        add = list(read_jsonl(O / a.round / f"add_{arm}.jsonl"))
        src = {r["group"].split("-", 1)[1] for r in add}
        assert not src & final, f"{len(src & final)} added sources are in the final half"
        assert not {r["group"] for r in add} & held, "added groups collide with calib / test"
        d = OUT / f"{a.name}{arm}"
        d.mkdir(parents=True, exist_ok=True)
        train = list(read_jsonl(base / "train.jsonl")) + add
        random.Random(f"own_fill|{a.round}|{arm}|shuffle").shuffle(train)
        write(d / "train.jsonl", train)
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copy(base / sp, d / sp)
        meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "arm": arm, "added": len(add),
                "added_label": dict(sorted(collections.Counter(r["label"] for r in add).items())), "added_sources": len(src),
                "added_sources_already_in_base_train": len(src & trained),
                "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        stat[arm] = {"added": len(add), "train": len(train)}
    print("OWN_FILL_BUILD", json.dumps(stat), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--arms", nargs="+", default=["cont8", "cont9"])
    s.set_defaults(fn=cmd_split)
    s = sub.add_parser("rows")
    s.add_argument("--arms", nargs="+", default=["cont8", "cont9"])
    s.add_argument("--critic", default="rm1__core_r2_bfcl", help="用哪个 critic 的分数挑难负例")
    s.add_argument("--neg_p", type=float, default=0.3)
    s.add_argument("--round", default="own1")
    s.set_defaults(fn=cmd_rows)
    s = sub.add_parser("build")
    s.add_argument("--base", default="rm1__core_r2_bfcl")
    s.add_argument("--name", default="rm1__core_own1")
    s.add_argument("--round", default="own1")
    s.add_argument("--arms", nargs="+", default=["e", "en"])
    s.set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
