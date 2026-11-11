"""合适度（错误类型先验）数据：只看原文和问题，预测回答里可能出现哪些错误机制（多标签，8 个机制各一个独立分数）。

输入不含任何回答；标签来自已有 Claude 金标里回答出现过的机制：
  - train / holdout：核心集（726 / 81 个原文，每个原文 1 条回答），机制出现记 1、没出现记 0；
  - test：金标测试集（68 个原文，每个原文约 6 个模型的回答），每个机制的软标签 = 出现该机制的回答占比，另给「至少一条出现」；
  - policy：我们自己策略输出的 Claude 复核（2B 整套方案 / 9B GRPO，各 100 条），看能不能迁移到自己的策略上。
三份之间原文不重叠（核心集和测试集原文互斥；policy 用验收原文）。

  python -m ragtruth.suit_data build       # → outputs/rt/suit/{train,holdout,test,policy}.jsonl
  python -m ragtruth.suit_data baselines   # 常数 / 任务类型 / 原文表面特征三个基线 → results/suit_baselines.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from common.schema import read_jsonl

# 「添加具体事实」和「无依据推论」合并成「无依据的添加」：核心集（训练）和金标 v1（测试）在这两类之间划分口径不同
# （添加具体事实：核心集每条回答 4.7%，测试集 19.4%），分开训会系统性低估；「other」太少、「self_description」核心集里没有，都不当选项
MERGE = {"added_specific_fact": "unsupported_addition", "unsupported_inference": "unsupported_addition"}
MECHS = ["certainty_or_scope_shift", "near_synonym_shift", "contradicts_source", "unsupported_addition", "wrong_attribution",
         "vague_evaluation_or_addon", "number_or_date", "merged_across_passages"]


def mech(m: str) -> str:
    return MERGE.get(m, m)
RT = Path("external_data/ragtruth/dataset")
OUT = Path("outputs/rt/suit")


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def from_core(core: list[dict], split: str) -> list[dict]:
    out = []
    for c in core:
        if c["split"] != split:
            continue
        ms = {mech(e["mechanism"]) for e in c["labels"]["errors"]}
        out.append({"id": f"core-{c['source_id']}", "source_id": c["source_id"], "task_type": c["task_type"], "source": c["source"],
                    "labels": {m: int(m in ms) for m in MECHS}, "n_resp": 1})
    return sorted(out, key=lambda r: r["id"])


def from_sentence_labels(rows: list[dict], prompts: dict[str, str], tag: str) -> list[dict]:
    """逐句金标 → 每个原文：各机制出现的回答占比（soft）和是否至少一条出现（any）。"""
    resp = collections.defaultdict(set)
    meta = {}
    for r in rows:
        resp[(r["source_id"], r["resp_id"])]   # 没有错的回答也要算进分母
        meta[r["source_id"]] = r["task_type"]
        if r.get("is_error") and mech(r.get("mechanism", "")) in MECHS:
            resp[(r["source_id"], r["resp_id"])].add(mech(r["mechanism"]))
    by_src = collections.defaultdict(list)
    for (sid, _), ms in resp.items():
        by_src[sid].append(ms)
    out = []
    for sid in sorted(by_src):
        if sid not in prompts:
            continue
        rs = by_src[sid]
        out.append({"id": f"{tag}-{sid}", "source_id": sid, "task_type": meta[sid], "source": prompts[sid], "n_resp": len(rs),
                    "soft": {m: round(sum(m in x for x in rs) / len(rs), 4) for m in MECHS},
                    "labels": {m: int(any(m in x for x in rs)) for m in MECHS}})
    return out


def cmd_build(_a) -> None:
    prompts = {str(s["source_id"]): s["prompt"] for s in read_jsonl(RT / "source_info.jsonl")}
    core = list(read_jsonl(Path("outputs/rt/core/core.jsonl")))
    tr, ho = from_core(core, "train"), from_core(core, "holdout")
    te = from_sentence_labels(list(read_jsonl(Path("outputs/rt/gold/test_final.jsonl"))), prompts, "test")
    pol_rows = []
    for p in sorted(Path("results/q9b_claude_check").glob("final_*.jsonl")):
        pol_rows += list(read_jsonl(p))
    po = from_sentence_labels(pol_rows, prompts, "policy")
    core_src = {r["source_id"] for r in tr + ho}
    assert not core_src & {r["source_id"] for r in te}, "core / test 原文重叠"
    for name, rows in (("train", tr), ("holdout", ho), ("test", te), ("policy", po)):
        write(OUT / f"{name}.jsonl", rows)
    rep = {name: {"sources": len(rows), "responses": sum(r["n_resp"] for r in rows),
                  "pos_rate": {m: round(sum(r["labels"][m] for r in rows) / max(1, len(rows)), 3) for m in MECHS},
                  "policy_overlap_core": len({r["source_id"] for r in rows} & core_src) if name == "policy" else None}
           for name, rows in (("train", tr), ("holdout", ho), ("test", te), ("policy", po))}
    (OUT / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["sources"], v["responses"]) for k, v in rep.items()}))


NUM = re.compile(r"\d")
DATE = re.compile(r"\b(1[89]\d\d|20\d\d)\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")
QUOTE = re.compile(r"[\"“”]")
TOPN = (10, 20)   # 按类型排序取前 N（holdout 81、test 68、policy 91 个原文）
PASSAGE = re.compile(r"(?m)^passage \d+:", re.I)


def features(r: dict) -> list[float]:
    """原文表面特征：任务类型、长度、数字 / 日期 / 引号密度、段落数、专名（大写词）密度。"""
    s = r["source"]
    words = max(1, len(s.split()))
    caps = len(re.findall(r"\b[A-Z][a-z]+", s))
    return [float(r["task_type"] == "QA"), len(s) / 4000, len(NUM.findall(s)) / words, len(DATE.findall(s)) / words * 10,
            len(QUOTE.findall(s)) / words * 10, len(PASSAGE.findall(s)) / 3, caps / words]


def _ranks(xs: list[float]) -> list[float]:
    """平均秩（并列取平均），给 AUC 和 Spearman 用；不依赖 scipy / sklearn（远端环境没有 scipy）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def roc_auc_score(y: list[int], p: list[float]) -> float:
    """Mann-Whitney 形式的 AUC（并列算一半）。"""
    r = _ranks(p)
    npos = sum(y)
    nneg = len(y) - npos
    return (sum(ri for ri, yi in zip(r, y) if yi) - npos * (npos + 1) / 2) / (npos * nneg)


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def evaluate(pred: dict[str, list[float]], rows: list[dict]) -> dict:
    """逐机制 AUC（对 labels）、Brier；test 另算软标签的 Spearman 和 AUC 宏平均。"""
    res = {}
    for m in MECHS:
        y = [r["labels"][m] for r in rows]
        p = pred[m]
        d = {"pos": sum(y), "brier": round(sum((a - b) ** 2 for a, b in zip(p, y)) / len(y), 4)}
        d["auc"] = round(roc_auc_score(y, p), 3) if 0 < sum(y) < len(y) and len(set(p)) > 1 else None
        if "soft" in rows[0]:
            sv = [r["soft"][m] for r in rows]
            d["spearman_soft"] = round(spearman(p, sv), 3) if len(set(p)) > 1 and len(set(sv)) > 1 else None
        # 按这个类型排序取前 N 个原文（实际用法）：前 N 个里该类错的出现率 / 全体平均（lift）；有软标签用软标签，否则用 0/1
        tgt = [r["soft"][m] for r in rows] if "soft" in rows[0] else y
        base = sum(tgt) / len(tgt)
        order = sorted(range(len(rows)), key=lambda i: (-p[i], rows[i]["id"]))
        for n in TOPN:
            if n < len(rows) and len(set(p)) > 1:
                top = sum(tgt[i] for i in order[:n]) / n
                d[f"top{n}"] = round(top, 3)
                d[f"lift{n}"] = round(top / base, 2) if base > 0 else None
        d["base"] = round(base, 3)
        res[m] = d
    aucs = [v["auc"] for v in res.values() if v["auc"] is not None]
    sps = [v.get("spearman_soft") for v in res.values() if v.get("spearman_soft") is not None]
    lifts = {n: [v[f"lift{n}"] for v in res.values() if v.get(f"lift{n}") is not None] for n in TOPN}
    return {"macro_auc": round(sum(aucs) / len(aucs), 3) if aucs else None, "macro_spearman_soft": round(sum(sps) / len(sps), 3) if sps else None,
            "mean_brier": round(sum(v["brier"] for v in res.values()) / len(res), 4),
            **{f"mean_lift{n}": round(sum(v) / len(v), 2) for n, v in lifts.items() if v}, "by_mech": res}


def cmd_baselines(_a) -> None:
    from sklearn.linear_model import LogisticRegression
    data = {n: list(read_jsonl(OUT / f"{n}.jsonl")) for n in ("train", "holdout", "test", "policy")}
    tr = data["train"]
    preds = {}
    rate = {m: sum(r["labels"][m] for r in tr) / len(tr) for m in MECHS}
    preds["const"] = lambda rows: {m: [rate[m]] * len(rows) for m in MECHS}
    by_task = {t: {m: sum(r["labels"][m] for r in tr if r["task_type"] == t) / max(1, sum(r["task_type"] == t for r in tr)) for m in MECHS}
               for t in ("QA", "Summary")}
    preds["task"] = lambda rows: {m: [by_task[r["task_type"]][m] for r in rows] for m in MECHS}
    X = [features(r) for r in tr]
    lr = {m: LogisticRegression(C=1.0, max_iter=2000).fit(X, [r["labels"][m] for r in tr]) for m in MECHS}
    preds["features"] = lambda rows: {m: list(lr[m].predict_proba([features(r) for r in rows])[:, 1]) for m in MECHS}
    out = {b: {n: evaluate(f(rows), rows) for n, rows in data.items() if n != "train" and rows} for b, f in preds.items()}
    out["_note"] = "AUC 对「是否出现该机制」；test 的 spearman_soft 是预测分数和 6 条回答里出现比例的秩相关（每个机制在 68 个原文上）"
    Path("results").mkdir(exist_ok=True)
    Path("results/suit_baselines.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for b in preds:
        print(b, {n: (out[b][n]["macro_auc"], out[b][n]["macro_spearman_soft"], out[b][n].get("mean_lift10"), out[b][n].get("mean_lift20")) for n in out[b]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "baselines"])
    a = ap.parse_args()
    {"build": cmd_build, "baselines": cmd_baselines}[a.cmd](a)


if __name__ == "__main__":
    main()
