"""关键对照的 Claude 独立验证：两组 RAG 回答（同一批验收原文和种子，配对），Claude 按金标 v1 逐句盲标，
和 9B 评测器的结论对照。用来检查「9B 策略 × 9B critic 训练、9B 评测器打分」会不会被同源的盲区抬高。

  python -m ragtruth.claude_pair_check label --arm q9b_s200=<目录>:<前缀> --arm final_2b=<目录>:<前缀> [--sets a b]
  python -m ragtruth.claude_pair_check report --arm ... --base final_2b

- 一条回答一次调用（批量标注会改变标注口径，见实验记录 1.5），两组回答打乱后一起送，Claude 看不到回答来自哪个模型；
- 提示词和格式检查与验收金标相同（ragtruth.gold.label），结果可续跑；
- 调用上限 --max_calls（默认 400）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.ans_accept import Paced, answer_flags, compare
from ragtruth.gold import label, write_final

D = Path("outputs/rt/core/claude_check")


def load_arms(specs: list[str], sets: list[str]) -> dict[str, dict[str, dict]]:
    """{组: {回答 id: 回答}}；回答 id 改成 <组>-<套><原文>-<种子>，配对键（answer_flags 取第一个 - 之后）跨组一致、跨套不撞。"""
    arms = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        rows = {}
        for s in sets:
            for r in read_jsonl(Path(d) / f"{prefix}_{s}.jsonl"):
                if r.get("finish") != "stop" or not r["sentences"]:
                    continue
                src_seed = r["id"].split("-", 1)[1]
                rid = f"{name}-{s}{src_seed}"
                rows[rid] = {**r, "id": rid, "orig_id": r["id"], "set": s}
        arms[name] = rows
    return arms


def paired_keys(arms: dict[str, dict]) -> list[str]:
    ks = [{rid.split("-", 1)[1] for rid in rows} for rows in arms.values()]
    return sorted(set.intersection(*ks))


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig
    arms = load_arms(a.arm, a.sets)
    keys = set(paired_keys(arms))
    items = [{"id": f"{rid}#p1", "r": r} for rows in arms.values() for rid, r in rows.items() if rid.split("-", 1)[1] in keys]
    random.Random(a.seed).shuffle(items)   # 两组交错送，Claude 看不出顺序
    assert len(items) <= a.max_calls, f"Claude 调用预算：{len(items)} > {a.max_calls}"
    D.mkdir(parents=True, exist_ok=True)
    llm = Paced(LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency)), a.gap)
    print("LABEL", len(items), "answers,", len(keys), "pairs", flush=True)
    label(llm, items, D / "labels.jsonl", a.concurrency)
    by = collections.defaultdict(dict)
    for x in read_jsonl(D / "labels.jsonl"):
        if not x.get("_dropped"):
            by[x["resp_id"]][1] = x["sentences"]
    for name, rows in arms.items():
        picks = {rid: r for rid, r in rows.items() if rid in by}
        write_final(picks, {rid: by[rid] for rid in picks}, D / f"final_{name}.jsonl")
    print("LABEL_DONE", len(by), "of", len(items), flush=True)


def nine_b_flags(arms: dict[str, dict], specs: list[str]) -> dict[str, dict[str, dict]]:
    """9B 评测器在同一批回答上的口径：一句 P ≥ 0.9 记含显然错，P ≥ 0.5 记含任意错。"""
    out = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        p = {}
        for s in {r["set"] for r in arms[name].values()}:
            for x in json.load(open(Path(d) / f"{prefix}_{s}_9b.json", encoding="utf-8")):
                rid, k = x["id"].rsplit("-S", 1)
                p.setdefault(f"{s}{rid.split('-', 1)[1]}", []).append(x["p"])
        out[name] = {k: {"any_clear": float(max(v) >= 0.9), "any_error": float(max(v) >= 0.5), "err_rate": sum(q >= 0.5 for q in v) / len(v)} for k, v in p.items()}
    return out


def kappa(a: list[float], b: list[float]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def cmd_report(a) -> None:
    arms = load_arms(a.arm, a.sets)
    cl = {name: answer_flags(list(read_jsonl(D / f"final_{name}.jsonl"))) for name in arms}
    nb = nine_b_flags(arms, a.arm)
    keys = sorted(set.intersection(*[set(v) for v in cl.values()]))
    src = {rid.split("-", 1)[1]: r["source_id"] for rows in arms.values() for rid, r in rows.items()}
    refuse = {name: round(sum("unable to answer" in rows[f"{name}-{k}"]["response"].lower() for k in keys) / len(keys), 4) for name, rows in arms.items()}
    rep = {"pairs": len(keys), "refusal": refuse, "claude": {}, "9b": {}, "agreement": {}}
    for name in arms:
        if name == a.base:
            continue
        rep["claude"][f"{name} - {a.base}"] = compare({k: cl[a.base][k] for k in keys}, {k: cl[name][k] for k in keys}, src)
        rep["9b"][f"{name} - {a.base}"] = compare({k: nb[a.base][k] for k in keys}, {k: nb[name][k] for k in keys}, src)
    for name in arms:
        rep["agreement"][name] = {f: {"claude": round(sum(cl[name][k][f] for k in keys) / len(keys), 4), "9b": round(sum(nb[name][k][f] for k in keys) / len(keys), 4),
                                      "kappa": kappa([cl[name][k][f] for k in keys], [nb[name][k][f] for k in keys])} for f in ("any_clear", "any_error")}
    out = D / "report.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "report"])
    ap.add_argument("--arm", action="append", required=True, help="名=目录:前缀")
    ap.add_argument("--sets", nargs="+", default=["a", "b"])
    ap.add_argument("--base", default="")
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--max_calls", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_label(a) if a.cmd == "label" else cmd_report(a)


if __name__ == "__main__":
    main()
