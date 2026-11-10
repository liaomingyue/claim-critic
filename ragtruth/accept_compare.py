"""四组 Claude 验收对比：Qwen3.5-9B 原始 / 9B GRPO（第二轮，满意度护栏）/ MiniCPM5-2B 基线（关闭思考 + QA 系统提示）/ 2B GRPO 第 300 步。
四组都是同样 100 个（原文, 种子），金标 v1 单轮标注。配对 bootstrap 按原文抽样（ragtruth.ans_accept.compare）。

  python -m ragtruth.accept_compare      # 写 outputs/rt/accept_compare.json
"""
from __future__ import annotations

import json
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.ans_accept import answer_flags, compare

ARMS = {"9b_base": "outputs/rt/base9b", "9b_grpo": "outputs/rt/ansgrpo/accept_sat",
        "2b_base": "outputs/rt/minicpm/sys", "2b_grpo": "outputs/rt/minicpm/grpo300"}
PAIRS = [("9b_base", "9b_grpo"), ("2b_base", "2b_grpo"), ("9b_base", "2b_grpo"), ("9b_grpo", "2b_grpo")]
key = lambda rid: rid.split("-", 1)[1]


def refusals(d: str) -> dict:
    out = {}
    for t in ("all", "QA", "Summary"):
        keep = {key(x["resp_id"]) for x in read_jsonl(Path(d) / "final.jsonl")}   # base9b 的 answers.jsonl 比验收的 100 条多
        ys = [r for r in read_jsonl(Path(d) / "answers.jsonl") if key(r["id"]) in keep and (t == "all" or r["task_type"] == t)]
        out[t] = round(sum("unable to answer" in r["response"].lower() for r in ys) / len(ys), 4)
    return out


def main() -> None:
    rows = {a: list(read_jsonl(Path(d) / "final.jsonl")) for a, d in ARMS.items()}
    src = {key(x["resp_id"]): x["source_id"] for x in rows["9b_base"]}
    rep = {"arms": {}, "pairs": {}}
    for a in ARMS:
        rep["arms"][a] = {"refusal": refusals(ARMS[a])}
        for t in ("all", "QA", "Summary"):
            f = answer_flags([x for x in rows[a] if t == "all" or x["task_type"] == t])
            rep["arms"][a][t] = {m: round(sum(v[m] for v in f.values()) / len(f), 4) for m in ("any_error", "any_clear", "err_rate")} | {"answers": len(f)}
    for b, c in PAIRS:
        rep["pairs"][f"{b}->{c}"] = {t: compare(answer_flags([x for x in rows[b] if t == "all" or x["task_type"] == t]),
                                                answer_flags([x for x in rows[c] if t == "all" or x["task_type"] == t]), src)
                                     for t in ("all", "QA", "Summary")}
    Path("outputs/rt/accept_compare.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for a, v in rep["arms"].items():
        print(a, {t: v[t] for t in ("all", "QA", "Summary")}, "refusal", v["refusal"])
    for k, v in rep["pairs"].items():
        print(k, {t: {m: (v[t][m]["delta"], v[t][m]["ci"]) for m in ("any_error", "any_clear")} for t in ("all", "QA", "Summary")})


if __name__ == "__main__":
    main()
