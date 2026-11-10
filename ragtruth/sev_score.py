"""用严重程度 RM（sev__mc）给 harvest 造数据打分，挑下一版 critic 的候选（推理机空着别浪费）。
只打过了 RM3 门槛（p_correct ≥ 0.7）、原句干净、有 2B critic 分（p_halluc）的样本；prompt 和 sev RM 训练时同格式（sev_rm.prompt：改后回答）。
输出每条的 6 个选项概率 + harvest 关键字段；summary 给 P(显然) 门槛 × critic 区间的条数。

  VLLM_API_KEY=... python -m ragtruth.sev_score --url http://127.0.0.1:6006/v1 --model sev --temp 3.15 --out outputs/rt/gentarget/sev_h1.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

from common.schema import read_jsonl
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.sev_rm import prompt

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default="outputs/rt/gentarget/harvest_h1.jsonl")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="sev")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--priority", type=int, default=10, help="vLLM priority 调度下数值越大越靠后，不挡同机的训练推理")
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.url, api_key=os.environ.get("VLLM_API_KEY", "x"), timeout=600)
    rows = [x for x in read_jsonl(a.harvest) if x.get("why") == "ok" and x.get("p_halluc") is not None and float(x["p_correct"]) >= 0.7]

    def run(x: dict) -> dict:
        r = cl.chat.completions.create(model=a.model, messages=prompt(x), max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20,
                                       extra_body={**NO_THINK, "priority": a.priority})
        lp = option_logprobs(r.choices[0], "ABCDEF")
        p = probs(lp, a.temp) if lp else None
        return {"id": x["id"], "p": p, "p_clear": (p["A"] + p["B"]) if p else None, "p_err": (p["A"] + p["B"] + p["C"] + p["D"]) if p else None,
                "p_halluc": float(x["p_halluc"]), "p_correct": float(x["p_correct"]), "type": x["type"], "tactic_raw": x.get("tactic_raw"),
                "source_id": x["source_id"], "task_type": x["task_type"]}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(run, rows))
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ok = [o for o in out if o["p_clear"] is not None]
    summ = collections.OrderedDict()
    for t in (0.4, 0.5, 0.6):
        for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)):
            summ[f"p_clear>={t} critic {lo}-{hi}"] = sum(o["p_clear"] >= t and lo <= o["p_halluc"] < hi for o in ok)
    print("SEV_SCORE", json.dumps({"scored": len(ok), "failed": len(out) - len(ok), **summ}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
