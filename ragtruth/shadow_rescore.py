"""训完补打对照分（2026-11-02，用户：训练外的调用用 9B）：MiniCPM GRPO 训练中只用 2B critic，
评测样本（quick_samples_*_s{step}.jsonl）训完再用 9B 金标 critic（rm1__rm1_gold_9b）逐句打一遍，看策略是不是只在迎合 2B critic。
只打 2B 评测里打过分的回答（停止正常、非空、非拒答），和训练中的口径一致；每步对第 0 步做按原文配对的 bootstrap。

  python -m ragtruth.shadow_rescore --run outputs/rt/minicpm/grpo/run --dev outputs/rt/minicpm/grpo/dev_ref.jsonl \
      --url http://127.0.0.1:8000/v1,http://127.0.0.1:8101/v1 --model critic --temp 1.3
写 {run}/shadow9b.jsonl（每步一行汇总）和 {run}/shadow9b_s{step}.jsonl（逐条分数）。
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train.ans_grpo import paired_delta
from train.ans_reward import AnswerRewarder, sentences

STEP = re.compile(r"_s(\d+)\.jsonl$")


def rescore(rw: AnswerRewarder, dev: dict, path: Path, pool: ThreadPoolExecutor) -> list[dict]:
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]

    def one(l):
        src = dev[l["id"]]["source"]
        out = []
        for t, s in zip(l["outputs"], l["scores"]):
            out.append(rw.p_answer(src, sentences(t.strip())) if s.get("ps") else None)   # 2B 打过分的才打
        return {"id": l["id"], "task_type": l["task_type"], "ps2b": [s.get("ps") for s in l["scores"]], "ps9b": out}

    return list(pool.map(one, lines))


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        xs = [(a, b) for r in rs for a, b in zip(r["ps2b"], r["ps9b"]) if a and b]
        if not xs:
            return {}
        m = lambda f: round(sum(map(f, xs)) / len(xs), 4)
        return {"n": len(xs), "p_mean_2b": m(lambda x: sum(x[0]) / len(x[0])), "answer_flag_2b": m(lambda x: max(x[0]) >= .5),
                "p_mean_9b": m(lambda x: sum(x[1]) / len(x[1])), "answer_flag_9b": m(lambda x: max(x[1]) >= .5),
                "sent_flag_9b": round(sum(p >= .5 for x in xs for p in x[1]) / sum(len(x[1]) for x in xs), 4)}
    return {**agg(rows), **{t: agg([r for r in rows if r["task_type"] == t]) for t in ("QA", "Summary")}}


def by_source(rows: list[dict], key: str, f) -> dict:
    out = {}
    for r in rows:
        v = [f(p) for p in r[key] if p]
        if v:
            out[r["id"]] = sum(v) / len(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--url", required=True, help="逗号分隔 = 几个同样的服务，按原文分摊")
    ap.add_argument("--model", default="critic")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    from openai import OpenAI
    run = Path(a.run)
    dev = {r["id"]: r for r in map(json.loads, open(a.dev, encoding="utf-8"))}
    pool, inner = ThreadPoolExecutor(a.workers), ThreadPoolExecutor(a.workers)
    rw = AnswerRewarder([OpenAI(base_url=u, api_key="x", timeout=600, max_retries=3) for u in a.url.split(",")], a.model, a.temp, inner)
    files = sorted(run.glob("quick_samples_*_s*.jsonl"), key=lambda p: int(STEP.search(p.name).group(1)))
    first, summ = None, open(run / "shadow9b.jsonl", "w", encoding="utf-8")
    for p in files:
        step = int(STEP.search(p.name).group(1))
        rows = rescore(rw, dev, p, pool)
        with open(run / f"shadow9b_s{step}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        flags = {k: by_source(rows, k, lambda ps: float(max(ps) >= .5)) for k in ("ps2b", "ps9b")}
        if first is None:
            first = flags
        rec = {"step": step, **summarize(rows),
               "delta_ci_9b": paired_delta(flags["ps9b"], first["ps9b"]), "delta_ci_2b": paired_delta(flags["ps2b"], first["ps2b"])}
        summ.write(json.dumps(rec, ensure_ascii=False) + "\n"); summ.flush()
        print("SHADOW9B", json.dumps(rec, ensure_ascii=False), flush=True)
    print("SHADOW9B_DONE", flush=True)


if __name__ == "__main__":
    main()
