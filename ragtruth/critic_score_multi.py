"""多个 critic × 多个回答集一起打分（2026-11-05，用户：16 个 critic 评测 GPU 利用率才 33%，要打满）。
打分口径和 ragtruth.critic_score 完全一样（同 prompt、max_tokens 8、温度校准、P = B + C、同输出格式），只改调度：
  - asyncio 一个进程挂几百个请求（原来 8 条回答 × 8 线程，最多 64 个，GPU 等客户端）；
  - 所有任务的请求混在一起发，vLLM 一个批次里同时处理多个 LoRA（原来一次只打一个 critic，其余 LoRA 槽空着）；
  - 同一条回答仍然先打第 1 句、再并发其余句（同一 LoRA 下共享前缀缓存）。
任务文件（JSON 列表）：[{"model": "c0", "temp": 1.15, "answers": ".../answers.jsonl", "out": ".../critic_x.json"}, ...]；out 已存在的跳过。

  python -m ragtruth.critic_score_multi --url http://127.0.0.1:8000/v1 --jobs jobs.json --concurrency 384
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.task import render_input
from train.ans_reward import NO_THINK


def usable(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("sentences") and r.get("finish") == "stop"]


def to_p(choice, temp: float) -> dict:
    lp = option_logprobs(choice, "ABC")
    p = probs(lp, temp) if lp else {"A": 0.0, "B": 1.0, "C": 0.0}
    return {"p": p["B"] + p["C"], "p_conflict": p["B"], "p_baseless": p["C"]}


async def score_all(client, jobs: list[dict], concurrency: int, log_every: float = 30.0) -> dict[str, list[dict]]:
    sem = asyncio.Semaphore(concurrency)
    done = {"req": 0}
    t0 = time.time()

    async def one(job: dict, src: str, sents: list[str], i: int) -> dict:
        msgs = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": src, "sentences": sents, "idx": i})}]
        async with sem:
            r = await client.chat.completions.create(model=job["model"], messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20,
                                                     extra_body=NO_THINK["extra_body"])
        done["req"] += 1
        return to_p(r.choices[0], job["temp"])

    async def resp(job: dict, r: dict) -> list[dict]:
        first = await one(job, r["source"], r["sentences"], 1)   # 先打第一句（前缀进缓存），再并发其余
        rest = await asyncio.gather(*(one(job, r["source"], r["sentences"], i) for i in range(2, len(r["sentences"]) + 1)))
        return [{"id": f"{r['id']}-S{i + 1}", **x} for i, x in enumerate([first, *rest])]

    async def run_job(job: dict) -> list[dict]:
        groups = await asyncio.gather(*(resp(job, r) for r in job["rows"]))
        return [x for g in groups for x in g]

    async def ticker(total: int) -> None:
        while True:
            await asyncio.sleep(log_every)
            print(f"[multi] {done['req']}/{total} requests, {done['req'] / max(1e-6, time.time() - t0):.0f}/s", flush=True)

    total = sum(len(r["sentences"]) for j in jobs for r in j["rows"])
    tick = asyncio.create_task(ticker(total))
    try:
        res = await asyncio.gather(*(run_job(j) for j in jobs))
    finally:
        tick.cancel()
    return {j["out"]: out for j, out in zip(jobs, res)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--concurrency", type=int, default=384)
    a = ap.parse_args()
    jobs = [j for j in json.load(open(a.jobs, encoding="utf-8")) if not Path(j["out"]).exists()]
    cache: dict[str, list[dict]] = {}
    for j in jobs:
        j["rows"] = cache.setdefault(j["answers"], usable(j["answers"]))
    if not jobs:
        print("MULTI_SCORED nothing to do", flush=True)
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.url, api_key="x", timeout=900, max_retries=3)
    t0 = time.time()
    res = asyncio.run(score_all(client, jobs, a.concurrency))
    for j in jobs:
        Path(j["out"]).write_text(json.dumps(res[j["out"]]), encoding="utf-8")
        print("SCORED", j["out"], len(j["rows"]), "answers", len(res[j["out"]]), "sentences", flush=True)
    n = sum(len(v) for v in res.values())
    print(f"MULTI_SCORED {len(jobs)} jobs, {n} sentences in {time.time() - t0:.0f}s ({n / max(1e-6, time.time() - t0):.0f}/s)", flush=True)


if __name__ == "__main__":
    main()
