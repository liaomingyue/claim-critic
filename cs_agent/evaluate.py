"""在 τ²-bench retail 上评测一个策略：每个任务跑 k 次，算 pass^1 和 pass^k（τ² 的定义：同一任务 k 次全对的概率，按 C(c,k)/C(n,k) 估）。
策略和用户模拟器都是 vLLM 服务（OpenAI 接口）。结果奖励只看数据库比对；NL 断言要 LLM 裁判，这里不算（分数和官方榜单不可比，见 docs）。
  python -m cs_agent.evaluate --policy_url http://127.0.0.1:8000/v1 --policy policy_v0 --tok <分词器> \
      --user_url http://127.0.0.1:8001/v1 --split test --k 4 --out outputs/rt/cs/base/test.jsonl
已有的结果文件会续跑（按 任务 × 第几次 跳过已完成的）。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cs_agent  # noqa: F401
from cs_agent import rollout as R


def pass_hat(ok: list[int], k: int) -> float:
    n, c = len(ok), sum(ok)
    return math.comb(c, k) / math.comb(n, k) if n >= k else float("nan")


def summarize(rows: list[dict], k: int) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(int(r["reward"] == 1.0))
    ks = sorted({1, k})
    out = {f"pass^{j}": round(sum(pass_hat(v, j) for v in by.values()) / max(1, len(by)), 4) for j in ks}
    out.update({"tasks": len(by), "episodes": len(rows), "terms": dict(collections.Counter(r["term"] for r in rows)),
                "avg_steps": round(sum(r["n_steps"] for r in rows) / max(1, len(rows)), 2),
                "avg_tokens": round(sum(r["n_tokens"] for r in rows) / max(1, len(rows))),
                "tool_errors_per_ep": round(sum(r.get("n_tool_errors", 0) for r in rows) / max(1, len(rows)), 2),
                "transfer_rate": round(sum(any(c["name"] == "transfer_to_human_agents" for t in r["turns"] for c in t["calls"]) for r in rows) / max(1, len(rows)), 4)})
    return out


def main() -> None:
    from transformers import AutoTokenizer
    from train.gen_grpo import Server
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_url", required=True)
    ap.add_argument("--policy", default="policy_v0")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--user_url", required=True)
    ap.add_argument("--user_model", default="base")
    ap.add_argument("--user_temp", type=float, default=0.0)   # τ² 默认 0
    ap.add_argument("--temp", type=float, default=0.6)        # 策略温度：pass^k 要有随机性；τ² 默认 0，这里写进报告
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--gen_max_tokens", type=int, default=4096)   # 一轮回复的上限：9B 会把几十个款式逐条列给用户，1024 截断了约一成的对话（实测）
    ap.add_argument("--max_ctx", type=int, default=32000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tok)
    tasks = R.load_tasks(a.split)
    tasks = tasks[: a.limit] if a.limit else tasks
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["task_id"], r["trial"]))
    server = Server(a.policy_url)
    sample = R.vllm_sampler(server, a.policy, a.temp)
    key = os.environ.get("USER_API_KEY", "x")
    # Qwen3.5 经 vLLM 的聊天接口默认开思考，模拟器会把思考过程当成用户发言（实测第一句就是 "Thinking Process: ..."）：显式关掉
    user_args = {"api_base": a.user_url, "api_key": key, "temperature": a.user_temp,
                 "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    jobs = [(t, j) for t in tasks for j in range(a.k) if (t.id, j) not in done]
    print(f"EVAL {len(tasks)} tasks x {a.k}, {len(jobs)} to run", flush=True)

    def one(tj):
        t, j = tj
        ep = R.run_episode(tok, t, sample, user_llm=f"openai/{a.user_model}", user_args={**user_args, "seed": j}, gen_max_tokens=a.gen_max_tokens, max_ctx=a.max_ctx)
        ep["trial"] = j
        return ep
    with ThreadPoolExecutor(a.concurrency) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(one, x) for x in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                ep = fu.result()
            except Exception as e:  # noqa: BLE001
                print("EPISODE_FAILED", repr(e)[:300], flush=True)
                continue
            f.write(R.dump(ep) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} done", flush=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    s = summarize(rows, a.k)
    s.update({"gen_max_tokens": a.gen_max_tokens, "max_ctx": a.max_ctx, "policy": a.policy, "split": a.split, "temp": a.temp, "user_temp": a.user_temp, "reward": "DB only"})
    Path(str(out).replace(".jsonl", "_summary.json")).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print("EVAL_DONE", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
