"""给验收回答逐句打 critic 分（2026-11-03，用户：看 critic 自己能不能判出 Claude 标的各类错误）。
和 outputs/rt/minicpm/sys/critic_scores.json 同格式：[{"id": "<回答 id>-S<句号>", "p": P(有错)}]，只打停止正常、有句子的回答。
另存 p_conflict / p_baseless（选项 B / C），重造 RM2 时给「自然」错误定类型用。

  python -m ragtruth.critic_score --answers outputs/rt/minicpm/grpo300/answers.jsonl --url http://127.0.0.1:8000/v1 --temp 1.3 \
      --out outputs/rt/minicpm/grpo300/critic_scores.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.task import render_input
from train.ans_reward import NO_THINK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="critic")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.url, api_key="x", timeout=900)

    def one(src: str, sents: list[str], i: int) -> dict:
        msgs = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": src, "sentences": sents, "idx": i})}]
        r = cl.chat.completions.create(model=a.model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], "ABC")
        p = probs(lp, a.temp) if lp else {"A": 0.0, "B": 1.0, "C": 0.0}
        return {"p": p["B"] + p["C"], "p_conflict": p["B"], "p_baseless": p["C"]}

    def resp(r: dict) -> list[dict]:   # 先打第一句再并发其余：同一回答的句子共享前缀缓存
        first = one(r["source"], r["sentences"], 1)
        with ThreadPoolExecutor(8) as ex:
            rest = list(ex.map(lambda i: one(r["source"], r["sentences"], i), range(2, len(r["sentences"]) + 1)))
        return [{"id": f"{r['id']}-S{i + 1}", **x} for i, x in enumerate([first] + rest)]

    rows = [json.loads(l) for l in open(a.answers, encoding="utf-8")]
    rows = [r for r in rows if r.get("sentences") and r.get("finish") == "stop"]
    with ThreadPoolExecutor(8) as ex:
        out = [x for g in ex.map(resp, rows) for x in g]
    open(a.out, "w", encoding="utf-8").write(json.dumps(out))
    print("SCORED", a.out, len(rows), "answers", len(out), "sentences", flush=True)


if __name__ == "__main__":
    main()
