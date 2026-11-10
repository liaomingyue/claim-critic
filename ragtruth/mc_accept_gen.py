"""2B GRPO 两版策略在同样的验收原文-种子上重新生成，用 9B 金标 critic 对比（边界样本训的 critic 当奖励，RAG 生成能不能比上一版提升更大；先不用 Claude）。
参照回答文件（grpo300 / grpo300b 的 answers.jsonl）逐条照抄设置：同 source、同 system（QA 系统提示有就带）、同种子（id 末尾数字），
grpo300 用 T 1.0 / top_p 0.95（accept_compare 的 MiniCPM 配置），grpo300b 用 T 0.7（accept_more）。新旧两版都重新生成，口径完全一样。

  python -m ragtruth.mc_accept_gen --ref outputs/rt/minicpm/grpo300/answers.jsonl --temperature 1.0 --top_p 0.95 \
      --url http://127.0.0.1:8000/v1 --model old --out outputs/rt/minicpm/cmp_h2/old_a.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

from ragtruth.base_eval import to_answer
from ragtruth.rm_eval import NO_THINK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    from openai import OpenAI
    # 双机推理机上做中途评测：VLLM_API_KEY 是推理机的密钥；EVAL_PRIORITY > 0 时按低优先级排队，不抢训练采样（vLLM 数值越小越先）
    cl = OpenAI(base_url=a.url, api_key=os.environ.get("VLLM_API_KEY", "x"), timeout=900)
    pri = int(os.environ.get("EVAL_PRIORITY", "0"))
    body = {**NO_THINK["extra_body"], **({"priority": pri} if pri else {})}
    refs = [json.loads(l) for l in open(a.ref, encoding="utf-8")]

    def run(r: dict) -> dict:
        k = int(r["id"].rsplit("-", 1)[1])
        msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [{"role": "user", "content": r["source"]}]
        c = cl.chat.completions.create(model=a.model, messages=msgs, max_tokens=1024, temperature=a.temperature, top_p=a.top_p, seed=k, extra_body=body).choices[0]
        return {**to_answer(r, k, (c.message.content or "").strip(), c.finish_reason), "id": r["id"], "ckpt": a.model, "system": r.get("system")}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(run, refs))
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("GEN", a.out, len(out), dict(collections.Counter(o["finish"] for o in out)), flush=True)


if __name__ == "__main__":
    main()
