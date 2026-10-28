"""在 RAGTruth 测试集上跑一个 critic（vLLM 服务或任意 OpenAI 兼容 API），输出预测和汇总指标。

两类 critic，指标口径相同（逐句展开后用 ragtruth.metrics 汇总）：
- 逐句（默认）：测试集回答展开成逐句样本，每句一次调用；--style full（evidence → reason → 类型 → 判定）| verdict（只判定）。
- 回答级：--resp_style plain | evidence，每条回答一次调用（见 ragtruth.resp）。

用法（Windows 访问 WSL 端口不稳，客户端放在 vLLM 容器里跑）：
  docker exec -w /work docker-vllm-1 python3 -m ragtruth.eval --model critic --base_url http://127.0.0.1:8000/v1 \
      --out outputs/rt/eval/sft-0.6b
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from common import manifest
from common.llm import LLM, LLMConfig, map_resumable
from common.schema import read_jsonl
from ragtruth import resp as rresp
from ragtruth.metrics import aggregate, bootstrap_f1, per_sample
from ragtruth.task import expand_claims, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs/external/ragtruth/test.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--sample", type=int, default=0, help="按回答随机抽 N 条（固定种子），0 = 全量")
    ap.add_argument("--no_think", action="store_true", help="关闭 Qwen3 思考模式（训练格式不含 think 时用）")
    ap.add_argument("--tips", action="store_true", help="system prompt 带判定要点（须与训练时一致）")
    ap.add_argument("--style", choices=["full", "verdict", "vfirst"], default="full", help="逐句 critic 的输出格式（ragtruth.task.STYLES）")
    ap.add_argument("--resp_style", choices=["plain", "evidence"], default=None, help="给出时按回答级 critic 评测")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                        temperature=0.0, max_tokens=args.max_tokens, concurrency=args.concurrency))
    resp = list(read_jsonl(args.test))
    if args.sample:
        resp = random.Random(0).sample(resp, args.sample)
    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if args.no_think else {}
    out = Path(args.out)
    pred_path = out / "predictions.jsonl"

    if args.resp_style:
        by_id = {s["id"]: s for s in resp}
        map_resumable(resp, lambda s: {"output": llm.chat_one(rresp.messages(s, args.resp_style), **extra)},
                      pred_path, concurrency=args.concurrency)
        rows = [r for p in read_jsonl(pred_path) if p["id"] in by_id
                for r in rresp.sentence_rows(by_id[p["id"]], p["output"], args.resp_style)]
    else:
        gold = {c["id"]: c for s in resp for c in expand_claims(s)}
        map_resumable(gold.values(),
                      lambda c: {"output": llm.chat_one(messages(c, args.tips, args.style), **extra)},
                      pred_path, concurrency=args.concurrency)
        rows = [per_sample(gold[p["id"]], p["output"], args.style) for p in read_jsonl(pred_path) if p["id"] in gold]

    res = aggregate(rows)
    res["f1_ci95"] = bootstrap_f1(rows)
    res["model"] = args.model
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    manifest.write("eval", out.name, out, vars(args), [args.test], started=t0,
                   extra={"model_served": args.model, "metrics": {k: res[k] for k in ("sentence", "response", "f1_ci95")
                                                                  if k in res}})


if __name__ == "__main__":
    main()
