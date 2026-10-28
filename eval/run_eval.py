"""在测试集上跑一个 critic（vLLM 服务或任意 OpenAI 兼容 API），输出逐条预测和汇总指标。

零样本大模型基线和同基座 SFT 基线也用这个脚本，只换 --model / --base_url，prompt 完全相同。

用法：
  vllm serve outputs/ckpt/critic-3 --served-model-name critic-3
  python -m eval.run_eval --test data_release/test.jsonl --model critic-3 --out outputs/eval/critic-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.llm import LLM, LLMConfig, map_resumable
from common.prompts import critic_messages
from common.schema import read_jsonl
from eval.metrics import aggregate, bootstrap_f1, per_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                        temperature=0.0, max_tokens=args.max_tokens, concurrency=args.concurrency))
    gold = {s["id"]: s for s in read_jsonl(args.test)}
    out = Path(args.out)
    pred_path = out / "predictions.jsonl"

    map_resumable(gold.values(), lambda s: {"output": llm.chat_one(critic_messages(s))},
                  pred_path, concurrency=args.concurrency)

    rows = [per_sample(gold[p["id"]], p["output"]) for p in read_jsonl(pred_path) if p["id"] in gold]
    res = aggregate(rows)
    res["f1_ci95"] = bootstrap_f1(rows)
    res["model"] = args.model
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
