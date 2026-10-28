"""不起 vLLM，直接用 transformers 本地生成，检查 critic 输出能否被解析、reward 和指标能否算出来。

用法：python -m scripts.smoke_eval --model outputs/smoke/ckpt/critic-0 --n 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.output_format import parse_output
from common.prompts import critic_messages
from common.schema import cited_ids, read_jsonl, write_jsonl
from eval.metrics import aggregate, per_sample
from train.reward import score_components, total_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="outputs/smoke/coldstart.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default="outputs/smoke/eval")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    samples = list({s["id"]: s for s in read_jsonl(args.data)}.values())[: args.n]  # 冒烟数据有重复，按 id 去重

    rows, preds = [], []
    for s in samples:
        prompt = tok.apply_chat_template(critic_messages(s), tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        output = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        p = parse_output(output)
        reward = total_reward(score_components(output, s["label"], s["type"], s["refs"], cited_ids(s),
                                               s.get("evidence", "")))
        print(f"--- {s['id']} gold={s['label']}/{s['type']} pred={p.verdict}/{p.type} "
              f"format={'ok' if p.format_ok else p.error} reward={reward:.2f}")
        print(output[:400])
        rows.append(per_sample(s, output))
        preds.append({"id": s["id"], "output": output, "reward": reward})

    out = Path(args.out)
    write_jsonl(out / "predictions.jsonl", preds)
    res = aggregate(rows)
    res["format_ok"] = sum(parse_output(p["output"]).format_ok for p in preds) / len(preds)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
