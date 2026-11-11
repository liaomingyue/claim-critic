"""合适度模型批量打分：给一批原文（每行至少 id、source），输出每个原文 8 个机制的校准概率，以及每个机制按分数排好的列表。

用法（按某个类型排序取前 N 个原文去造这类错）：
  python -m train.suit_score --base <MiniCPM5-2B> --model_dir <suit_v1 目录（含 adapter/ 和 report.json）> --inp <原文.jsonl> --out <目录>
  → <目录>/scores.jsonl：{id, source_id, probs: {机制: 概率}}
    <目录>/rank_<机制>.jsonl：按该机制概率从高到低排好的 {id, source_id, prob, rank}

- 读分和训练脚本完全一样（同一提示、同样按 --perms 个固定顺序取平均 logit、report.json 里的全局温度）；
- 原文如果出现在合适度模型的测试 / 策略验收原文里会报警（这些原文选出来造错再进 critic 训练，等于泄题），--allow_eval 才放行。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.suit_data import MECHS, OUT
from train.suit_train import letter_ids, perms_for, render


def eval_sources() -> set[str]:
    s = set()
    for n in ("test", "policy"):
        p = OUT / f"{n}.jsonl"
        if p.exists():
            s |= {str(r["source_id"]) for r in read_jsonl(p)}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--allow_eval", action="store_true")
    a = ap.parse_args()
    rows = list(read_jsonl(Path(a.inp)))
    leak = [r["id"] for r in rows if str(r.get("source_id", "")) in eval_sources()]
    if leak and not a.allow_eval:
        raise SystemExit(f"{len(leak)} 个原文在评测原文里（如 {leak[:3]}）；确实要打分加 --allow_eval")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    md = Path(a.model_dir)
    T = json.loads((md / "report.json").read_text(encoding="utf-8"))["temperature"]
    tok = AutoTokenizer.from_pretrained(a.base)
    lids = letter_ids(tok)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda"), str(md / "adapter")).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scored = []
    with torch.no_grad(), open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            acc = {m: 0.0 for m in MECHS}
            ps = perms_for(str(r["id"]), a.perms)
            for perm in ps:
                ids = tok.apply_chat_template(render(r["source"], perm), tokenize=True, add_generation_prompt=True)
                ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)   # transformers 5：返回 BatchEncoding（不是 dict 子类）
                if len(ids) > a.max_len:
                    ids = ids[: a.max_len // 2] + ids[-(a.max_len - a.max_len // 2):]
                z = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1, lids].float().tolist()
                for m, v in zip(perm, z):
                    acc[m] += v / len(ps)
            rec = {"id": r["id"], "source_id": r.get("source_id"), "probs": {m: round(1 / (1 + math.exp(-acc[m] / T)), 4) for m in MECHS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            scored.append(rec)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(rows)}", flush=True)
    for m in MECHS:
        order = sorted(scored, key=lambda x: (-x["probs"][m], str(x["id"])))
        with open(out / f"rank_{m}.jsonl", "w", encoding="utf-8") as f:
            for i, x in enumerate(order, 1):
                f.write(json.dumps({"id": x["id"], "source_id": x["source_id"], "prob": x["probs"][m], "rank": i}, ensure_ascii=False) + "\n")
    print("SUIT_SCORE_DONE", len(scored), flush=True)


if __name__ == "__main__":
    main()
