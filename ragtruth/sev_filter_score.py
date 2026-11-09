"""用 sev__core 给已审计的造错样本打分，看它当「过滤器」（不当奖励）够不够准（2026-11-08，用户：sev 模型就是干这个的）。

输入：outputs/rt/core/gen_audit/items_{pair,seed}.jsonl（有 Claude 审计结论）+ samples_9b_audit.jsonl 全部合格样本（之后可直接按 sev 过滤）。
sev 输出六选一概率：A 显然错 + 解释对、B 显然错 + 解释不对、C 细微错 + 解释对、D 细微错 + 解释不对、E 没错、F 原句有错。
  python -m ragtruth.sev_filter_score --url http://127.0.0.1:8000/v1 --model sev --temp <T> --out outputs/rt/core/gen_audit/sev_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="sev")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--out", default=str(D / "gen_audit/sev_scores.jsonl"))
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--prompt", default="old", choices=["old", "new", "diff", "dne"], help="old = sev__core（sev_rm.prompt）；new / diff / dne = sev__gen 及其变体（sev_gen.PROMPTS）")
    ap.add_argument("--extra", nargs="*", default=[], help="outputs/rt/core 下另外的 9B 采样文件（aid = 9b/<id>）")
    ap.add_argument("--files", nargs="*", default=["items_pair.jsonl", "items_seed.jsonl"], help="gen_audit 下的样本文件")
    a = ap.parse_args()
    from openai import OpenAI
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.sev_gen import PROMPTS
    from ragtruth.sev_rm import prompt as prompt_old
    prompt = prompt_old if a.prompt == "old" else PROMPTS["base" if a.prompt == "new" else a.prompt]
    from train.gen_reward import NO_THINK
    cl = OpenAI(base_url=a.url, api_key="x", timeout=600, max_retries=3)
    items = {}
    for f in a.files:
        for x in read_jsonl(D / "gen_audit" / f):
            items[x["aid"]] = x
    for x in (y for f in ["samples_9b_audit.jsonl", *a.extra] for y in read_jsonl(D / f)):
        if x.get("why") == "ok":
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            items.setdefault(f"9b/{x['id']}", {**x, "aid": f"9b/{x['id']}", "context": ctx, "idx": int(x["idx"])})

    def one(x: dict) -> dict:
        r = cl.chat.completions.create(model=a.model, messages=prompt(x), max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20,
                                       extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], "ABCDEF")
        return {"aid": x["aid"], "p_sev": probs(lp, a.temp) if lp else None}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(one, sorted(items.values(), key=lambda x: x["aid"])))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("SEV_SCORE", len(out), "none:", sum(r["p_sev"] is None for r in out), flush=True)


if __name__ == "__main__":
    main()
