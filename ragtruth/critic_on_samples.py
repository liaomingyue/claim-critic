"""基线 critic 给造错样本的改写句打分（P(判出) = P(B) + P(C)），用来挑「critic 判不出」的错（r2 的做法，核实换成 sev__gen）。
2026-11-08，用户：sev 过滤之外再加一道「critic 判不出」的筛选。

样本：gen_audit 的审计样本（items_pair / items_sevnew，有 Claude 结论，用来量 sev 在 critic 盲区里准不准）+ 9B 全部合格样本
（samples_9b_audit.jsonl + --extra 里的新采样），aid 同 sev_filter_score（9b/<id>）。
  python -m ragtruth.critic_on_samples --model critic --temp <T> --extra samples_9b_rest.jsonl --out outputs/rt/core/gen_audit/critic_core_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")


def load_items(extra: list[str]) -> dict[str, dict]:
    items = {}
    for f in ("items_pair.jsonl", "items_sevnew.jsonl"):
        if (D / "gen_audit" / f).exists():
            for x in read_jsonl(D / "gen_audit" / f):
                items[x["aid"]] = x
    for f in ["samples_9b_audit.jsonl", *extra]:
        for x in read_jsonl(D / f):
            if x.get("why") == "ok":
                ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
                items.setdefault(f"9b/{x['id']}", {**x, "aid": f"9b/{x['id']}", "context": ctx, "idx": int(x["idx"])})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="critic")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    from openai import OpenAI
    from ragtruth.rm_data import RM1_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.task import render_input
    from train.gen_reward import NO_THINK
    cl = OpenAI(base_url=a.url, api_key="x", timeout=600, max_retries=3)
    items = load_items(a.extra)

    def one(x: dict) -> dict:
        u = render_input({"source": x["source"], "sentences": x["context"], "idx": x["idx"]})
        r = cl.chat.completions.create(model=a.model, messages=[{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": u}],
                                       max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], "ABC")
        p = probs(lp, a.temp) if lp else None
        return {"aid": x["aid"], "p_halluc": (p["B"] + p["C"]) if p else None}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(one, sorted(items.values(), key=lambda x: x["aid"])))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("CRITIC_ON_SAMPLES", len(out), "none:", sum(r["p_halluc"] is None for r in out), flush=True)


if __name__ == "__main__":
    main()
