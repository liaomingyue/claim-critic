"""扩充 2B GRPO 的按类型测试集（2026-11-03，用户：可以从 2B GRPO 回答里再请 Claude 标 100 条左右）。
原来的验收集（outputs/rt/minicpm/grpo300，100 条）的盲区类别太少（clear 张冠李戴 11 句、与原文矛盾 10 句），一句差 3–9 个点。
RAGTruth++ 测试集的 68 个原文已经全用掉了，这里从 RAGTruth 原版测试集（outputs/external/ragtruth/test.jsonl）里另取 100 个原文：
QA / Summary 各半，不含 RAGTruth++ 的 68 个，也不在回答 GRPO 训练 / 开发集、造数据的原文里（critic 都没见过）。
回答生成和标注口径同 ans_accept：2B GRPO 第 300 步策略、温度 0.7、种子 0、不加系统提示；标注用 ans_accept label（金标 v1，每条回答一次 Claude）。

  python -m ragtruth.accept_more pick                                         # → outputs/rt/minicpm/grpo300b/picks.jsonl
  python -m ragtruth.accept_more gen --base_url http://127.0.0.1:8000/v1 --model g300
  python -m ragtruth.ans_accept label --dir outputs/rt/minicpm/grpo300b      # Claude，约 100 次
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.base_eval import to_answer
from ragtruth.rm_eval import NO_THINK

D = Path("outputs/rt/minicpm/grpo300b")
TEST, TEST_PP = "outputs/external/ragtruth/test.jsonl", "outputs/external/ragtruth/test_pp.jsonl"
USED = ("outputs/rt/ansgrpo/train_all.jsonl", "outputs/rt/ansgrpo/dev.jsonl", "outputs/rt/minicpm/grpo300/answers.jsonl")


def cmd_pick(a) -> None:
    excl = {str(r["source_id"]) for r in read_jsonl(TEST_PP)} | {str(r["source_id"]) for p in USED for r in read_jsonl(p)}
    srcs = {}
    for r in sorted(read_jsonl(TEST), key=lambda r: r["id"]):
        if r["task_type"] in ("QA", "Summary") and str(r["source_id"]) not in excl:
            srcs.setdefault(str(r["source_id"]), r)
    out = []
    for t in ("QA", "Summary"):
        ids = sorted(s for s, r in srcs.items() if r["task_type"] == t)
        random.Random(f"{a.seed}|accept_more|{t}").shuffle(ids)
        out += [srcs[s] for s in ids[:a.n // 2]]
    D.mkdir(parents=True, exist_ok=True)
    (D / "picks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("PICKED", len(out), dict(collections.Counter(r["task_type"] for r in out)), "free sources", len(srcs), flush=True)


def cmd_gen(a) -> None:
    llm = LLM(LLMConfig(model=a.model, base_url=a.base_url, max_tokens=1024, concurrency=a.concurrency, timeout=900))
    items = [{"id": f"g3b-{r['source_id']}-0", "src": r} for r in read_jsonl(D / "picks.jsonl")]

    def run(it: dict) -> dict:
        resp = llm.client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": it["src"]["source"]}], max_tokens=1024, temperature=0.7, seed=0, **NO_THINK)
        c = resp.choices[0]
        return {**to_answer(it["src"], 0, (c.message.content or "").strip(), c.finish_reason), "id": it["id"], "ckpt": a.model, "model": "minicpm5-2b-grpo300"}

    map_resumable(items, run, D / "answers.jsonl", concurrency=a.concurrency)
    rows = kept(D / "answers.jsonl")
    print("GEN_DONE", len(rows), dict(collections.Counter(r["finish"] for r in rows)), "with sentences", sum(bool(r["sentences"]) for r in rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "gen"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="g300")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"pick": cmd_pick, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
