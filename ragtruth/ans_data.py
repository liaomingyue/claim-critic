"""回答 GRPO 的 prompt 池（2026-11-02）。

prompt 就是 RAGTruth 原文（含任务说明），9B 直接回答，和 ragtruth.base_eval 同一格式。只用 QA、Summary（评测集只有这两类）。
不碰：评测集、校准集、金标种子的原文（excluded_sources），以及 critic 训练用过 9B 回答的 300 个原文（critic 在这些原文上偏乐观）。

  python -m ragtruth.ans_data split                       # 分 dev（每个检查点评测）和 train，本地可跑
  python -m ragtruth.ans_data ref --base_url … --model policy_v0   # 起点模型每个 prompt 采 4 次：长度参考、去掉 9B 通常就拒答的 prompt
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.base_eval import pick_pool_sources
from ragtruth.gold import POOL, excluded_sources

D = Path("outputs/rt/ansgrpo")
# MiniCPM5-2B 不开思考时 QA 几乎都拒答（60 条里 54 条），用户要求写进系统提示：不是所有题都要拒答（2026-11-02）
SYSTEM_QA = "Answer the question using the passages. If the passages contain information relevant to the question, give the best answer they support, even if it is partial. Reply that you are unable to answer only when the passages contain nothing relevant to the question."


def policy_prompt(tok, row: dict, sys_qa: bool) -> str:
    from train.gen_grpo import chat_prompt
    return chat_prompt(tok, row["source"], SYSTEM_QA if sys_qa and row["task_type"] == "QA" else "")
MAX_CHARS = 8000   # critic 训练截断在 3072 token，再长的原文它没见过


def cmd_split(args) -> None:
    excl = excluded_sources() | {r["source_id"] for r in read_jsonl("outputs/rt/gold/seed_ids.jsonl")} | {r["source_id"] for r in read_jsonl("outputs/rt/base9b/pool/answers.jsonl")}
    rows = [r for r in read_jsonl(POOL) if len(r["source"]) <= MAX_CHARS]
    picked = pick_pool_sources(rows, excl, 100_000, args.seed)   # 每个原文一行
    dev = {t: [r for r in picked if r["task_type"] == t][:args.dev // 2] for t in ("QA", "Summary")}
    dev_ids = {r["source_id"] for v in dev.values() for r in v}
    keep = lambda r: {"id": r["source_id"], "source_id": r["source_id"], "task_type": r["task_type"], "source": r["source"]}
    train = [keep(r) for r in picked if r["source_id"] not in dev_ids]
    random.Random(f"{args.seed}|ans").shuffle(train)
    D.mkdir(parents=True, exist_ok=True)
    write(D / "dev.jsonl", [keep(r) for v in dev.values() for r in v])
    write(D / "train_all.jsonl", train)
    print("SPLIT", json.dumps({"train": len(train), "dev": len(dev_ids), "by_task": dict(collections.Counter(r["task_type"] for r in train))}), flush=True)


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def filter_ref(rows: list[dict], gens: list[list[dict]], k: int, keep_refusal: bool = False) -> tuple[list[dict], dict]:
    """正常结束的回答的中位长度记 ref_chars；一半以上拒答或截断的 prompt 不训（拒答在这里是对的，罚它会教坏模型）。
    keep_refusal：有拒答 RM 时这些题留着（用户：MiniCPM 偏向拒答，要教它少拒答），没有正常回答可参考就不设长度参考（ref_chars None，长度规则跳过）。"""
    from train.ans_reward import REFUSAL
    out, dropped = [], collections.Counter()
    for r, gs in zip(rows, gens):
        ok = [g["text"].strip() for g in gs if g["finish"] == "stop" and g["text"].strip() and not REFUSAL.search(g["text"])]
        if len(ok) * 2 <= k and not keep_refusal:
            dropped[r["task_type"]] += 1
            continue
        out.append({**r, "ref_chars": int(statistics.median(len(t) for t in ok)) if ok else None})
    return out, {"kept": len(out), "dropped": dict(dropped)}


def cmd_ref(args) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from train.gen_grpo import Server
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    srv = Server(args.base_url)
    D = Path(args.dir)
    rows = list(read_jsonl(D / "train_all.jsonl"))
    with ThreadPoolExecutor(args.concurrency) as ex:
        gens = list(ex.map(lambda r: srv.sample(args.model, policy_prompt(tok, r, args.sys_qa), args.k, args.max_tokens, 1.0), rows))
    kept, meta = filter_ref(rows, gens, args.k, bool(args.keep_refusal))
    write(D / "train.jsonl", kept)
    meta |= {"by_task": dict(collections.Counter(r["task_type"] for r in kept)), "ref_chars_median": statistics.median(r["ref_chars"] for r in kept if r["ref_chars"] is not None), "no_ref": sum(r["ref_chars"] is None for r in kept)}
    (D / "train.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("REF_DONE", json.dumps(meta, ensure_ascii=False), flush=True)


def pick_ref(gens: list[dict]) -> str | None:
    """正常结束、不拒答的回答里长度居中的一条；没有就 None（这个 prompt 不做满意度比较）。"""
    from train.ans_reward import REFUSAL
    ok = sorted((g["text"].strip() for g in gens if g["finish"] == "stop" and g["text"].strip() and not REFUSAL.search(g["text"])), key=len)
    return ok[len(ok) // 2] if ok else None


def cmd_ref2(args) -> None:
    """满意度比较的参考回答：原始 9B（policy_v0），温度 0.7，每个 prompt 采 k 次取长度居中的一条。"""
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from train.gen_grpo import Server
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    srv = Server(args.base_url)
    D = Path(args.dir)
    for name in ("train", "dev"):
        rows = list(read_jsonl(D / f"{name}.jsonl"))
        with ThreadPoolExecutor(args.concurrency) as ex:
            gens = list(ex.map(lambda r: srv.sample(args.model, policy_prompt(tok, r, args.sys_qa), args.k, args.max_tokens, 0.7), rows))
        out = [{**r, "ref_text": pick_ref(g)} for r, g in zip(rows, gens)]
        write(D / f"{name}_ref.jsonl", out)
        print("REF2", name, len(out), "no_ref", sum(r["ref_text"] is None for r in out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "ref", "ref2"])
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="policy_v0")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--dir", default=str(D), help="ref / ref2 的数据目录（MiniCPM 用自己的目录）")
    ap.add_argument("--sys_qa", type=int, default=0, help="QA 加系统提示（MiniCPM）")
    ap.add_argument("--keep_refusal", type=int, default=0, help="ref：常拒答的 prompt 也留着（配合拒答 RM）")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    {"split": cmd_split, "ref": cmd_ref, "ref2": cmd_ref2}[args.cmd](args)


if __name__ == "__main__":
    main()
