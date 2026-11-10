"""回答 GRPO 的最终验收。

原始 9B 的基线是 ragtruth.base_eval 留出集：评测集原文上的 100 条回答，Claude 按金标 v1 单轮标注。这里用选定的检查点在同样的原文、同样的种子上重新生成（温度 0.7，
格式和 base_eval gen 相同），金标提示词不变，保证和基线同一口径。回答质量另外单独判（ragtruth.ans_quality 的提示词），两边打乱，Claude 不知道回答来自哪个模型；
不和金标放一个调用，否则金标提示词变了就和基线不可比。Claude 调用摊开跑，训练不用等。

  python -m ragtruth.ans_accept gen --model <vLLM 上的 LoRA 名>
  python -m ragtruth.ans_accept label [--gap 20]        # 幻觉，100 次
  python -m ragtruth.ans_accept quality --pairs 50 [--gap 20]    # 质量，每对 2 次
  python -m ragtruth.ans_accept report

别的模型的摸底（MiniCPM5-2B 和 9B 比）也用它，--dir 换目录、--prefix 换回答 id 前缀，例如
  python -m ragtruth.ans_accept gen --model minicpm --dir outputs/rt/minicpm/base --prefix mc --source_model minicpm5-2b
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.ans_quality import SYSTEM as QUALITY_SYSTEM, summarize, valid
from ragtruth.base_eval import D as BASE, to_answer
from ragtruth.gold import label, write_final
from ragtruth.rm_eval import NO_THINK
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/ansgrpo/accept")   # 默认目录；--dir 覆盖


class Paced:
    """每次调用前等 gap 秒，把 Claude 调用摊到较长时间里。"""

    def __init__(self, llm: LLM, gap: float):
        self.llm, self.gap = llm, gap

    def chat_one(self, messages: list[dict], **kw) -> str:
        time.sleep(self.gap)
        return self.llm.chat_one(messages, **kw)


def claude(args) -> Paced:
    return Paced(LLM(LLMConfig(model=args.claude, provider="claude_cli", effort=args.effort, concurrency=args.concurrency)), args.gap)


def cmd_gen(args) -> None:
    D = Path(args.dir)
    D.mkdir(parents=True, exist_ok=True)
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=1024, concurrency=args.concurrency, timeout=900))
    picks = sorted(read_jsonl(BASE / "picks.jsonl"), key=lambda r: r["id"])
    items = [{"id": args.prefix + r["id"][2:], "src": r, "k": int(r["id"].rsplit("-", 1)[1])} for r in picks]

    def run(it: dict) -> dict:
        resp = llm.client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": it["src"]["source"]}], max_tokens=1024, temperature=0.7, seed=it["k"], **NO_THINK)
        c = resp.choices[0]
        return {**to_answer(it["src"], it["k"], (c.message.content or "").strip(), c.finish_reason), "id": it["id"], "ckpt": args.model, **({"model": args.source_model} if args.source_model else {})}

    map_resumable(items, run, D / "answers.jsonl", concurrency=args.concurrency)
    rows = kept(D / "answers.jsonl")
    print("GEN_DONE", len(rows), collections.Counter(r["finish"] for r in rows), flush=True)


def cmd_label(args) -> None:
    D = Path(args.dir)
    picks = {r["id"]: r for r in kept(D / "answers.jsonl") if r["finish"] == "stop" and r["sentences"]}
    assert len(picks) <= 100, "Claude 调用预算：最多 100 次幻觉标注"
    label(claude(args), [{"id": f"{rid}#p1", "r": picks[rid]} for rid in sorted(picks)], D / "labels.jsonl", args.concurrency)
    by = {x["resp_id"]: {1: x["sentences"]} for x in kept(D / "labels.jsonl")}
    rows = write_final(picks, by, D / "final.jsonl")
    print("LABEL_DONE", len(by), "missing", sorted(set(picks) - set(by)), "sentences", len(rows), flush=True)


def quality_items(base: list[dict], arm: list[dict], pairs: int, seed: int) -> list[dict]:
    """按原文配对（同一原文同一种子），QA、Summary 按各自比例抽，打乱。条目 id 不含模型信息。"""
    key = lambda r: r["id"].split("-", 1)[1]
    a = {key(r): r for r in arm}
    both = sorted((key(r), r) for r in base if key(r) in a)
    rng = random.Random(f"{seed}|accept")
    rng.shuffle(both)
    if pairs < len(both):
        both = both[:pairs]
    out = []
    for k, r in both:
        out.append({"pair": k, "source_id": k, "task_type": r["task_type"], "arm": "base", "answer": r["response"], "source": r["source"]})
        out.append({"pair": k, "source_id": k, "task_type": r["task_type"], "arm": "ckpt", "answer": a[k]["response"], "source": r["source"]})
    rng.shuffle(out)
    for n, x in enumerate(out):
        x["id"] = f"a{n:03d}"
    return out


def cmd_quality(args) -> None:
    D = Path(args.dir)
    items = quality_items(list(read_jsonl(BASE / "picks.jsonl")), kept(D / "answers.jsonl"), args.pairs, args.seed)
    assert len(items) <= 2 * args.max_pairs, "超出 Claude 调用预算"
    llm = claude(args)

    def run(it: dict) -> dict | None:
        out = llm.chat_one([{"role": "system", "content": QUALITY_SYSTEM}, {"role": "user", "content": f"[task]\n{it['source']}\n\n[answer]\n{it['answer']}"}])
        o = valid(parse_json_obj(out))
        if o is None:
            print("invalid output", it["id"], out[:200], flush=True)
            return None
        return {k: v for k, v in {**it, **o}.items() if k != "source"}

    map_resumable(items, run, D / "quality.jsonl", concurrency=args.concurrency)
    print("QUALITY_DONE", json.dumps(summarize(kept(D / "quality.jsonl")), ensure_ascii=False), flush=True)


def answer_flags(rows: list[dict]) -> dict[str, dict]:
    """{原文-种子: {any_error, any_clear, err_rate}}。"""
    by = collections.defaultdict(list)
    for x in rows:
        by[x["resp_id"].split("-", 1)[1]].append(x)
    return {k: {"any_error": float(any(x["is_error"] for x in v)), "any_clear": float(any(x["severity"] == "clear" for x in v)), "err_rate": sum(x["is_error"] for x in v) / len(v), "n": len(v)} for k, v in by.items()}


def compare(base: dict, arm: dict, src: dict, n: int = 2000, seed: int = 0) -> dict:
    """配对 bootstrap，按原文抽样。src：{键: 原文}。"""
    keys = sorted(set(base) & set(arm))
    groups = collections.defaultdict(list)
    for k in keys:
        groups[src[k]].append(k)
    gs = sorted(groups)
    rng = random.Random(seed)
    draws = [[k for _ in gs for k in groups[rng.choice(gs)]] for _ in range(n)]
    out = {"answers": len(keys), "sources": len(gs)}
    for f in ("any_error", "any_clear", "err_rate"):
        m = lambda d, ks: sum(d[k][f] for k in ks) / max(1, len(ks))
        v = sorted(m(arm, ks) - m(base, ks) for ks in draws)
        out[f] = {"base": round(m(base, keys), 4), "ckpt": round(m(arm, keys), 4), "delta": round(m(arm, keys) - m(base, keys), 4), "ci": [round(v[int(.025 * n)], 4), round(v[int(.975 * n)], 4)]}
    return out


def cmd_report(args) -> None:
    D = Path(args.dir)
    base_rows, arm_rows = list(read_jsonl(Path(args.base_dir or BASE) / "final.jsonl")), list(read_jsonl(D / "final.jsonl"))
    src = {x["resp_id"].split("-", 1)[1]: x["source_id"] for x in base_rows}
    rep = {"all": compare(answer_flags(base_rows), answer_flags(arm_rows), src)}
    for t in ("QA", "Summary"):
        rep[t] = compare(answer_flags([x for x in base_rows if x["task_type"] == t]), answer_flags([x for x in arm_rows if x["task_type"] == t]), src)
    if (D / "quality.jsonl").exists():
        rep["quality"] = summarize(kept(D / "quality.jsonl"))
    (D / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "label", "quality", "report"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gap", type=float, default=20.0, help="每次 Claude 调用前等几秒")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--max_pairs", type=int, default=50, help="Claude 调用上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=str(D))
    ap.add_argument("--prefix", default="ag", help="回答 id 前缀（替换基线的 b9）")
    ap.add_argument("--source_model", default="", help="写进回答的 model 字段")
    ap.add_argument("--base_dir", default="", help="report 的对照组目录（默认原始 9B 基线；MiniCPM GRPO 对 outputs/rt/minicpm/sys）")
    args = ap.parse_args()
    {"gen": cmd_gen, "label": cmd_label, "quality": cmd_quality, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
