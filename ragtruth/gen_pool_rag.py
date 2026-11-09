"""更大的造错提示池（2026-11-09 用户）：不局限核心集。RAGTruth 训练原文都用 2B RAG 模型生成回答，
2B critic 逐句打分、只留最可能没错的句子，再让生成器在这些回答上改错。每个回答 × 9 种错误类型，各采 1 条、温度 0（确定性）。
RAG 回答用较高温度（默认 1.0）每个原文采 --k 条（用户：温度调高一点就能多出问答对），每个原文固定随机种子，重跑得到同样的回答；完全相同的回答去重。

不碰的原文：RAGTruth 测试集 + 校准集（gold.excluded_sources）、critic cap3 评测原文、核心集留出、RAG 的 dev、金标种子、
9B 评测池、生成器试采的 64 个提示所在原文。只用 QA、Summary（RAG 模型 / critic / 核心集都只有这两类），原文 ≤ 8000 字符。

  RAG 模型 = RAG GRPO 最好版（mc-grpo-r3_core_cont/step100），见 run_gen_d.sh
  python -m ragtruth.gen_pool_rag answers --model rag --tokenizer $MC --k 4 --temperature 1.0   # ① 每个原文采 k 个回答
  python -m ragtruth.gen_pool_rag clean --tokenizer $MC                  # ② critic r2 逐句打分 → 干净句；生成 9 种类型的提示
之后：python -m ragtruth.gen_v2_grpo harvest --prompts rag_pool/prompts.jsonl --n 1 --temperature 0 …
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl

O = Path("outputs/rt/gen_v2_grpo/rag_pool")
CAP3 = Path("outputs/rt/gen_v2_grpo/cap3_src.json")   # cap3 三个目录 final.jsonl 的 source_id（本机算好上传）
MAX_CHARS = 8000


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def excluded() -> dict[str, set[str]]:
    from ragtruth.gold import excluded_sources
    core = list(read_jsonl("outputs/rt/core/core.jsonl"))
    src_of = {f"core-{x['id']}": x["source_id"] for x in core}
    probe = list(read_jsonl("outputs/rt/gen_v2_grpo/prompts.jsonl"))[-64:]
    ex = {"test_calib": excluded_sources(),
          "cap3": set(json.loads(CAP3.read_text(encoding="utf-8"))),
          "core_holdout": {x["source_id"] for x in core if x["split"] == "holdout"},
          "rag_dev": {str(r["source_id"]) for r in read_jsonl("outputs/rt/minicpm/grpo/dev.jsonl")},
          "gold_seed": {str(r["source_id"]) for r in read_jsonl("outputs/rt/gold/seed_ids.jsonl")},
          "base9b_pool": {str(r["source_id"]) for r in read_jsonl("outputs/rt/base9b/pool/answers.jsonl")},
          "probe": {src_of[r["resp_id"]] for r in probe if r["resp_id"] in src_of}}
    return {k: {str(s) for s in v} for k, v in ex.items()}


def cmd_answers(a) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from ragtruth.ans_data import policy_prompt
    from ragtruth.convert import split_sentences
    from ragtruth.gold import POOL
    from train.ans_reward import REFUSAL
    from train.gen_grpo import Server
    ex = excluded()
    bad = set().union(*ex.values())
    srcs = {}
    for r in sorted(read_jsonl(POOL), key=lambda r: str(r["id"])):
        sid = str(r["source_id"])
        if r["task_type"] in ("QA", "Summary") and sid not in bad and len(r["source"]) <= MAX_CHARS:
            srcs.setdefault(sid, {"source_id": sid, "task_type": r["task_type"], "source": r["source"]})
    rows = [srcs[k] for k in sorted(srcs)]
    import zlib
    from openai import OpenAI
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)

    def gen(r):   # 每个原文固定种子（确定性）；n 条一次采
        o = cl.completions.create(model=a.model, prompt=tok(policy_prompt(tok, r, True))["input_ids"], n=a.k, max_tokens=a.max_tokens,
                                  temperature=a.temperature, top_p=1.0, seed=zlib.crc32(("rag_pool|" + r["source_id"]).encode()) % 100000,
                                  extra_body={"top_k": -1})
        return [{"text": c.text, "finish": c.finish_reason} for c in o.choices]
    with ThreadPoolExecutor(a.concurrency) as pool:
        gens = list(pool.map(gen, rows))
    out, why = [], collections.Counter()
    for r, gs in zip(rows, gens):
        seen = set()
        for j, g in enumerate(gs):
            t = g["text"].strip()
            sents = [t[x:y] for x, y in split_sentences(t)]
            w = "length" if g["finish"] != "stop" else "refusal" if REFUSAL.search(t) else "one_sentence" if len(sents) < 2 else "dup" if t in seen else "ok"
            why[w] += 1
            if w == "ok":
                seen.add(t)
                out.append({**r, "resp_id": f"rag-{r['source_id']}-{j}", "sentences": sents})
    write(O / "answers.jsonl", out)
    print("RAG_POOL_ANSWERS", json.dumps({"sources": len(rows), "k": a.k, "temperature": a.temperature, "kept": len(out), "why": dict(why),
                                          "excluded": {k: len(v) for k, v in ex.items()},
                                          "by_task": dict(collections.Counter(r["task_type"] for r in out))}), flush=True)


def cmd_clean(a) -> None:
    """critic r2 给每句打 P(有幻觉)，P < --thr 的算干净句（只允许改这些句）；至少 1 句干净的回答 × 9 种类型 → prompts.jsonl。"""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from ragtruth.gen_v2 import MECH, prompt
    from train.gen_reward import load_temps
    from train.gen_reward_v2 import V2SevRewarder
    from ragtruth.rm_data import RM1_SYSTEM
    from ragtruth.task import render_input
    rw = V2SevRewarder(OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3), load_temps(a.temps))
    xs = list(read_jsonl(O / "answers.jsonl"))
    jobs = [(i, k) for i, x in enumerate(xs) for k in range(1, len(x["sentences"]) + 1)]

    def one(j):
        x = xs[j[0]]
        p = rw.option_p("rm1", [{"role": "system", "content": RM1_SYSTEM},
                                {"role": "user", "content": render_input({"source": x["source"], "sentences": x["sentences"], "idx": j[1]})}], "ABC")
        return (p["B"] + p["C"]) if p else 1.0
    with ThreadPoolExecutor(a.concurrency) as pool:
        ps = list(pool.map(one, jobs))
    for (i, k), p in zip(jobs, ps):
        xs[i].setdefault("p_sent", [None] * len(xs[i]["sentences"]))[k - 1] = round(p, 4)
    for x in xs:
        x["clean_idx"] = [k for k, p in enumerate(x["p_sent"], 1) if p < a.thr]
    write(O / "answers_scored.jsonl", xs)
    keep = [x for x in xs if x["clean_idx"]]
    rows = [{"id": f"{x['resp_id']}|{m}", "prompt": prompt(x["source"], x["sentences"], m), "source": x["source"], "sentences": x["sentences"],
             "mechanism": m, "resp_id": x["resp_id"], "task_type": x["task_type"], "clean_idx": x["clean_idx"]} for x in keep for m in MECH]
    random.Random("gen_pool_rag|prompts").shuffle(rows)
    write(O / "prompts.jsonl", rows)
    allp = [p for x in xs for p in x["p_sent"]]
    hist = {f"<{t}": round(sum(p < t for p in allp) / len(allp), 3) for t in (0.05, 0.1, 0.2, 0.3, 0.5)}
    print("RAG_POOL_CLEAN", json.dumps({"answers": len(xs), "sentences": len(allp), "sent_p_hist": hist, "thr": a.thr,
                                        "answers_all_clean": sum(len(x["clean_idx"]) == len(x["sentences"]) for x in xs),
                                        "answers_kept": len(keep), "clean_sents_kept": sum(len(x["clean_idx"]) for x in keep),
                                        "prompts": len(rows)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answers", "clean"])
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="rag")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--temps", default="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=rm1__core_r2")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4, help="answers：每个原文采几个回答")
    ap.add_argument("--temperature", type=float, default=1.0, help="answers：RAG 回答的采样温度")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.1, help="clean：critic P(有幻觉) < thr 的句子算干净")
    a = ap.parse_args()
    {"answers": cmd_answers, "clean": cmd_clean}[a.cmd](a)


if __name__ == "__main__":
    main()
