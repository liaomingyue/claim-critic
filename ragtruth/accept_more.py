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


# ── 第三批（2026-11-05，用户：显然错误扩到约 200 句；多条回答放一次 Claude 调用）──
# 没用过的测试原文只剩 132 篇（Summary 71 / QA 61）→ 全部取，每篇 3 个种子（温度 0.7，同 grpo300b），约 396 条回答，预计新增显然错误约 150 句。
# 标注口径同 ragtruth.gold v1（SYSTEM / valid 原样），一次调用标同一篇原文的 3 条回答（原文只发一遍），按 id 对齐；漏回 / 不合格的单条重标。
D3 = Path("outputs/rt/minicpm/grpo300c")
USED3 = USED + ("outputs/rt/minicpm/grpo300b/answers.jsonl",)


def cmd_pick3(a) -> None:
    excl = {str(r["source_id"]) for r in read_jsonl(TEST_PP)} | {str(r["source_id"]) for p in USED3 for r in read_jsonl(p)}
    srcs = {}
    for r in sorted(read_jsonl(TEST), key=lambda r: r["id"]):
        if r["task_type"] in ("QA", "Summary") and str(r["source_id"]) not in excl:
            srcs.setdefault(str(r["source_id"]), r)
    out = [srcs[k] for k in sorted(srcs)]
    D3.mkdir(parents=True, exist_ok=True)
    (D3 / "picks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("PICKED3", len(out), dict(collections.Counter(r["task_type"] for r in out)), flush=True)


def cmd_gen3(a) -> None:
    llm = LLM(LLMConfig(model=a.model, base_url=a.base_url, max_tokens=1024, concurrency=a.concurrency, timeout=900))
    items = [{"id": f"g3c-{r['source_id']}-{k}", "src": r, "k": k} for r in read_jsonl(D3 / "picks.jsonl") for k in range(a.seeds)]

    def run(it: dict) -> dict:
        resp = llm.client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": it["src"]["source"]}], max_tokens=1024, temperature=0.7, seed=it["k"], **NO_THINK)
        c = resp.choices[0]
        return {**to_answer(it["src"], it["k"], (c.message.content or "").strip(), c.finish_reason), "id": it["id"], "ckpt": a.model, "model": "minicpm5-2b-grpo300"}

    map_resumable(items, run, D3 / "answers.jsonl", concurrency=a.concurrency)
    rows = kept(D3 / "answers.jsonl")
    print("GEN3_DONE", len(rows), dict(collections.Counter(r["finish"] for r in rows)), "with sentences", sum(bool(r["sentences"]) for r in rows), flush=True)


BATCH_NOTE = """

This message contains ONE source and SEVERAL responses written from it, each in its own <response id="..."> block. Label every response independently, exactly as above.
Reply with one JSON object only: {"responses": [{"id": "<response id>", "sentences": [ ...one entry per sentence of that response, in order, same fields as above... ]}, ...]} with one entry per response."""


def batch_msg(rs: list[dict]) -> str:
    blocks = []
    for r in rs:
        sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(r["sentences"], 1))
        blocks.append(f'<response id="{r["id"]}">\n{sents}\n</response>')
    return f"<source>\n{rs[0]['source'].strip()}\n</source>\n\n" + "\n\n".join(blocks)


def cmd_label_batch(a) -> None:
    """同一原文的回答一次标（max_per_call 条）；返回按 id 对齐，单条不合格只丢这一条，最后单条重标补齐。"""
    from ragtruth.gold import SYSTEM, VERSION, user_msg, valid, write_final
    from ragtruth.synth import parse_json_obj
    D_ = Path(a.dir)
    picks = {r["id"]: r for r in kept(D_ / "answers.jsonl") if r["finish"] == "stop" and r["sentences"]}
    by_src = collections.defaultdict(list)
    for rid in sorted(picks):
        by_src[str(picks[rid]["source_id"])].append(picks[rid])
    calls = [{"id": f"c{n:04d}", "rs": g[k:k + a.max_per_call]} for n, (s, g) in enumerate(sorted(by_src.items())) for k in range(0, len(g), a.max_per_call)]
    assert len(calls) <= a.max_calls, f"{len(calls)} calls > max_calls {a.max_calls}"
    llm = LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency))
    raw = D_ / "labels_batch_raw.jsonl"

    def run(c: dict) -> dict:
        out = llm.chat_one([{"role": "system", "content": SYSTEM + BATCH_NOTE}, {"role": "user", "content": batch_msg(c["rs"])}])
        o = parse_json_obj(out)
        got = {str(x.get("id")): x.get("sentences") for x in (o.get("responses") or []) if isinstance(x, dict)} if isinstance(o, dict) else {}
        rows = []
        for r in c["rs"]:
            xs = valid({"sentences": got.get(r["id"])}, len(r["sentences"]))
            if xs is not None:
                rows.append({"id": f"{r['id']}#p1", "resp_id": r["id"], "pass": 1, "version": VERSION, "sentences": xs, "batched": len(c["rs"])})
        return {"id": c["id"], "rows": rows, "missing": [r["id"] for r in c["rs"] if r["id"] not in {x["resp_id"] for x in rows}]}

    map_resumable(calls, run, raw, concurrency=a.concurrency)
    lab = {x["resp_id"]: x for b in kept(raw) for x in b["rows"]}
    miss = sorted(set(picks) - set(lab))
    single = D_ / "labels_single_raw.jsonl"
    if miss:   # 漏回的单条重标（每条一次，最多 max_single 条）
        assert len(miss) <= a.max_single, f"{len(miss)} missing > max_single {a.max_single}"

        def one(it: dict) -> dict | None:
            r = it["r"]
            for _ in range(2):
                xs = valid(parse_json_obj(llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}])), len(r["sentences"]))
                if xs is not None:
                    return {"id": it["id"], "resp_id": r["id"], "pass": 1, "version": VERSION, "sentences": xs, "batched": 1}
            return None
        map_resumable([{"id": f"{rid}#p1", "r": picks[rid]} for rid in miss], one, single, concurrency=a.concurrency)
        for x in kept(single):
            lab[x["resp_id"]] = x
    (D_ / "labels.jsonl").write_text("".join(json.dumps(lab[k], ensure_ascii=False) + "\n" for k in sorted(lab)), encoding="utf-8")
    rows = write_final({k: picks[k] for k in lab}, {k: {1: lab[k]["sentences"]} for k in lab}, D_ / "final.jsonl")
    c = collections.Counter("clear" if x["is_error"] and x["severity"] == "clear" else "minor" if x["is_error"] else "ok" for x in rows)
    print("LABEL_BATCH_DONE", json.dumps({"answers": len(lab), "missing_final": sorted(set(picks) - set(lab)), "calls": len(calls), "single_retries": len(miss),
                                          "sentences": len(rows), "by_severity": dict(c)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "gen", "pick3", "gen3", "label_batch"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dir", default=str(D3))
    ap.add_argument("--max_per_call", type=int, default=3)
    ap.add_argument("--max_calls", type=int, default=140)
    ap.add_argument("--max_single", type=int, default=40)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="g300")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"pick": cmd_pick, "gen": cmd_gen, "pick3": cmd_pick3, "gen3": cmd_gen3, "label_batch": cmd_label_batch}[a.cmd](a)


if __name__ == "__main__":
    main()
