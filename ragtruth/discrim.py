"""真实性判别器：区分「模型自然犯的错」（RAGTruth 训练集里人工标出的幻觉句）和「人为合成的错」（合成池里的有效改写）。

用途（技术报告 Q6）：给合成错误打「像真实错误」的分数，作为生成器 DPO 的第二个偏好信号——只奖励「骗过 critic」
会把生成器推向「为错而错」；加上真实性，看提升能否迁移到真实测试集。思路类似 GAN 的判别器，但不做端到端对抗训练。

防捷径：
- 两类都是错误句，判别器学的是错误的「样子」，不是「有没有错」；
- 只给原文 + 目标句 + 前后各一句，不给整条回答：合成错误所在的回答其余都没问题，真实回答常有多处错误，
  给整条回答会让「别处还有没有错」变成线索；
- 按任务类型、生成回答的模型配平两类，按 source 划分训练 / 留出，避免同一原文两边都出现。

用法：
  python -m ragtruth.discrim build --out_dir outputs/rt/discrim
  python -m train.sft --config configs/rt/discrim_9b.yaml --mode coldstart
  python -m ragtruth.discrim score --synth <合成记录 jsonl> --out <分数 jsonl> --model discrim
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.llm import LLM, LLMConfig, map_resumable
from common.schema import read_jsonl, write_jsonl

SYSTEM = """You see a source (the exact input an assistant received) and one sentence that an assistant wrote with this source, with the sentence before and after it for context. The sentence is known to be unfaithful to the source: it contradicts the source or adds information the source does not contain.

Decide where the error comes from:
- natural: the assistant made this mistake on its own while writing its answer;
- synthetic: a faithful sentence was deliberately edited by someone to introduce the error.

Output exactly: <origin>natural or synthetic</origin>"""


# 统一判别器：「模型自己写的 vs 被改过的」和这句话对不对无关，正确句、错句一起判。
# 自然类 = RAGTruth 正确句 + 真实错句；合成类 = 修复出来的正确句 + 生成器造的错句；四类配平，prompt 不再预设这句有错
SYSTEM_UNIFIED = """You see a source (the exact input an assistant received) and one sentence from an assistant's answer written with this source, with the sentence before and after it for context. The sentence may or may not be faithful to the source.

Decide how the sentence came about:
- natural: the assistant wrote this sentence itself, as it stands;
- synthetic: someone edited the sentence afterwards (to fix it or to introduce a mistake).

Output exactly: <origin>natural or synthetic</origin>"""


def render(source: str, sentences: list[str], idx: int) -> str:
    prev = sentences[idx - 2] if idx >= 2 else ""
    nxt = sentences[idx] if idx < len(sentences) else ""
    return "\n".join(["[source]", source.strip(), "", "[context before]", prev or "(none)",
                      "[sentence]", sentences[idx - 1], "[context after]", nxt or "(none)"])


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_UNIFIED if row.get("unified") else SYSTEM},
            {"role": "user", "content": render(row["source"], row["sentences"], row["idx"])}]


def target_text(row: dict) -> str:
    return f"<origin>{row['origin_label']}</origin>"


def parse(text: str) -> str | None:
    t = text.lower()
    return "natural" if "<origin>natural" in t else "synthetic" if "<origin>synthetic" in t else None


def logprob_score(client, model: str, row: dict) -> dict:
    """生成式判别器的连续分数：贪心生成到标签词，取该位置 natural / synthetic 两个候选的 logprob，两者之间归一化。
    比采样 4 次数票（只有 0 / .25 / .5 / .75 / 1 五档）分辨率高，帕累托配对时少并列。"""
    import math

    resp = client.chat.completions.create(
        model=model, messages=messages(row), max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    toks = resp.choices[0].logprobs.content or []
    # 标签词常和标签的 '>' 合成一个 token（'>natural'），先去掉开头的标点再比；只看 '<origin' 之后的位置。
    # 旧写法按原样比前缀，判为 natural 时一个位置都对不上，全部退回 0/1（踩过：D1 分数因此几乎都是 0 或 1）
    norm = lambda s: re.sub(r"^[^a-z]+", "", s.strip().lower())  # noqa: E731
    is_n = lambda s: bool(s) and ("natural".startswith(s) or s.startswith("natural"))  # noqa: E731
    is_s = lambda s: bool(s) and ("synthetic".startswith(s) or s.startswith("synthetic"))  # noqa: E731
    seen = ""
    for t in toks:
        tok = norm(t.token)
        if "origin" in (seen + t.token).lower() and (is_n(tok) or is_s(tok)):
            pn = sum(math.exp(c.logprob) for c in t.top_logprobs if is_n(norm(c.token)))
            ps = sum(math.exp(c.logprob) for c in t.top_logprobs if is_s(norm(c.token)))
            # 判别器很自信时另一个标签排不进前 20，按 1 − 选中概率估计（否则分数只剩 0 / 1，DPO 配不出对；
            # 踩过：D1 在 G_free 产出上 1,126 条全是 0 或 1）
            chosen = math.exp(t.logprob)
            if pn + ps > 0 and pn == 0:
                pn = max(1e-6, 1 - chosen)
            if pn + ps > 0 and ps == 0:
                ps = max(1e-6, 1 - chosen)
            if pn + ps > 0:
                return {"p_natural": pn / (pn + ps), "score": math.log(pn + 1e-12) - math.log(ps + 1e-12)}
        seen += t.token
    lab = parse(resp.choices[0].message.content or "")
    return {"p_natural": 1.0 if lab == "natural" else 0.0, "score": 10.0 if lab == "natural" else -10.0,
            "fallback": True}


def natural_rows(paths: list[str]) -> list[dict]:
    """RAGTruth 训练集里人工标出的幻觉句（不含大片段标出的粗粒度句）。"""
    out = []
    for p in paths:
        for s in read_jsonl(p):
            coarse = set(s.get("coarse") or [])
            for k, kind in s["bad"].items():
                i = int(k)
                if i in coarse:
                    continue
                out.append({"id": f"{s['id']}-S{i}-nat", "source_id": s.get("source_id"), "source": s["source"],
                            "sentences": s["sentences"], "idx": i, "task_type": s["task_type"], "model": s["model"],
                            "type": kind, "origin_label": "natural"})
    return out


def synthetic_rows(paths: list[str], src_of: dict[str, str]) -> list[dict]:
    out = []
    for p in paths:
        for r in read_jsonl(p):
            i = r["idx"]
            sents = r["sentences"][:i - 1] + [r["sentence"]] + r["sentences"][i:]
            out.append({"id": f"{r['id']}-syn", "source_id": src_of.get(r["response_id"]), "source": r["source"],
                        "sentences": sents, "idx": i, "task_type": r["task_type"], "model": r["model"],
                        "type": r["type"], "tactic": r.get("tactic"), "origin_label": "synthetic"})
    return out


def synthetic_rows_from(recs: list[dict], src_of: dict[str, str]) -> list[dict]:
    out = []
    for r in recs:
        i = r["idx"]
        out.append({"id": f"{r['id']}-gen", "source_id": src_of.get(r["response_id"]), "source": r["source"],
                    "sentences": r["sentences"][:i - 1] + [r["sentence"]] + r["sentences"][i:], "idx": i,
                    "task_type": r["task_type"], "model": r["model"], "type": r["type"], "tactic": r.get("tactic"),
                    "origin_label": "synthetic"})
    return out


def balance(nat: list[dict], syn: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """按（任务类型, 生成回答的模型）分层，两类条数一样多。"""
    by_n, by_s = collections.defaultdict(list), collections.defaultdict(list)
    for r in nat:
        by_n[(r["task_type"], r["model"])].append(r)
    for r in syn:
        by_s[(r["task_type"], r["model"])].append(r)
    keep_n, keep_s = [], []
    for k in sorted(set(by_n) & set(by_s)):   # 排序：集合遍历顺序随 PYTHONHASHSEED 变，会让每次构建的抽样不同
        m = min(len(by_n[k]), len(by_s[k]))
        keep_n += rng.sample(by_n[k], m)
        keep_s += rng.sample(by_s[k], m)
    return keep_n, keep_s


def cmd_build(args) -> None:
    rng = random.Random(args.seed)
    pool = list(read_jsonl(args.pool))
    src_of = {r["id"]: r.get("source_id") for r in pool}
    nat = natural_rows([args.pool, args.sub])
    syn = synthetic_rows(args.synth, src_of)
    for gd in args.gen_dirs:   # 生成器新产出的有效改写（ragtruth.gen_dpo valid），重训判别器时加入合成类
        recs = [c["rec"] for r in read_jsonl(Path(gd) / "valid.jsonl") if not r.get("_dropped")
                for c in r["cands"] if c["valid"]]
        syn += synthetic_rows_from(recs, src_of)
    nat, syn = balance(nat, syn, rng)
    if args.cap and len(nat) > args.cap:   # 每类上限，控制重训时间
        keep = set(rng.sample(range(len(nat)), args.cap))
        nat = [r for i, r in enumerate(nat) if i in keep]
        syn = [r for i, r in enumerate(syn) if i in keep]
    sources = sorted({r["source_id"] for r in nat + syn})
    rng.shuffle(sources)
    held = set(sources[:int(len(sources) * args.heldout_frac)])
    rows = nat + syn
    rng.shuffle(rows)
    train = [r for r in rows if r["source_id"] not in held]
    test = [r for r in rows if r["source_id"] in held]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "heldout.jsonl", test)
    meta = {"natural": len(nat), "synthetic": len(syn), "train": len(train), "heldout": len(test),
            "by_task": dict(collections.Counter(r["task_type"] for r in rows)),
            "type_natural": dict(collections.Counter(r["type"] for r in nat)),
            "type_synthetic": dict(collections.Counter(r["type"] for r in syn))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def unified_rows(args, rng: random.Random) -> dict[str, list[dict]]:
    """四个格子：nat_ok / nat_err / syn_ok / syn_err。上下文（前后各一句）里不能留着没修好的错句，
    否则「旁边有错」会成为区分修复句和自然正确句的线索：修复句的相邻幻觉句换成它们的修复句，修不好就不用。"""
    from ragtruth.synth import eligible_sentences
    pool = list(read_jsonl(args.pool))
    src_of = {r["id"]: r.get("source_id") for r in pool}
    resps = {}
    for p in (args.pool, args.sub):
        for s in read_jsonl(p):
            resps.setdefault(s["id"], s)
    base = lambda s: {"source_id": s.get("source_id"), "source": s["source"], "task_type": s["task_type"],  # noqa: E731
                      "model": s["model"], "unified": True}
    nat_ok = []
    for rid in sorted(resps):
        s = resps[rid]
        bad = {int(k) for k in s["bad"]} | set(s.get("coarse") or [])
        ok = [i for i in eligible_sentences(s) if i not in bad and i - 1 not in bad and i + 1 not in bad]
        if ok:
            i = rng.choice(ok)
            nat_ok.append({**base(s), "id": f"{rid}-S{i}-natok", "sentences": s["sentences"], "idx": i, "type": "none",
                           "origin_label": "natural", "correct": True})
    nat_err = [{**r, "unified": True, "correct": False} for r in natural_rows([args.pool, args.sub])]
    rep = {r["id"]: r for r in read_jsonl(args.repair) if "failed" not in r}
    syn_ok = []
    for r in sorted(rep.values(), key=lambda r: r["id"]):
        s = resps[r["resp_id"]]
        sents = list(s["sentences"])
        sents[r["idx"] - 1] = r["repaired"]
        okctx = True
        for j in (r["idx"] - 1, r["idx"] + 1):
            if str(j) in s["bad"] or j in (s.get("coarse") or []):
                nb = rep.get(f"{s['id']}-S{j}")
                if nb is None:
                    okctx = False
                else:
                    sents[j - 1] = nb["repaired"]
        if okctx:
            syn_ok.append({**base(s), "id": f"{r['id']}-fix", "sentences": sents, "idx": r["idx"], "type": "none",
                           "origin_label": "synthetic", "correct": True})
    syn_err = [{**x, "unified": True, "correct": False} for x in synthetic_rows(args.synth, src_of)]
    for gd in args.gen_dirs:
        if (Path(gd) / "valid.jsonl").exists():
            recs = [c["rec"] for r in read_jsonl(Path(gd) / "valid.jsonl") for c in r["cands"] if c["valid"]]
            syn_err += [{**x, "unified": True, "correct": False} for x in synthetic_rows_from(recs, src_of)]
    return {"nat_ok": nat_ok, "nat_err": nat_err, "syn_ok": syn_ok, "syn_err": syn_err}


def cmd_build_unified(args) -> None:
    rng = random.Random(args.seed)
    cells = unified_rows(args, rng)
    nat_ok, syn_ok = balance(cells["nat_ok"], cells["syn_ok"], rng)       # 正确句：按（任务, 模型）配平
    nat_err, syn_err = balance(cells["nat_err"], cells["syn_err"], rng)   # 错句：同上
    if args.cap and len(nat_err) > args.cap:   # 错句远多于修复句，限个上限，免得 D 主要在看错句
        keep = set(rng.sample(range(len(nat_err)), args.cap))
        nat_err = [r for i, r in enumerate(nat_err) if i in keep]
        syn_err = [r for i, r in enumerate(syn_err) if i in keep]
    rows = nat_ok + syn_ok + nat_err + syn_err
    for r in rows:
        r["cell"] = f"{r['origin_label']}-{'ok' if r['correct'] else 'err'}"
    sources = sorted({r["source_id"] for r in rows if r["source_id"]})
    rng.shuffle(sources)
    held = set(sources[:int(len(sources) * args.heldout_frac)])
    rng.shuffle(rows)
    train = [r for r in rows if r["source_id"] not in held]
    test = [r for r in rows if r["source_id"] in held]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "heldout.jsonl", test)
    meta = {"available": {k: len(v) for k, v in cells.items()},
            "used": dict(collections.Counter(r["cell"] for r in rows)), "train": len(train), "heldout": len(test),
            "by_task": dict(collections.Counter(r["task_type"] for r in rows))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def cmd_score(args) -> None:
    """对一批记录打「像自然错误」的分数：采样 n 次，判为 natural 的比例。--eval 时报留出集准确率。"""
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=16, temperature=1.0,
                        concurrency=args.concurrency))
    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    rows = list(read_jsonl(args.rows))

    def run(r: dict) -> dict:
        meta = {"label": r.get("origin_label"), "tactic": r.get("tactic"), "type": r.get("type"),
                "task_type": r.get("task_type"), "correct": r.get("correct")}
        if args.logprob:
            return {**logprob_score(llm.client, args.model, r), **meta}
        outs = llm.chat(messages(r), n=args.n, **extra)
        votes = [parse(o) for o in outs]
        return {"p_natural": sum(v == "natural" for v in votes) / len(votes), **meta}

    map_resumable(rows, run, args.out, concurrency=args.concurrency)
    if args.eval:
        res = [r for r in read_jsonl(args.out) if r.get("label")]
        from train.discrim_head import summarize
        for r in res:
            r.setdefault("score", r["p_natural"])
        summary_core = summarize(res)
        summary_core["by_type"] = {t: summarize([r for r in res if r.get("type") == t]) for t in ("conflict", "baseless")}
        if any(r.get("correct") is not None for r in res):   # 统一判别器：正确句、错句分开看（各自是自然 vs 合成）
            summary_core["by_correct"] = {k: summarize([r for r in res if r.get("correct") is v])
                                          for k, v in (("ok", True), ("err", False))}
        acc = sum((r["p_natural"] >= 0.5) == (r["label"] == "natural") for r in res) / len(res)
        by_tactic = collections.defaultdict(list)
        for r in res:
            if r["label"] == "synthetic":
                by_tactic[r.get("tactic") or "?"].append(r["p_natural"])
        summary = {**summary_core, "n": len(res), "accuracy": round(acc, 3),
                   "mean_p_natural": {lab: round(sum(r["p_natural"] for r in res if r["label"] == lab) /
                                             max(1, sum(r["label"] == lab for r in res)), 3)
                                      for lab in ("natural", "synthetic")},
                   "synthetic_p_natural_by_tactic": {k: round(sum(v) / len(v), 3) for k, v in
                                                     sorted(by_tactic.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}}
        Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pool", default="outputs/external/ragtruth/train_pool.jsonl")
    b.add_argument("--sub", default="outputs/external/ragtruth/train_sub.jsonl")
    b.add_argument("--synth", nargs="+", default=["outputs/rt/synth/r1_ft.jsonl", "outputs/rt/synth/r2_ft.jsonl",
                                                  "outputs/rt/synth/r3_ft.jsonl"])
    b.add_argument("--gen_dirs", nargs="*", default=[], help="gen_dpo 目录：其中的有效改写加入合成类")
    b.add_argument("--cap", type=int, default=0, help="每类最多多少条，0 = 不限")
    b.add_argument("--heldout_frac", type=float, default=0.15)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out_dir", default="outputs/rt/discrim")
    b.add_argument("--unified", action="store_true", help="统一判别器：正确句、错句一起，四类配平，中性 prompt")
    b.add_argument("--repair", default="outputs/rt/realgen/repair.jsonl", help="--unified：修复句（合成的正确句）来源")
    s = sub.add_parser("score")
    s.add_argument("--rows", required=True, help="判别器格式的记录（build 的输出，或 gen_dpo 转出来的合成记录）")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="discrim")
    s.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--concurrency", type=int, default=64)
    s.add_argument("--eval", action="store_true")
    s.add_argument("--logprob", action="store_true", help="用标签词 logprob 打连续分数（默认采样 n 次数票）")
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build_unified(args) if args.unified else cmd_build(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
