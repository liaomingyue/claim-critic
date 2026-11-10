"""9B 原始模型的幻觉摸底（GRPO 之前先看 9B 原始版的幻觉情况，模型和 Claude 都评一遍）。

在金标评测集的原文上生成回答，rm1-9b 全部打分，随机抽回答按金标标准由 Claude 标注，看 9B 的幻觉率和 critic 在 9B 回答上准不准。
进入 GRPO 的条件（事先定好）：带明显错误的回答不低于 30%；critic 明显错误 AUC 不低于 0.90。

  python -m ragtruth.base_eval gen --base_url http://127.0.0.1:8100/v1 --model base   # 每个原文 16 次
  python -m ragtruth.base_eval score --base_url http://127.0.0.1:8100/v1 --model rm1
  python -m ragtruth.base_eval pick                                                   # 随机抽 100 条，不看分数
  python -m ragtruth.base_eval label                                                  # Claude，约 100 次调用
  python -m ragtruth.base_eval report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl
from ragtruth.convert import split_sentences
from ragtruth.gold import POOL, TEST, excluded_sources, label, write_final
from ragtruth.rm_data import RM1_OPT, RM1_SYSTEM, too_long
from ragtruth.rm_eval import NO_THINK, auc, option_logprobs, probs
from ragtruth.task import render_input

D = Path("outputs/rt/base9b")
MODEL_NAME = "qwen3.5-9b-base"
POOL_D = D / "pool"   # 训练池原文上的 9B 回答（约 300 次 Claude 调用）：标注后进 critic 训练，原文不和留出集重叠


def to_answer(src: dict, k: int, text: str, finish: str) -> dict:
    spans = split_sentences(text)
    return {"id": f"b9-{src['source_id']}-{k}", "source_id": src["source_id"], "task_type": src["task_type"], "model": MODEL_NAME, "source": src["source"], "response": text,
            "sentences": [text[a:b] for a, b in spans], "finish": finish, "bad": {}}


def pick_pool_sources(rows: list[dict], excl: set[str], n: int, seed: int) -> list[dict]:
    """每个原文取一行，QA、Summary 各一半。"""
    srcs = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["task_type"] in ("QA", "Summary") and r["source_id"] not in excl:
            srcs.setdefault(r["source_id"], r)
    out = []
    for t in ("QA", "Summary"):
        ids = sorted(s for s, r in srcs.items() if r["task_type"] == t)
        random.Random(f"{seed}|b9pool|{t}").shuffle(ids)
        out += [srcs[s] for s in ids[:n // 2]]
    return out


def cmd_gen(args) -> None:
    out = POOL_D if args.pool else D
    out.mkdir(parents=True, exist_ok=True)
    srcs = {}
    if args.pool:
        excl = excluded_sources() | {r["source_id"] for r in read_jsonl("outputs/rt/gold/seed_ids.jsonl")}   # 不碰测试、校准、金标种子的原文
        srcs = {r["source_id"]: r for r in pick_pool_sources(list(read_jsonl(POOL)), excl, args.n, args.seed)}
    else:
        for r in sorted(read_jsonl(TEST), key=lambda r: r["id"]):
            srcs.setdefault(r["source_id"], r)
    print("sources", len(srcs), collections.Counter(r["task_type"] for r in srcs.values()), flush=True)
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=args.max_tokens, concurrency=args.concurrency, timeout=900))

    def run(it: dict) -> dict:
        resp = llm.client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": it["src"]["source"]}], max_tokens=args.max_tokens, temperature=args.temperature, seed=it["seed"], **NO_THINK)
        c = resp.choices[0]
        return {**to_answer(it["src"], it["k"], (c.message.content or "").strip(), c.finish_reason), "id": it["id"]}

    items = [{"id": f"b9-{s}-{k}", "src": srcs[s], "k": k, "seed": k} for s in sorted(srcs) for k in range(1 if args.pool else args.n)]
    map_resumable(items, run, out / "answers.jsonl", concurrency=args.concurrency)
    rows = kept(out / "answers.jsonl")
    print("GEN_DONE", len(rows), collections.Counter(r["finish"] for r in rows), "mean sentences", round(statistics.mean(len(r["sentences"]) for r in rows), 2), flush=True)


def cmd_score(args) -> None:
    limit = json.load(open("outputs/rt/rm/rm1/meta.json", encoding="utf-8"))["max_tokens"]
    llm = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=8, concurrency=args.concurrency, timeout=600))
    items = []
    for r in (read_jsonl(D / "picks.jsonl") if args.rm else kept(D / "answers.jsonl")):   # --rm：新版 critic 只需给留出集打分
        for i in range(1, len(r["sentences"]) + 1):
            prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": r["source"], "sentences": r["sentences"], "idx": i})}]
            if not too_long(prompt, limit):
                items.append({"id": f"{r['id']}-S{i}", "resp_id": r["id"], "idx": i, "prompt": prompt})

    def run(it: dict) -> dict:
        resp = llm.client.chat.completions.create(model=args.model, messages=it["prompt"], max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, **NO_THINK)
        return {"id": it["id"], "resp_id": it["resp_id"], "idx": it["idx"], "lp": option_logprobs(resp.choices[0], "".join(sorted(RM1_OPT.values())))}

    path = D / (f"holdout_{args.rm}.jsonl" if args.rm else "scores.jsonl")
    map_resumable(items, run, path, concurrency=args.concurrency)
    print("SCORE_DONE", len(kept(path)), flush=True)


def pick_random(rows: list[dict], n: int, seed: int) -> list[dict]:
    """每个原文先取一条，剩下的名额随机。只看回答是否正常结束，不看分数。"""
    ok = sorted((r for r in rows if r["finish"] == "stop" and r["sentences"]), key=lambda r: r["id"])
    rng = random.Random(f"{seed}|base9b")
    rng.shuffle(ok)
    picks, used = [], set()
    for r in ok:
        if r["source_id"] not in used:
            used.add(r["source_id"])
            picks.append(r)
    chosen = {r["id"] for r in picks}
    picks += [r for r in ok if r["id"] not in chosen][:max(0, n - len(picks))]
    return sorted(picks[:n], key=lambda r: r["id"])


def cmd_pick(args) -> None:
    picks = pick_random(kept(D / "answers.jsonl"), args.n, args.seed)
    (D / "picks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picks), encoding="utf-8")
    print("PICKED", len(picks), "sources", len({r["source_id"] for r in picks}), collections.Counter(r["task_type"] for r in picks), "sentences", sum(len(r["sentences"]) for r in picks), flush=True)


def cmd_label(args) -> None:
    d = POOL_D if args.pool else D
    picks = {r["id"]: r for r in (kept(d / "answers.jsonl") if args.pool else read_jsonl(d / "picks.jsonl")) if r["finish"] == "stop" and r["sentences"]}
    assert len(picks) <= (310 if args.pool else 110), "Claude 调用预算：约 300（池）或 100（留出集）次"
    llm = LLM(LLMConfig(model=args.model, provider="claude_cli", effort=args.effort, concurrency=args.concurrency))
    label(llm, [{"id": f"{rid}#p1", "r": picks[rid]} for rid in sorted(picks)], d / "labels.jsonl", args.concurrency)
    by = {x["resp_id"]: {1: x["sentences"]} for x in kept(d / "labels.jsonl")}
    rows = write_final(picks, by, d / "final.jsonl")
    print("LABEL_DONE", len(by), "missing", sorted(set(picks) - set(by)), "sentences", len(rows), "errors", sum(x["is_error"] for x in rows), flush=True)


def boot(keys: list[str], stat, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """按原文抽样的 95% 区间。stat 输入抽到的原文列表（可重复）。"""
    rng = random.Random(seed)
    vals = sorted(v for v in (stat([rng.choice(keys) for _ in keys]) for _ in range(n)) if v is not None)
    return round(vals[int(.025 * len(vals))], 4), round(vals[int(.975 * len(vals))], 4)


def cmd_report(args) -> None:
    T = json.load(open("outputs/rt/rm/rm1/eval/report.json", encoding="utf-8"))["temperature"]
    P = {r["id"]: sum(v for k, v in probs(r["lp"], T).items() if k in "BC") for r in kept(D / "scores.jsonl") if r.get("lp")}
    answers = kept(D / "answers.jsonl")
    final = list(read_jsonl(D / "final.jsonl"))
    src = {r["id"]: r["source_id"] for r in answers}
    by_src_sent, by_src_resp = collections.defaultdict(list), collections.defaultdict(dict)
    for x in final:
        if x["id"] in P:
            by_src_sent[x["source_id"]].append(x)
        by_src_resp[x["source_id"]].setdefault(x["resp_id"], []).append(x)
    keys = sorted(by_src_resp)

    def rate(ks, pred):
        rs = [xs for k in ks for xs in by_src_resp[k].values()]
        return sum(pred(xs) for xs in rs) / len(rs) if rs else None

    def sent_auc(ks, keep, pos):
        xs = [x for k in ks for x in by_src_sent[k] if keep(x)]
        ys = [int(pos(x)) for x in xs]
        return auc(ys, [P[x["id"]] for x in xs]) if 0 < sum(ys) < len(ys) else None

    def any_clear(xs):
        return any(x["severity"] == "clear" for x in xs)

    def any_err(xs):
        return any(x["is_error"] for x in xs)

    subs = {"all_errors": (lambda x: True, lambda x: x["is_error"]), "clear_only": (lambda x: x["severity"] != "minor", lambda x: x["severity"] == "clear")}
    rep = {"answers": {"n": len(answers), "finish": dict(collections.Counter(r["finish"] for r in answers)), "mean_sentences": round(statistics.mean(len(r["sentences"]) for r in answers), 2), "mean_chars": round(statistics.mean(len(r["response"]) for r in answers), 1),
                       "sentences": sum(len(r["sentences"]) for r in answers), "scored": len(P), "unscored": sum(len(r["sentences"]) for r in answers) - len(P)},
           "critic_all_answers": {}, "claude": {}, "critic_vs_claude": {}}
    resp_max = collections.defaultdict(float)
    for i, p in P.items():
        rid = i.rsplit("-S", 1)[0]
        resp_max[rid] = max(resp_max[rid], p)
    rep["critic_all_answers"] = {"sentence_flag_rate": round(sum(p >= .5 for p in P.values()) / len(P), 4), "response_flag_rate": round(sum(v >= .5 for v in resp_max.values()) / len(resp_max), 4), "mean_p": round(statistics.mean(P.values()), 4)}
    errs = [x for x in final if x["is_error"]]
    n_resp = sum(len(v) for v in by_src_resp.values())
    rep["claude"] = {"responses": n_resp, "sources": len(keys), "sentences": len(final), "errors": len(errs), "clear": sum(x["severity"] == "clear" for x in errs), "minor": sum(x["severity"] == "minor" for x in errs), "conflict": sum(x["type"] == "conflict" for x in errs),
                     "sentence_error_rate": round(len(errs) / len(final), 4), "resp_any_error": round(rate(keys, any_err), 4), "resp_any_error_ci": boot(keys, lambda k: rate(k, any_err)), "resp_any_clear": round(rate(keys, any_clear), 4), "resp_any_clear_ci": boot(keys, lambda k: rate(k, any_clear)),
                     "clear_per_response": round(sum(x["severity"] == "clear" for x in errs) / n_resp, 3), "mechanism": dict(collections.Counter(x["mechanism"] for x in errs).most_common()),
                     "by_task": {t: {"responses": len({x["resp_id"] for x in final if x["task_type"] == t}), "any_clear": len({x["resp_id"] for x in errs if x["task_type"] == t and x["severity"] == "clear"}), "any_error": len({x["resp_id"] for x in errs if x["task_type"] == t})} for t in sorted({x["task_type"] for x in final})}}
    for name, (keep, pos) in subs.items():
        xs = [x for k in keys for x in by_src_sent[k] if keep(x)]
        tp = sum(pos(x) and P[x["id"]] >= .5 for x in xs)
        fl = sum(P[x["id"]] >= .5 for x in xs)
        npos = sum(bool(pos(x)) for x in xs)
        rep["critic_vs_claude"][name] = {"n": len(xs), "pos": npos, "auc": sent_auc(keys, keep, pos), "auc_ci": boot(keys, lambda k: sent_auc(k, keep, pos)), "recall": round(tp / max(1, npos), 4), "precision": round(tp / max(1, fl), 4), "flagged": fl}
    resp_ids = [rid for v in by_src_resp.values() for rid in v]
    y = [int(any_clear(xs)) for v in by_src_resp.values() for xs in v.values()]
    rep["critic_vs_claude"]["response_any_clear_auc"] = auc(y, [resp_max[r] for r in resp_ids]) if 0 < sum(y) < len(y) else None
    c = rep["critic_vs_claude"]["clear_only"]
    rep["gate"] = {"resp_any_clear_ge_0.30": rep["claude"]["resp_any_clear"] >= .30, "critic_clear_auc_ge_0.90": (c["auc"] or 0) >= .90}
    (D / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


def holdout_metrics(ys: list[int], base: list[float], arm: list[float]) -> dict:
    """arm 对基线：AUC、0.5 阈值的召回和精确度，以及和 arm 判错数量相同时基线的召回（same-flag，避免只是阈值变松）。"""
    k = sum(p >= .5 for p in arm)

    def at(ps: list[float]) -> float:
        th = sorted(ps, reverse=True)[max(1, k) - 1]
        return sum(y and p >= th for y, p in zip(ys, ps)) / max(1, sum(ys))

    tp = sum(y and p >= .5 for y, p in zip(ys, arm))
    return {"auc": auc(ys, arm), "auc_delta": auc(ys, arm) - auc(ys, base), "recall": tp / max(1, sum(ys)), "precision": tp / max(1, k), "flagged": k, "same_flag_delta": at(arm) - at(base)}


def cmd_holdout(args) -> None:
    """重训的 critic 在留出集（评测集原文上的 9B 回答，Claude 标注）上和 rm1-9b 比。配对 bootstrap 按原文抽样。"""
    def load(path: Path, rm: str) -> dict[str, float]:
        T = json.load(open(f"outputs/rt/rm/{rm}/eval/report.json", encoding="utf-8"))["temperature"]
        return {r["id"]: sum(v for k, v in probs(r["lp"], T).items() if k in "BC") for r in kept(path) if r.get("lp")}

    S = {"rm1": load(D / "scores.jsonl", "rm1"), **{a: load(D / f"holdout_{a}.jsonl", a) for a in args.arms}}
    final = [x for x in read_jsonl(D / "final.jsonl") if all(x["id"] in s for s in S.values())]
    subs = {"all_errors": (lambda x: True, lambda x: x["is_error"]), "clear_only": (lambda x: x["severity"] != "minor", lambda x: x["severity"] == "clear")}
    rep = {}
    for name, (keep, pos) in subs.items():
        xs = sorted((x for x in final if keep(x)), key=lambda x: x["id"])
        ys = [int(pos(x)) for x in xs]
        src = collections.defaultdict(list)
        for n, x in enumerate(xs):
            src[x["source_id"]].append(n)
        keys = sorted(src)
        rng = random.Random(args.seed)
        draws = [[n for _ in keys for n in src[rng.choice(keys)]] for _ in range(1000)]
        base = [S["rm1"][x["id"]] for x in xs]
        rep[name] = {"n": len(xs), "pos": sum(ys), "rm1": holdout_metrics(ys, base, base)}
        for a in args.arms:
            ps = [S[a][x["id"]] for x in xs]
            m = holdout_metrics(ys, base, ps)
            vals = [holdout_metrics([ys[n] for n in d], [base[n] for n in d], [ps[n] for n in d]) for d in draws if 0 < sum(ys[n] for n in d) < len(d)]
            for f in ("auc", "auc_delta", "same_flag_delta"):
                v = sorted(x[f] for x in vals)
                m[f + "_ci"] = [round(v[int(.025 * len(v))], 4), round(v[int(.975 * len(v))], 4)]
            rep[name][a] = m
    rep["gate_clear_auc_ge_0.90"] = {a: rep["clear_only"][a]["auc"] >= .90 for a in args.arms}
    (D / "holdout_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "pick", "label", "report", "holdout"])
    ap.add_argument("--pool", action="store_true", help="gen、label：用训练池原文，每个原文一条")
    ap.add_argument("--rm", default="", help="score：给留出集打分，存 holdout_<rm>.jsonl")
    ap.add_argument("--arms", nargs="*", default=[], help="holdout：参加比较的 rm1__<组>")
    ap.add_argument("--base_url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=0, help="gen：每个原文几次（默认 16）；pick：抽几条（默认 100）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    if not args.n:
        args.n = 300 if args.pool else 16 if args.cmd == "gen" else 100
    if args.cmd == "label":
        args.model = args.model or "claude-opus-5-5"
        args.concurrency = min(args.concurrency, 6)
    {"gen": cmd_gen, "score": cmd_score, "pick": cmd_pick, "label": cmd_label, "report": cmd_report, "holdout": cmd_holdout}[args.cmd](args)


if __name__ == "__main__":
    main()
