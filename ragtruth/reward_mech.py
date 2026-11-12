"""critic 当奖励的机制分析：在各个起点策略自己的采样上，比较不同 critic 给出的优势值和金标错句的关系。

  sample：某个起点策略在固定的训练题上每题采 n 条（同训练：温度 1.0、top_p 1），记硬规则结果、句子切分、每句的 token 数
  score ：几个打分模型（同一个 vLLM 服务上的 LoRA）逐句打分；满意度、拒答各打一次
  report：按训练代码（ans_reward / gen_reward.group_advantages）重算每个 critic 的奖励和 token 优势，对金标 critic 的标签算指标

奖励和优势值不另写公式，全部调训练用的函数；tests/test_reward_mech.py 检查和 AnswerRewarder.score_group 逐条一致。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train.ans_reward import count_reward, group_baseline, guarded, sentence_adv
from train.gen_reward import group_advantages

SAT_W, REFUSE_W, BETA, GAMMA, COUNT_RATIO, COUNT_MIN = 0.8, 0.4, 0.6, 0.3, 0.5, 2.0   # 同 run_r3full.sh 的 R3 设定
CLEAR, ANY = 0.9, 0.5   # 金标 critic 的口径：P ≥ 0.9 明显错，P ≥ 0.5 任意错（同 RAG 评测）
PEN = 0.1   # 「被罚到」：超出底噪至少 0.1（σ 的下限，相当于一个单位的句子优势）


# ── 采样 / 打分（远程）──

def pick_ids(path: str, k: int, seed: int = 0) -> list[str]:
    ids = sorted(json.loads(l)["id"] for l in open(path, encoding="utf-8"))
    random.Random(seed).shuffle(ids)
    return sorted(ids[:k])


def cmd_sample(a) -> None:
    from transformers import AutoTokenizer
    from ragtruth.ans_data import policy_prompt
    from ragtruth.convert import split_sentences
    from train.ans_grpo import token_sentence_index
    from train.ans_reward import AnswerRewarder
    from train.gen_grpo import Server
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    srv = Server(a.url)
    rw = AnswerRewarder(srv.client, "none", 1.0, None, count_ratio=COUNT_RATIO, count_min=COUNT_MIN)
    want = set(pick_ids(a.prompts, a.k))
    rows = [r for r in map(json.loads, open(a.prompts, encoding="utf-8")) if r["id"] in want]
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]

    def one(r: dict) -> dict:
        gs = srv.sample(a.model, tok(policy_prompt(tok, r, 1))["input_ids"], a.n, a.max_tokens, 1.0)
        gens = []
        for g in gs:
            t = g["text"].strip()
            spans = [list(x) for x in split_sentences(t)]
            idx = token_sentence_index(tok, g["ids"], g["text"], spans) if spans else []
            gens.append({"text": g["text"], "finish": g["finish"], **rw.describe(g["text"], g["finish"], r), "spans": spans,
                         "sent_tok": [sum(1 for j in idx if j == s) for s in range(len(spans))], "n_tok": len(g["ids"])})
        return {"id": r["id"], "task_type": r["task_type"], "count_k": rw.count_k_for(r), "ref_chars": r.get("ref_chars"), "has_ref": bool(r.get("ref_text")), "gens": gens}

    with ThreadPoolExecutor(a.concurrency) as ex, open(out, "a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(one, rows)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print("SAMPLE", a.model, i + 1, len(rows), flush=True)
    print("SAMPLE_DONE", out, flush=True)


def cmd_score(a) -> None:
    from openai import OpenAI
    from train.ans_reward import AnswerRewarder, sentences
    temp = lambda n: json.load(open(f"outputs/rt/rm/{n}/eval/report.json", encoding="utf-8"))["temperature"]
    critics = dict(kv.split("=", 1) for kv in a.critics)   # 服务上的 LoRA 名 = 数据目录名（取温度）
    sat, refuse = (a.sat.split("=", 1) if a.sat else ("", "")), (a.refuse.split("=", 1) if a.refuse else ("", ""))
    rw = AnswerRewarder(OpenAI(base_url=a.url, api_key="x", timeout=600, max_retries=3), "none", 1.0, ThreadPoolExecutor(a.concurrency),
                        sat[0], "pair", temp(sat[1]) if sat[0] else 1.0, SAT_W, refuse[0], temp(refuse[1]) if refuse[0] else 1.0, REFUSE_W)
    temps = {n: temp(d) for n, d in critics.items()}
    prompts = {r["id"]: r for r in map(json.loads, open(a.prompts, encoding="utf-8"))}
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    recs = [r for r in map(json.loads, open(a.samples, encoding="utf-8")) if r["id"] not in done]

    def one(rec: dict) -> dict:
        row = prompts[rec["id"]]
        res = {"id": rec["id"], "ps": {d: [None] * len(rec["gens"]) for d in critics.values()}, "sat": [None] * len(rec["gens"]), "p_refuse": None}
        for i, g in enumerate(rec["gens"]):
            if g["why"] != "ok":
                continue
            t = g["text"].strip()
            sents = sentences(t)
            for n, d in critics.items():
                res["ps"][d][i] = [round(p, 4) for p in rw.p_answer(row["source"], sents, model=n, temp=temps[n])]
            if sat[0]:
                res["sat"][i] = rw.p_sat(row, t)
        if refuse[0] and any(g["why"] == "refusal" for g in rec["gens"]):
            res["p_refuse"] = rw.p_refuse(row)
        return res

    with ThreadPoolExecutor(a.groups) as ex, open(out, "a", encoding="utf-8") as f:
        for i, res in enumerate(ex.map(one, recs)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print("SCORE", a.out, i + 1, len(recs), flush=True)
    print("SCORE_DONE", out, flush=True)


# ── 离线重算奖励和优势（和训练同一套函数）──

def group_reward(rec: dict, ps: list, sat: list, p_refuse: float | None) -> list[dict]:
    """一组（同一题的 n 条）：返回每条的 reward、整段优势 adv、每句的句子优势和 e。同 AnswerRewarder.score_group + Producer.make_batch。"""
    gens = rec["gens"]
    live = [i for i, g in enumerate(gens) if g["why"] == "ok" and ps[i]]
    out = [{"reward": 0.0, "sent_adv": None, "e": None} for _ in gens]
    if live:
        b, sd = group_baseline([ps[i] for i in live], 0.1)
        for i in live:
            h = count_reward(ps[i], b, rec["count_k"], 0.5, 0.5)
            out[i].update(reward=guarded(h, sat[i] if rec["has_ref"] else None, SAT_W), sent_adv=sentence_adv(ps[i], b, sd, "neg", GAMMA),
                          e=[max(0.0, p - b) for p in ps[i]])
    for i, g in enumerate(gens):
        if g["why"] == "refusal":
            out[i]["reward"] = REFUSE_W * p_refuse if p_refuse is not None else 0.0
    for o, a in zip(out, group_advantages([o["reward"] for o in out], [None] * len(out))):
        o["adv"] = a
    return out


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    i, s = 0, 0.0
    while i < len(xs):   # 并列取平均秩
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        s += sum(1 for k in range(i, j) if xs[k][1]) * (i + j + 1) / 2
        i = j
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def prompt_stats(rec: dict, ps: list, sat: list, p_refuse, gold: list, prev_ps: list | None = None) -> dict:
    """一道题在一个 critic 下的统计量（之后对题取平均 / 自助抽样）。gold：金标 critic 的逐句 P（同 ps 的结构）。"""
    gens = rec["gens"]
    gr = group_reward(rec, ps, sat, p_refuse)
    rewards = [o["reward"] for o in gr]
    m = sum(rewards) / len(rewards)
    std = (sum((r - m) ** 2 for r in rewards) / len(rewards)) ** 0.5
    live = [i for i, o in enumerate(gr) if o["sent_adv"] is not None and gold[i]]
    # token 级：错句（明显 / 任意）和对句（金标 < 0.5）的优势，按 token 数加权；拆成整段项和句子项 β·a
    acc = {k: [0.0, 0.0, 0.0] for k in ("clear", "any", "clean")}   # [token 数, Σ 整段项, Σ 句子项]
    pen = {"mass": 0.0, "mass_clear": 0.0, "mass_any": 0.0, "n_clear": 0, "hit_clear": 0, "n_any": 0, "hit_any": 0, "n_sent": 0, "n_pen": 0}
    inc = {"miss_prev": 0, "new_hit": 0, "new_only": 0, "new_only_clear": 0, "new_only_any": 0}
    crit_pos, crit_neg, e_abs, pb = [], [], 0.0, 0.0
    if prev_ps is not None:
        pl = [i for i in live if prev_ps[i]]
        pb = group_baseline([prev_ps[i] for i in pl], 0.1)[0] if pl else 0.0
    for i in live:
        o = gr[i]
        for j, (g, sa, e, nt) in enumerate(zip(gold[i], o["sent_adv"], o["e"], gens[i]["sent_tok"])):
            for k, c in (("clear", g >= CLEAR), ("any", g >= ANY), ("clean", g < ANY)):
                if c:
                    acc[k][0] += nt
                    acc[k][1] += nt * o["adv"]
                    acc[k][2] += nt * BETA * sa
            e_abs += nt * abs(o["adv"] + BETA * sa)
            pen["n_sent"] += 1
            pen["mass"] += e
            pen["n_pen"] += e >= PEN
            if g >= CLEAR:
                pen["mass_clear"] += e
                pen["n_clear"] += 1
                pen["hit_clear"] += e >= PEN
                crit_pos.append(ps[i][j])
            if g >= ANY:
                pen["mass_any"] += e
                pen["n_any"] += 1
                pen["hit_any"] += e >= PEN
            else:
                crit_neg.append(ps[i][j])
            if prev_ps is not None and prev_ps[i]:
                pe = max(0.0, prev_ps[i][j] - pb)
                if g >= CLEAR and pe < PEN:
                    inc["miss_prev"] += 1
                    inc["new_hit"] += e >= PEN
                if e >= PEN and pe < PEN:
                    inc["new_only"] += 1
                    inc["new_only_clear"] += g >= CLEAR
                    inc["new_only_any"] += g >= ANY
    n_clear = {i: sum(g >= CLEAR for g in gold[i]) for i in live}
    n_any = {i: sum(g >= ANY for g in gold[i]) for i in live}
    top = sorted(live, key=lambda i: -rewards[i])[:max(1, len(live) // 4)] if live else []
    sel = {}
    if live:
        for name, f in (("n_clear", lambda i: n_clear[i]), ("has_clear", lambda i: float(n_clear[i] > 0)), ("n_any", lambda i: n_any[i]),
                        ("n_sent", lambda i: gens[i]["n_sent"]), ("chars", lambda i: gens[i]["chars"]), ("copy", lambda i: gens[i]["copy"]),
                        ("sat", lambda i: sat[i])):
            sel[name] = (mean([f(i) for i in top]), mean([f(i) for i in live]))
    return {"std": std, "n_live": len(live), "n_gens": len(gens), "acc": acc, "pen": pen, "inc": inc, "e_abs": e_abs, "sel": sel,
            "var_clear": bool(live) and 0 < sum(x > 0 for x in n_clear.values()) < len(live), "has_clear": any(x > 0 for x in n_clear.values()),
            "has_any": any(x > 0 for x in n_any.values()), "auc_in": auc(crit_pos, crit_neg), "pos": crit_pos, "neg": crit_neg, "nonzero": std >= 1e-4}


def aggregate(st: list[dict]) -> dict:
    """对题汇总。M1：先在题内算（对句的平均优势 − 错句的平均优势），再对两类 token 都有的题取平均；pooled 是所有题的 token 混在一起。"""
    def gap(s, k, part):   # part：1 = 整段项，2 = 句子项，None = 合计
        a, c = s["acc"][k], s["acc"]["clean"]
        if a[0] == 0 or c[0] == 0:
            return None
        f = (lambda x: x[1] + x[2]) if part is None else (lambda x: x[part])
        return f(c) / c[0] - f(a) / a[0]
    tot = lambda k, f: sum(f(s["acc"][k]) for s in st)
    both = lambda x: x[1] + x[2]
    pooled = lambda k: (tot("clean", both) / tot("clean", lambda x: x[0]) - tot(k, both) / tot(k, lambda x: x[0])) if tot(k, lambda x: x[0]) and tot("clean", lambda x: x[0]) else None
    P = lambda k: sum(s["pen"][k] for s in st)
    I = lambda k: sum(s["inc"][k] for s in st)
    sel = lambda k: mean([s["sel"][k][0] - s["sel"][k][1] for s in st if s["sel"].get(k) and s["sel"][k][0] is not None and s["sel"][k][1] is not None])
    noise = [s for s in st if s["nonzero"] and not s["has_any"]]
    ea = sum(s["e_abs"] for s in st)
    return {
        "n_prompts": len(st),
        "M1_clear": mean([gap(s, "clear", None) for s in st]), "M1_clear_resp": mean([gap(s, "clear", 1) for s in st]), "M1_clear_sent": mean([gap(s, "clear", 2) for s in st]),
        "M1_any": mean([gap(s, "any", None) for s in st]), "M1_clear_pooled": pooled("clear"), "M1_any_pooled": pooled("any"),
        "M1_n": sum(gap(s, "clear", None) is not None for s in st),
        "M2_prec_clear": P("mass_clear") / P("mass") if P("mass") else None, "M2_prec_any": P("mass_any") / P("mass") if P("mass") else None,
        "base_clear": P("n_clear") / max(1, P("n_sent")), "base_any": P("n_any") / max(1, P("n_sent")),
        "M2_recall_clear": P("hit_clear") / P("n_clear") if P("n_clear") else None, "M2_recall_any": P("hit_any") / P("n_any") if P("n_any") else None,
        "pen_rate": P("n_pen") / max(1, P("n_sent")),
        "M3_sel_n_clear": sel("n_clear"), "M3_sel_has_clear": sel("has_clear"), "M3_sel_n_any": sel("n_any"),
        "M4_learnable": mean([float(s["var_clear"] and s["nonzero"]) for s in st]), "M4_has_clear": mean([float(s["has_clear"]) for s in st]),
        "M4_noise_groups": len(noise) / len(st), "M4_noise_adv_share": sum(s["e_abs"] for s in noise) / ea if ea else None,
        "M4_std_median": statistics.median(s["std"] for s in st), "M4_std_lt_0.02": mean([float(s["std"] < 0.02) for s in st]), "M4_std_lt_0.05": mean([float(s["std"] < 0.05) for s in st]),
        "M5_inc_recall": I("new_hit") / I("miss_prev") if I("miss_prev") else None, "M5_miss_prev": I("miss_prev"),
        "M5_inc_prec_clear": I("new_only_clear") / I("new_only") if I("new_only") else None, "M5_inc_prec_any": I("new_only_any") / I("new_only") if I("new_only") else None, "M5_new_only": I("new_only"),
        "M6_n_sent": sel("n_sent"), "M6_chars": sel("chars"), "M6_copy": sel("copy"), "M6_sat": sel("sat"),
        "auc_pooled": auc([p for s in st for p in s["pos"]], [p for s in st for p in s["neg"]]), "auc_within": mean([s["auc_in"] for s in st]),
        "live_rate": sum(s["n_live"] for s in st) / max(1, sum(s["n_gens"] for s in st)),
    }


def boot(st: list[dict], key: str, n: int = 1000, seed: int = 0) -> list[float] | None:
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = aggregate([st[rng.randrange(len(st))] for _ in st])[key]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)] if len(vals) > n // 2 else None


def load_scores(paths: list[str]) -> dict:
    by = {}
    for p in paths:
        tag = Path(p).stem.rsplit("_", 1)[-1]   # scores_<起点>_<2b|9b>.jsonl
        for r in map(json.loads, open(p, encoding="utf-8")):
            d = by.setdefault(r["id"], {"ps": {}, "sat": {}, "p_refuse": {}})
            d["ps"].update(r["ps"])
            d["sat"][tag], d["p_refuse"][tag] = r["sat"], r["p_refuse"]
    return by


SHOW = ("M1_clear", "M1_clear_ci", "M1_clear_pooled", "M2_prec_clear", "base_clear", "M2_recall_clear", "M3_sel_has_clear", "M4_learnable", "M4_noise_groups", "M4_std_median", "auc_pooled", "auc_within")


def cmd_report(a) -> None:
    prev = dict(kv.split("=", 1) for kv in a.prev)   # 起点 = 训出这个起点的 critic（增量命中的参照）
    res = {}
    for start in a.starts:
        recs = [json.loads(l) for l in open(f"{a.dir}/samples_{start}.jsonl", encoding="utf-8")]
        sc = load_scores(sorted(str(p) for p in Path(a.dir).glob(f"scores_{start}_*.jsonl")))
        recs = [r for r in recs if r["id"] in sc and a.gold in sc[r["id"]]["ps"]]
        res[start] = {}
        for c in sorted({c for r in recs for c in sc[r["id"]]["ps"]}):
            if c == a.gold:
                continue
            tag = "9b" if c.endswith("_9b") else "2b"   # 9B 奖励 critic 配 9B 的满意度 / 拒答（同 2×2 里的 9B 打分组）
            st = []
            for r in recs:
                d = sc[r["id"]]
                if c not in d["ps"] or tag not in d["sat"]:
                    continue
                st.append(prompt_stats(r, d["ps"][c], d["sat"][tag], d["p_refuse"][tag], d["ps"][a.gold], d["ps"].get(prev.get(start, ""))))
            agg = aggregate(st)
            agg["M1_clear_ci"] = boot(st, "M1_clear", a.boot)
            agg["M1_clear_pooled_ci"] = boot(st, "M1_clear_pooled", a.boot)
            res[start][c] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()}
            print(start, c, json.dumps({k: res[start][c][k] for k in SHOW}), flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("REPORT ->", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1")
    s.add_argument("--model", default="policy_v0")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--k", type=int, default=200)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--max_tokens", type=int, default=1024)
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_sample)
    s = sub.add_parser("score")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1")
    s.add_argument("--samples", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--critics", nargs="+", required=True, help="服务上的 LoRA 名=数据目录名（outputs/rt/rm/<目录名>/eval/report.json 取温度）")
    s.add_argument("--sat", default="")
    s.add_argument("--refuse", default="")
    s.add_argument("--concurrency", type=int, default=64)
    s.add_argument("--groups", type=int, default=8)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_score)
    s = sub.add_parser("report")
    s.add_argument("--dir", required=True)
    s.add_argument("--starts", nargs="+", required=True)
    s.add_argument("--gold", default="rm1__rm1_gold_9b")
    s.add_argument("--prev", nargs="*", default=[])
    s.add_argument("--boot", type=int, default=1000)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_report)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
