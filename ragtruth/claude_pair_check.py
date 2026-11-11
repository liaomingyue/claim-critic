"""关键对照的 Claude 独立验证：两组 RAG 回答（同一批验收原文和种子，配对），Claude 按金标 v1 逐句盲标，
和 9B 评测器的结论对照。用来检查「9B 策略 × 9B critic 训练、9B 评测器打分」会不会被同源的盲区抬高。

  python -m ragtruth.claude_pair_check label_batch --arm q9b_s200=<目录>:<前缀> --arm final_2b=<目录>:<前缀> [--batch 10]
  python -m ragtruth.claude_pair_check label --arm ...            # 一条一次调用（贵 10 倍，只留作批量口径的抽查）
  python -m ragtruth.claude_pair_check report --arm ... --base final_2b

不调用 Claude、不用 GPU，从仓库里的发布结果重算 9B 对照的 Claude 核对（表 10）：
  python -m ragtruth.claude_pair_check report --dir results/q9b_claude_check --base final_2b       --arm q9b_s200=results/q9b_claude_check/q9b_s200:s200 --arm final_2b=results/q9b_claude_check/final_2b:s100

- 批量：一次调用标 --batch 条（默认 10）。批量曾改变过标注口径（实验记录 1.5：同一原文的 3 条一起标，显然错标得更多），
  所以同一批里原文互不相同（配对的两条永远不在一批，Claude 没法并排比较同一原文的回答），每批两组各一半、顺序打乱，
  批量带来的偏差两组一样；这次的全部回答都用同一种方式标，不和一条一次的混用。两组回答都看不到来自哪个模型；
- 提示词和格式检查与验收金标相同（ragtruth.gold.label），结果可续跑；
- 调用上限 --max_calls（默认 400）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.ans_accept import Paced, answer_flags, compare
from ragtruth.gold import SYSTEM, VERSION, label, valid, write_final
from ragtruth.synth import parse_json_obj

D = Path("outputs/rt/core/claude_check")


def load_arms(specs: list[str], sets: list[str]) -> dict[str, dict[str, dict]]:
    """{组: {回答 id: 回答}}；回答 id 改成 <组>-<套><原文>-<种子>，配对键（answer_flags 取第一个 - 之后）跨组一致、跨套不撞。"""
    arms = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        rows = {}
        for s in sets:
            for r in read_jsonl(Path(d) / f"{prefix}_{s}.jsonl"):
                if r.get("finish") != "stop" or not r["sentences"]:
                    continue
                src_seed = r["id"].split("-", 1)[1]
                rid = f"{name}-{s}{src_seed}"
                rows[rid] = {**r, "id": rid, "orig_id": r["id"], "set": s}
        arms[name] = rows
    return arms


def paired_keys(arms: dict[str, dict]) -> list[str]:
    ks = [{rid.split("-", 1)[1] for rid in rows} for rows in arms.values()]
    return sorted(set.intersection(*ks))


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig
    arms = load_arms(a.arm, a.sets)
    keys = set(paired_keys(arms))
    items = [{"id": f"{rid}#p1", "r": r} for rows in arms.values() for rid, r in rows.items() if rid.split("-", 1)[1] in keys]
    random.Random(a.seed).shuffle(items)   # 两组交错送，Claude 看不出顺序
    assert len(items) <= a.max_calls, f"Claude 调用预算：{len(items)} > {a.max_calls}"
    D.mkdir(parents=True, exist_ok=True)
    llm = Paced(LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency)), a.gap)
    print("LABEL", len(items), "answers,", len(keys), "pairs", flush=True)
    label(llm, items, D / "labels.jsonl", a.concurrency)
    by = collections.defaultdict(dict)
    for x in read_jsonl(D / "labels.jsonl"):
        if not x.get("_dropped"):
            by[x["resp_id"]][1] = x["sentences"]
    for name, rows in arms.items():
        picks = {rid: r for rid, r in rows.items() if rid in by}
        write_final(picks, {rid: by[rid] for rid in picks}, D / f"final_{name}.jsonl")
    print("LABEL_DONE", len(by), "of", len(items), flush=True)


SYSTEM_BATCH = SYSTEM.rsplit("Reply with one JSON object only:", 1)[0] + (
    "You will get several unrelated items, each with its own source and response. Judge every item on its own, using only its own source; "
    "do not compare items with each other.\n\n"
    'Reply with one JSON object only: {"items": [{"id": "<item id>", "sentences": [{"idx": 1, "is_error": true|false, "type": "...", "mechanism": "...", '
    '"severity": "...", "evidence": "...", "note": "..."}, ...]}, ...]} with one entry per item and exactly one entry per sentence of that item, in order.')


def batch_msg(rs: list[tuple[str, dict]]) -> str:
    parts = []
    for bid, r in rs:
        sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(r["sentences"], 1))
        parts.append(f"<item id=\"{bid}\">\n<source>\n{r['source'].strip()}\n</source>\n\n<response>\n{sents}\n</response>\n</item>")
    return "\n\n".join(parts)


def make_batches(arms: dict[str, dict], keys: list[str], size: int, seed: int) -> list[list[dict]]:
    """每批两组各 size/2 条，同一批里原文互不相同；批内顺序打乱。"""
    rng = random.Random(seed)
    names = sorted(arms)
    pools = {n: [arms[n][f"{n}-{k}"] for k in keys] for n in names}
    for n in names:
        rng.shuffle(pools[n])
    batches = []
    while any(pools.values()):
        b, srcs = [], set()
        for n in names:
            take = 0
            for r in list(pools[n]):
                if take >= size // len(names):
                    break
                if r["source_id"] in srcs:
                    continue
                b.append(r); srcs.add(r["source_id"]); pools[n].remove(r); take += 1
        rng.shuffle(b)
        batches.append(b)
    return batches


def cmd_label_batch(a) -> None:
    from common.llm import LLM, LLMConfig
    from concurrent.futures import ThreadPoolExecutor
    arms = load_arms(a.arm, a.sets)
    keys = paired_keys(arms)
    path = D / "labels_batch.jsonl"
    done = {x["resp_id"] for x in read_jsonl(path)} if path.exists() else set()
    if a.n_pairs:   # 只抽 n_pairs 对：已经标过一条的配对优先（不浪费已花的调用），其余随机；各套均分
        started = {rid.split("-", 1)[1] for rid in done}
        rng = random.Random(a.seed)
        pick = []
        for s in a.sets:
            ks = [k for k in keys if k.startswith(s)]
            first = sorted(k for k in ks if k in started)
            rest = sorted(k for k in ks if k not in started)
            rng.shuffle(rest)
            pick += (first + rest)[: a.n_pairs // len(a.sets)]
        keys = sorted(pick)
        arms = {n: {rid: r for rid, r in rows.items() if rid.split("-", 1)[1] in set(keys)} for n, rows in arms.items()}
        (D / "pairs.json").write_text(json.dumps(keys), encoding="utf-8")
        print("pairs", len(keys), "already started", len(started & set(keys)), flush=True)
    llm = Paced(LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency)), a.gap)
    calls = 0
    for rnd in range(3):   # 第一轮全部；格式不合格的条目下一轮重新分批
        todo_keys = [k for k in keys if any(f"{n}-{k}" not in done for n in arms)]
        sub = {n: {rid: r for rid, r in rows.items() if rid not in done} for n, rows in arms.items()}
        batches = [b for b in make_batches({n: {**arms[n]} for n in arms}, todo_keys, a.batch, a.seed + rnd) if b]
        batches = [[r for r in b if r["id"] in sub[r["id"].split("-", 1)[0]]] for b in batches]
        batches = [b for b in batches if b]
        if not batches:
            break
        assert calls + len(batches) <= a.max_calls, f"Claude 调用预算：{calls + len(batches)} > {a.max_calls}"
        print(f"round {rnd}: {sum(map(len, batches))} answers in {len(batches)} calls", flush=True)

        def run(ib):
            i, b = ib
            ids = {f"i{j + 1}": r for j, r in enumerate(b)}   # 批内编号，和模型、组无关
            out = llm.chat_one([{"role": "system", "content": SYSTEM_BATCH}, {"role": "user", "content": batch_msg(list(ids.items()))}])
            o = parse_json_obj(out)
            got = {x.get("id"): x.get("sentences") for x in (o.get("items") or []) if isinstance(x, dict)} if isinstance(o, dict) else {}
            rows = []
            for bid, r in ids.items():
                xs = valid({"sentences": got.get(bid)}, len(r["sentences"]))
                if xs is not None:
                    rows.append({"id": f"{r['id']}#b{rnd}", "resp_id": r["id"], "pass": 1, "version": VERSION, "batch": f"r{rnd}b{i}", "batch_size": len(b), "sentences": xs})
            return rows

        with ThreadPoolExecutor(a.concurrency) as ex:
            for rows in ex.map(run, enumerate(batches)):
                calls += 1
                with open(path, "a", encoding="utf-8") as f:
                    for x in rows:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                done |= {x["resp_id"] for x in rows}
                print(f"  call {calls}: +{len(rows)} labeled, total {len(done)}", flush=True)
    by = {x["resp_id"]: {1: x["sentences"]} for x in read_jsonl(path)}
    for name, rows in arms.items():
        picks = {rid: r for rid, r in rows.items() if rid in by}
        write_final(picks, {rid: by[rid] for rid in picks}, D / f"final_{name}.jsonl")
    print("LABEL_BATCH_DONE", len(by), "answers,", calls, "calls", flush=True)


def nine_b_flags(arms: dict[str, dict], specs: list[str]) -> dict[str, dict[str, dict]]:
    """9B 评测器在同一批回答上的口径：一句 P ≥ 0.9 记含显然错，P ≥ 0.5 记含任意错。"""
    out = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        d, prefix = rest.rsplit(":", 1)
        p = {}
        for s in {r["set"] for r in arms[name].values()}:
            for x in json.load(open(Path(d) / f"{prefix}_{s}_9b.json", encoding="utf-8")):
                rid, k = x["id"].rsplit("-S", 1)
                p.setdefault(f"{s}{rid.split('-', 1)[1]}", []).append(x["p"])
        out[name] = {k: {"any_clear": float(max(v) >= 0.9), "any_error": float(max(v) >= 0.5), "err_rate": sum(q >= 0.5 for q in v) / len(v)} for k, v in p.items()}
    return out


def kappa(a: list[float], b: list[float]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def cmd_report(a) -> None:
    arms = load_arms(a.arm, a.sets)
    if (D / "pairs.json").exists():
        ks = set(json.loads((D / "pairs.json").read_text()))
        arms = {n: {rid: r for rid, r in rows.items() if rid.split("-", 1)[1] in ks} for n, rows in arms.items()}
    cl = {name: answer_flags(list(read_jsonl(D / f"final_{name}.jsonl"))) for name in arms}
    nb = nine_b_flags(arms, a.arm)
    keys = sorted(set.intersection(*[set(v) for v in cl.values()]))
    src = {rid.split("-", 1)[1]: r["source_id"] for rows in arms.values() for rid, r in rows.items()}
    refuse = {name: round(sum("unable to answer" in rows[f"{name}-{k}"]["response"].lower() for k in keys) / len(keys), 4) for name, rows in arms.items()}
    rep = {"pairs": len(keys), "refusal": refuse, "claude": {}, "9b": {}, "agreement": {}}
    for name in arms:
        if name == a.base:
            continue
        rep["claude"][f"{name} - {a.base}"] = compare({k: cl[a.base][k] for k in keys}, {k: cl[name][k] for k in keys}, src)
        rep["9b"][f"{name} - {a.base}"] = compare({k: nb[a.base][k] for k in keys}, {k: nb[name][k] for k in keys}, src)
    for name in arms:
        rep["agreement"][name] = {f: {"claude": round(sum(cl[name][k][f] for k in keys) / len(keys), 4), "9b": round(sum(nb[name][k][f] for k in keys) / len(keys), 4),
                                      "kappa": kappa([cl[name][k][f] for k in keys], [nb[name][k][f] for k in keys])} for f in ("any_clear", "any_error")}
    out = D / "report.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main() -> None:
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "label_batch", "report"])
    ap.add_argument("--arm", action="append", required=True, help="名=目录:前缀")
    ap.add_argument("--sets", nargs="+", default=["a", "b"])
    ap.add_argument("--base", default="")
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--max_calls", type=int, default=400)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=0, help="只抽这么多对（0 = 全部）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default="outputs/rt/core/claude_check", help="标注和报告所在目录（发布的结果在 results/q9b_claude_check）")
    a = ap.parse_args()
    D = Path(a.dir)
    D.mkdir(parents=True, exist_ok=True)
    {"label": cmd_label, "label_batch": cmd_label_batch, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
