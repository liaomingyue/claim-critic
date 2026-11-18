"""回答有没有答非所问、漏要点（和忠实度分开查；critic 和 9B 都只看有没有幻觉，看不到这两样）。

几组模型在同一批 prompt 上的回答混在一起、不带模型名，一次调用标一批。每条回答标四项：
  addresses：有没有回应任务（yes / partly / no）
  off_topic：有没有和任务无关的内容（none / some / mostly）
  missing：漏没漏原文里回答这个任务必需的要点（none / minor / major），并列出漏的要点
  incomplete：是不是没写完（半句结尾、列表断掉）

  python -m ragtruth.claude_quality_check label --arm best=<目录>:<前缀> --arm early=<目录>:<前缀> --n 120 --batch 20 --max_calls 14 --dir <目录>
  python -m ragtruth.claude_quality_check report --arm ... --base early --dir <目录>
目录:前缀 读 <目录>/<前缀>_{a,b,c}.jsonl；也可以写 名字=@grpo300（读 outputs/rt/minicpm/grpo300{,b,c}/answers.jsonl）、名字=@base（没训过的 2B，只有验收集 a）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.ans_accept import Paced
from ragtruth.synth import parse_json_obj

SYSTEM = """You review answers written by an assistant that was given a source (instructions plus material) and had to answer from that material only.
You do NOT check factual faithfulness here. You check only whether the answer does the job that the source asks for.

For every item give:
- "addresses": "yes" if the answer responds to the task (answers the question asked, or summarizes the given text); "partly" if it responds to only part of it or drifts to something else for a substantial part; "no" if it does not respond to the task.
  For a question whose passages do not contain the answer, saying that the passages do not contain it counts as "yes". Refusing when the passages do contain the answer is "no".
- "off_topic": "none", "some" (one or more sentences that do not serve the task) or "mostly".
- "missing": whether the answer leaves out points of the source that are necessary for this task: "none", "minor" (a secondary point a careful reader would have liked) or "major" (a central point; for a question: a part of the answer the passages give; for a summary: one of the main events or conclusions).
  Respect the length limit stated in the task: do not call a point missing when it could not reasonably fit.
- "missing_points": a list of the missing points, a few words each (empty when "none").
- "incomplete": true if the answer stops before it is finished (ends in the middle of a sentence or of a list).
- "overall": 1 to 5, how well the answer does the job for a user, faithfulness aside.

Judge each item on its own. Reply with JSON only:
{"items": [{"id": "i1", "addresses": "...", "off_topic": "...", "missing": "...", "missing_points": ["..."], "incomplete": false, "overall": 4}, ...]}"""
SETS = {"a": "grpo300", "b": "grpo300b", "c": "grpo300c"}
OK = {"addresses": {"yes", "partly", "no"}, "off_topic": {"none", "some", "mostly"}, "missing": {"none", "minor", "major"}}


def load(spec: str, sets: list[str]) -> tuple[str, dict[str, dict]]:
    name, rest = spec.split("=", 1)
    rows = {}
    if rest == "@base":   # 没训过的 2B，只有验收集 a 的 100 个 prompt；编号换成验收集的写法
        return name, {"g3-" + r["id"][3:]: r for r in read_jsonl(Path("outputs/rt/minicpm/base/answers.jsonl"))}
    for s in sets:
        if rest.startswith("@"):
            path = Path(f"outputs/rt/minicpm/{SETS[s]}/answers.jsonl")
        else:
            d, pre = rest.rsplit(":", 1)
            path = Path(d) / f"{pre}_{s}.jsonl"
        for r in read_jsonl(path):
            rows[r["id"]] = r
    return name, rows


refusal = lambda r: "unable to answer" in r["response"].lower()   # noqa: E731  拒答不标，报告里也去掉（拒答由拒答率单独看）


def valid(x) -> dict | None:
    if not isinstance(x, dict) or any(x.get(k) not in v for k, v in OK.items()) or x.get("overall") not in (1, 2, 3, 4, 5):
        return None
    return {**{k: x[k] for k in OK}, "missing_points": [str(p)[:200] for p in (x.get("missing_points") or [])][:8], "incomplete": bool(x.get("incomplete")), "overall": x["overall"]}


def cmd_label(a) -> None:
    from common.llm import LLM, LLMConfig
    D = Path(a.dir); D.mkdir(parents=True, exist_ok=True)
    arms = dict(load(s, a.sets) for s in a.arm)
    keys = sorted(set.intersection(*(set(v) for v in arms.values())))
    rng = random.Random(a.seed)
    pick = []
    for t in ("QA", "Summary"):   # 两类任务各一半，一个原文最多取一条
        ks = [k for k in keys if next(iter(arms.values()))[k]["task_type"] == t]
        rng.shuffle(ks)
        seen, got = set(), []
        for k in ks:
            sid = next(iter(arms.values()))[k]["source_id"]
            if sid not in seen:
                seen.add(sid); got.append(k)
        pick += got[: a.n // 2]
    if a.all:   # 几组共有的 prompt 全标（已经标过的不重标）
        pick = keys
    else:
        (D / "picks.json").write_text(json.dumps(sorted(pick)), encoding="utf-8")
    path = D / "labels.jsonl"
    done = {(x["arm"], x["id"]) for x in read_jsonl(path)} if path.exists() else set()
    llm = Paced(LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=1)), a.gap)
    calls = 0
    for rnd in range(2):
        todo = [(n, k) for k in sorted(pick) for n in sorted(arms) if (n, k) not in done and not refusal(arms[n][k])]
        random.Random(a.seed + rnd).shuffle(todo)
        batches, cur, srcs = [], [], set()
        for n, k in todo:   # 同一批里一个 prompt 只出现一次，免得两组的回答互相对照
            if k in srcs or len(cur) >= a.batch:
                batches.append(cur); cur, srcs = [], set()
            cur.append((n, k)); srcs.add(k)
        batches += [cur] if cur else []
        if not batches:
            break
        assert calls + len(batches) <= a.max_calls, f"Claude 调用预算：{calls + len(batches)} > {a.max_calls}"
        print(f"round {rnd}: {len(todo)} answers in {len(batches)} calls", flush=True)
        for b in batches:
            ids = {f"i{j + 1}": nk for j, nk in enumerate(b)}
            msg = "\n\n".join(f"<item id=\"{i}\">\n<source>\n{arms[n][k]['source'].strip()}\n</source>\n\n<answer>\n{arms[n][k]['response'].strip()}\n</answer>\n</item>" for i, (n, k) in ids.items())
            try:
                o = parse_json_obj(llm.chat_one([{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]))
            except Exception as e:   # noqa: BLE001
                print("  call failed", type(e).__name__, flush=True); o = None
            calls += 1
            got = {x.get("id"): valid(x) for x in (o.get("items") or []) if isinstance(x, dict)} if isinstance(o, dict) else {}
            new = 0
            with open(path, "a", encoding="utf-8") as f:
                for i, (n, k) in ids.items():
                    if got.get(i):
                        f.write(json.dumps({"arm": n, "id": k, "task_type": arms[n][k]["task_type"], "source_id": arms[n][k]["source_id"], **got[i]}, ensure_ascii=False) + "\n")
                        done.add((n, k)); new += 1
            print(f"  call {calls}: +{new}, total {len(done)}", flush=True)
    print("QUALITY_LABEL_DONE", len(done), "answers,", calls, "calls", flush=True)


def cmd_report(a) -> None:
    D = Path(a.dir)
    arms = dict(load(s, a.sets) for s in a.arm)
    L = collections.defaultdict(dict)
    for x in read_jsonl(D / "labels.jsonl"):
        L[x["arm"]][x["id"]] = x
    keys = sorted(k for k in set.intersection(*(set(L[n]) for n in arms)) if not any(refusal(arms[n][k]) for n in arms))
    F = {"not_addressed": lambda x: x["addresses"] == "no", "partly_or_not": lambda x: x["addresses"] != "yes", "off_topic": lambda x: x["off_topic"] != "none",
         "missing_major": lambda x: x["missing"] == "major", "missing_any": lambda x: x["missing"] != "none", "incomplete": lambda x: x["incomplete"], "overall": lambda x: x["overall"]}
    rng = random.Random(0)
    rep = {"pairs": len(keys), "arms": {}, "diff": {}}
    for n in arms:
        rep["arms"][n] = {}
        for t in ("all", "QA", "Summary"):
            ks = [k for k in keys if t == "all" or L[n][k]["task_type"] == t]
            rep["arms"][n][t] = {"n": len(ks), **{f: round(sum(g(L[n][k]) for k in ks) / max(1, len(ks)), 4) for f, g in F.items()},
                                 "chars": round(sum(len(arms[n][k]["response"]) for k in ks) / max(1, len(ks)), 1), "sents": round(sum(len(arms[n][k]["sentences"]) for k in ks) / max(1, len(ks)), 2)}
    for n in arms:
        if n == a.base:
            continue
        rep["diff"][f"{n} - {a.base}"] = {}
        for f, g in F.items():
            d = [g(L[n][k]) - g(L[a.base][k]) for k in keys]
            bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
            rep["diff"][f"{n} - {a.base}"][f] = {"delta": round(sum(d) / len(d), 4), "ci90": [round(bs[100], 4), round(bs[1899], 4)]}
    (D / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, v in rep["arms"].items():
        for t, x in v.items():
            print(n, t, x)
    for k, v in rep["diff"].items():
        print(k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "report"])
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--sets", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--n", type=int, default=120, help="抽多少个 prompt（QA、Summary 各一半）")
    ap.add_argument("--all", action="store_true", help="几组共有的 prompt 全标，不抽样")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max_calls", type=int, default=14)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--gap", type=float, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"label": cmd_label, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
