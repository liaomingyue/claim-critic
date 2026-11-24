"""客服轨迹：Claude 盲判仲裁，测 9B 特权判断（cs_agent.judge2）和代码标注（cs_agent.label）的准确率。

  1. sample（在服务器上跑，要 τ² 环境）：按分层抽样，把每条轨迹渲染成带步骤编号的文本 + 特权信息，连同 9B 和代码的结论写成一个小文件
       python -m scripts.analysis.cs_claude_arb sample --dir outputs/rt/cs/sample_r0 --out outputs/rt/cs/claude_arb/bundle.jsonl
  2. label（本地，调 Claude）：每次调用 2–3 条，Claude 看不到 9B 和代码的结论，独立判
       python -m scripts.analysis.cs_claude_arb label --bundle ... --out .../claude.jsonl --max_calls 60
  3. report：和 9B、代码逐条对照
       python -m scripts.analysis.cs_claude_arb report --bundle ... --labels .../claude.jsonl

分层（失败的轨迹看第一个错误，成功的轨迹看违规）：
  earlier   9B 指的第一个错误比代码早（50）       → 9B 找的是不是真正的根源
  only9b    只有 9B 定位到错误，优先回复有误 E（50） → 9B 的「回复有误」准不准
  succ9b    成功轨迹里只有 9B 报违规（全部）        → 语义违规是不是真的
  same      两边指向同一步（28）                    → 对照：9B 的基础准确率
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

SOFT = {"R5", "R2b", "R10"}
QUOTA = {"earlier": 50, "only9b": 50, "succ9b": 100, "same": 28}

SYSTEM = """You are an expert auditor of customer-service agent conversations in a retail domain. For each conversation you get the agent policy, privileged information the agent did not have (the customer's true intent, the correct actions, the actions the agent actually executed, whether the final database state was correct), and the conversation with numbered agent steps [S1], [S2], ...

Labels: B = query error (wrong/irrelevant/pointlessly repeated lookup); C = operation error (database-changing call with wrong object or arguments, or a change that should not be made); D = policy violation (acting before authentication, without the customer's explicit yes to the listed details, doing what the policy forbids, wrong refund destination, revealing private data, promises outside the policy); E = wrong or unsupported statement to the customer (contradicts or is not supported by tool results or policy, misleading claims, wrong or missing answers to the customer's questions).
Judge each step by what a perfect agent with the same information at that moment should have done; use the privileged information only to know the truth.

For a conversation whose database state is WRONG: give the FIRST critical error, the earliest agent step that made the outcome wrong or broke the policy.
For a conversation whose database state is CORRECT: list every step that is a violation (B/C/D) or a wrong/missing answer (E); usually there are none.

Reply with JSON only: {"items": [{"id": "<id>", "first_error_step": <n>, "label": "<B|C|D|E>", "reason": "<short>"} or {"id": "<id>", "problems": [{"step": <n>, "label": "<B|C|D|E>", "reason": "<short>"}]}, ...]}"""


def _load(path: Path, failed_ok: bool = False) -> dict:
    out = {}
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if failed_ok or not r.get("failed"):
            out[r["episode_id"]] = r
    return out


def sample(a) -> None:
    import cs_agent  # noqa: F401
    from cs_agent.judge import render
    from cs_agent.rollout import load_tasks
    from cs_agent.rules import WRITE
    from tau2.domains.retail.environment import get_environment
    tasks = {t.id: t for t in load_tasks("train")}
    policy = get_environment().policy
    D = Path(a.dir)
    pools = collections.defaultdict(list)
    for p in a.pols:
        J = _load(D / f"{p}_judge2.jsonl")
        L = _load(D / f"{p}_labels.jsonl", True)
        for eid, j in J.items():
            c = L.get(eid)
            if c is None:
                continue
            if j["mode"] == "failed":
                jidx = j["step_idx"][j["first_error_step"] - 1]
                if c["first_err"] is None:
                    pools["only9b_E" if j["label"] == "E" else "only9b"].append((p, eid))
                elif jidx < c["first_err"]:
                    pools["earlier"].append((p, eid))
                elif jidx == c["first_err"]:
                    pools["same"].append((p, eid))
            else:
                jv = {j["step_idx"][v["step"] - 1] for v in j["violations"]} | {j["step_idx"][v["step"] - 1] for v in j.get("info_errors", [])}
                cv = {v["idx"] for v in c["violations"] if v["rule"] not in SOFT and v["sev"] in ("S1", "S2")}
                if jv and not cv:
                    pools["succ9b"].append((p, eid))
    rng = random.Random(0)
    pick = []
    for k in ("earlier", "same", "succ9b"):
        xs = sorted(pools[k])
        rng.shuffle(xs)
        pick += [(k, x) for x in xs[:QUOTA[k]]]
    e, o = sorted(pools["only9b_E"]), sorted(pools["only9b"])
    rng.shuffle(e), rng.shuffle(o)
    pick += [("only9b", x) for x in (e + o)[:QUOTA["only9b"]]]
    want = collections.defaultdict(set)
    for _, (p, eid) in pick:
        want[p].add(eid)
    eps = {}
    for p, ids in want.items():
        for l in open(D / f"{p}_train.jsonl", encoding="utf-8"):
            try:
                ep = json.loads(l)
            except ValueError:
                continue
            if ep["episode_id"] in ids:
                eps[(p, ep["episode_id"])] = ep
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, (p, eid) in pick:
            ep = eps[(p, eid)]
            task = tasks[ep["task_id"]]
            conv, idxs = render(ep)
            res = {m.get("id"): m for m in ep["messages"] if m.get("role") == "tool"}
            exe = [f"- {c['name']}({json.dumps(c.get('arguments') or {}, ensure_ascii=False)}) -> "
                   + ("ERROR: " + (res.get(c.get('id'), {}).get('content') or '')[:150] if res.get(c.get('id'), {}).get('error') else "executed")
                   for m in ep["messages"] for c in (m.get("tool_calls") or []) if c["name"] in WRITE]
            gold = [f"- {x.name}({json.dumps(x.arguments, ensure_ascii=False)})" for x in (task.evaluation_criteria.actions or []) if (x.requestor or "assistant") == "assistant"]
            f.write(json.dumps({"id": f"{p}:{eid}", "stratum": k, "policy_name": p, "db_ok": ep["reward"] == 1.0, "n_steps": len(idxs), "step_idx": idxs,
                                "privileged": f"Customer's true situation and intent:\n{task.user_scenario}\n\nGold actions:\n" + ("\n".join(gold) or "- (none)")
                                + "\n\nDatabase-changing actions the agent executed:\n" + ("\n".join(exe) or "- (none)")
                                + f"\n\nFinal database state correct: {'yes' if ep['reward'] == 1.0 else 'no'}",
                                "conversation": conv}, ensure_ascii=False) + "\n")
    # 9B 和代码的结论单独存（Claude 看不到）
    JJ = {p: _load(D / f"{p}_judge2.jsonl") for p in a.pols}
    LL = {p: _load(D / f"{p}_labels.jsonl", True) for p in a.pols}
    with open(out.with_name("answers.jsonl"), "w", encoding="utf-8") as f:
        for k, (p, eid) in pick:
            j, c = JJ[p][eid], LL[p][eid]
            f.write(json.dumps({"id": f"{p}:{eid}", "stratum": k, "judge9b": j, "code_first_err": c["first_err"],
                                "code_steps": c["steps"], "code_violations": c["violations"]}, ensure_ascii=False) + "\n")
    (out.with_name("policy.txt")).write_text(policy, encoding="utf-8")
    print("SAMPLED", dict(collections.Counter(k for k, _ in pick)), {k: len(v) for k, v in pools.items()}, flush=True)


def label(a) -> None:
    from common.llm import LLM, LLMConfig
    from ragtruth.ans_accept import Paced
    from ragtruth.synth import parse_json_obj
    B = [json.loads(l) for l in open(a.bundle, encoding="utf-8")]
    policy = Path(a.bundle).with_name("policy.txt").read_text(encoding="utf-8")
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    todo = [b for b in B if b["id"] not in done]
    random.Random(1).shuffle(todo)   # 同一次调用里混不同分层，Claude 看不出抽样来源
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    assert len(batches) <= a.max_calls, f"needs {len(batches)} calls > budget {a.max_calls}"
    llm = Paced(LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency, max_tokens=4096)), a.gap)
    sys = SYSTEM + "\n\nThe agent's policy:\n<policy>\n" + policy + "\n</policy>"

    def run(batch):
        body = "\n\n".join(f"=== Conversation id={b['id']} (database state {'CORRECT' if b['db_ok'] else 'WRONG'}; agent steps S1..S{b['n_steps']}) ===\n"
                           f"<privileged_information>\n{b['privileged']}\n</privileged_information>\n<conversation>\n{b['conversation']}\n</conversation>" for b in batch)
        txt = llm.chat_one([{"role": "system", "content": sys}, {"role": "user", "content": body}])
        o = parse_json_obj(txt)
        items = {x.get("id"): x for x in (o.get("items") or []) if isinstance(x, dict)} if isinstance(o, dict) else {}
        return [{"id": b["id"], **({"claude": items[b["id"]]} if b["id"] in items else {"missing": True})} for b in batch]
    from concurrent.futures import ThreadPoolExecutor
    n = 0
    with ThreadPoolExecutor(a.concurrency) as ex, open(out, "a", encoding="utf-8") as f:
        for rows in ex.map(run, batches):
            n += 1
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  call {n}/{len(batches)}", flush=True)
    print("LABEL_DONE", n, "calls", flush=True)


def report(a) -> None:
    B = {json.loads(l)["id"]: json.loads(l) for l in open(a.bundle, encoding="utf-8")}
    A = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.bundle).with_name("answers.jsonl"), encoding="utf-8")}
    C = {}
    for l in open(a.labels, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("missing"):
            C[r["id"]] = r["claude"]
    st = collections.defaultdict(collections.Counter)
    for i, c in C.items():
        b, ans = B[i], A[i]
        k, j = ans["stratum"], ans["judge9b"]
        if not b["db_ok"]:
            cs = c.get("first_error_step")
            if not isinstance(cs, int):
                st[k]["claude_unparsed"] += 1
                continue
            cidx = b["step_idx"][cs - 1] if 1 <= cs <= len(b["step_idx"]) else None
            jidx = j["step_idx"][j["first_error_step"] - 1]
            st[k]["n"] += 1
            st[k]["9b_step_match"] += cidx == jidx
            st[k]["9b_step_within1"] += cidx is not None and abs(b["step_idx"].index(cidx) - b["step_idx"].index(jidx)) <= 1
            st[k]["9b_label_match"] += cidx == jidx and c.get("label") == j["label"]
            if ans["code_first_err"] is not None:
                st[k]["code_n"] += 1
                st[k]["code_step_match"] += cidx == ans["code_first_err"]
        else:
            cl = {b["step_idx"][x["step"] - 1] for x in c.get("problems") or [] if isinstance(x, dict) and isinstance(x.get("step"), int) and 1 <= x["step"] <= len(b["step_idx"])}
            jl = {j["step_idx"][v["step"] - 1] for v in j["violations"]} | {j["step_idx"][v["step"] - 1] for v in j.get("info_errors", [])}
            st[k]["n"] += 1
            st[k]["9b_flag_confirmed"] += bool(jl & cl)
            st[k]["claude_any_problem"] += bool(cl)
    for k, v in st.items():
        n = max(1, v["n"])
        line = {kk: f"{vv} ({vv / n:.0%})" if kk not in ("n", "code_n") else vv for kk, vv in v.items()}
        if v.get("code_n"):
            line["code_step_match"] = f"{v['code_step_match']} ({v['code_step_match'] / v['code_n']:.0%} of {v['code_n']})"
        print(k, json.dumps(line, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "label", "report"])
    ap.add_argument("--dir", default="outputs/rt/cs/sample_r0")
    ap.add_argument("--pols", nargs="+", default=["base9", "sftB"])
    ap.add_argument("--out", default="outputs/rt/cs/claude_arb/bundle.jsonl")
    ap.add_argument("--bundle", default="outputs/rt/cs/claude_arb/bundle.jsonl")
    ap.add_argument("--labels", default="outputs/rt/cs/claude_arb/claude.jsonl")
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--max_calls", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gap", type=float, default=3.0)
    a = ap.parse_args()
    if a.cmd == "label":
        a.out = a.labels
    {"sample": sample, "label": label, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
