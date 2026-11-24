"""带特权信息的 9B 逐步判断（离线标注，步骤模型的主要标签来源）。

策略在线时看不到的东西，这里全部交给 9B：用户的真实意图（任务设定）、标准动作、整条轨迹（包括这一步之后发生的事）、
最终数据库比对的结果、实际执行的写操作和标准写操作的对照。9B 判的是「知道了标准答案和结局之后，回头看这一步错没错」，
而不是凭表面合理性判断；所以它和被判的策略、以后的步骤模型不在同一个信息水平上。
一次调用判一整条轨迹，输出每个客服步骤的标签（A 正常 / B 查询错误 / C 操作错误 / D 不合规 / E 回复有误或无依据）和理由。

  python -m cs_agent.judge --eps <rollout jsonl> --labels <cs_agent.label 的输出> --url http://127.0.0.1:8001/v1 --out <jsonl> [--select neg+300]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import cs_agent  # noqa: F401
from cs_agent.rules import WRITE

SYSTEM = """You audit a customer-service agent's conversation, step by step, with hindsight.

You are given privileged information the agent did NOT have: the customer's true situation and intent, the correct sequence of actions (gold actions), the actions the agent actually executed, and whether the final database state was correct. Use it to know what the right moves were. But judge each agent step by what a perfect agent with the same information at that moment should have done: a reasonable clarifying question is not an error just because the agent did not know the answer yet.

Labels for each agent step:
A = correct / acceptable
B = query error: looked up the wrong object, an irrelevant lookup, or a pointless repeated lookup
C = operation error: a database-changing call with the wrong object or wrong arguments, or a change that should not have been made
D = policy violation: e.g. acting before authenticating the user, acting without the user's explicit "yes" to the listed details, doing something the policy forbids, refunding to a method the policy does not allow, helping a different user
E = wrong or unsupported statement to the user: a fact, number, status or option that contradicts or is not supported by the tool results or the policy; a misleading claim (e.g. saying something was done when it was not); omitting information the agent was required to tell
If a step has several problems, give the most serious label (C or D over E over B).

The agent's policy:
<policy>
{policy}
</policy>"""

USER = """<privileged_information>
Customer's true situation and intent:
{scenario}

Gold actions (the correct database-changing actions are the ones that matter; lookups may differ):
{gold}

Database-changing actions the agent actually executed (with tool result status):
{executed}

Final database state correct: {db}
</privileged_information>

<conversation>
{conv}
</conversation>

Label every agent step [S1]..[S{n}]. Reply with JSON only, one object per step, in order:
[{{"step": 1, "label": "A", "reason": "<short reason, required unless A>"}}, ...]
Then on the last line, after the JSON: FIRST_ERROR=<step number of the first step that made the outcome wrong or broke the policy, or 0 if none>"""


def render(ep: dict, max_tool: int = 3000) -> tuple[str, list[int]]:
    """对话渲染成带步骤编号的文本；返回 (文本, 每个步骤对应的消息位置)。"""
    lines, idxs = [], []
    for i, m in enumerate(ep["messages"]):
        r = m.get("role")
        if r == "user":
            lines.append(f"CUSTOMER: {m.get('content') or ''}")
        elif r == "tool":
            c = m.get("content") or ""
            lines.append(f"TOOL RESULT{' (ERROR)' if m.get('error') else ''}: {c if len(c) <= max_tool else c[:max_tool] + ' ...[truncated]'}")
        elif r == "assistant":
            if i == 0 and not m.get("tool_calls"):
                lines.append(f"AGENT (greeting): {m.get('content')}")
                continue
            idxs.append(i)
            n = len(idxs)
            for c in m.get("tool_calls") or []:
                lines.append(f"[S{n}] AGENT CALLS {c['name']}({json.dumps(c.get('arguments') or {}, ensure_ascii=False)})")
            if not m.get("tool_calls"):
                lines.append(f"[S{n}] AGENT SAYS: {m.get('content') or ''}")
            elif (m.get("content") or "").strip():
                lines.append(f"      (text sent with the call: {m['content'].strip()[:500]})")
    return "\n".join(lines), idxs


def build(ep: dict, task, policy: str) -> tuple[list[dict], list[int]]:
    gold = [a for a in (task.evaluation_criteria.actions or []) if (a.requestor or "assistant") == "assistant"]
    res = {m.get("id"): m for m in ep["messages"] if m.get("role") == "tool"}
    exe = []
    for m in ep["messages"]:
        for c in m.get("tool_calls") or []:
            if c["name"] in WRITE:
                t = res.get(c.get("id"), {})
                exe.append(f"- {c['name']}({json.dumps(c.get('arguments') or {}, ensure_ascii=False)}) -> {'ERROR: ' + (t.get('content') or '')[:150] if t.get('error') else 'executed'}")
    conv, idxs = render(ep)
    u = USER.format(scenario=str(task.user_scenario), gold="\n".join(f"- {a.name}({json.dumps(a.arguments, ensure_ascii=False)})" for a in gold) or "- (none)",
                    executed="\n".join(exe) or "- (none)", db="yes" if ep.get("reward") == 1.0 else "no", conv=conv, n=len(idxs))
    return [{"role": "system", "content": SYSTEM.format(policy=policy)}, {"role": "user", "content": u}], idxs


def parse(text: str, n: int) -> tuple[dict, int | None] | None:
    text = text.split("</think>")[-1]
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except ValueError:
        return None
    out = {}
    for x in arr:
        if isinstance(x, dict) and isinstance(x.get("step"), int) and 1 <= x["step"] <= n and str(x.get("label", ""))[:1] in "ABCDE":
            out[x["step"]] = {"label": str(x["label"])[:1], "reason": str(x.get("reason", ""))[:300]}
    fe = re.search(r"FIRST_ERROR\s*=\s*(\d+)", text)
    return (out, int(fe.group(1)) if fe else None) if len(out) == n else None


def main() -> None:
    from openai import OpenAI
    from cs_agent.rollout import load_tasks
    from tau2.domains.retail.environment import get_environment
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--split", default="base")
    ap.add_argument("--select", default="neg+0.15", help="neg+F：全部负样本轨迹 + 正样本按比例 F 抽；all：全部")
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t.id: t for t in load_tasks(a.split)}
    policy = get_environment().policy
    eps = {}
    for l in open(a.eps, encoding="utf-8"):
        try:   # 采样还在写的文件，最后一行可能不完整
            e = json.loads(l)
        except ValueError:
            continue
        eps[e["episode_id"]] = e
    labs = [x for x in (json.loads(l) for l in open(a.labels, encoding="utf-8")) if x["episode_id"] in eps]
    if a.select != "all":
        frac = float(a.select.split("+")[1])   # neg+F：全部负样本 + 正样本里按 episode_id 的哈希抽比例 F（续跑时同一条轨迹的去留不变）
        import hashlib
        keep_pos = lambda x: int(hashlib.md5(x["episode_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac
        labs = [x for x in labs if x["traj"] == "neg" or keep_pos(x)]
    done = set()   # 续跑：已经判过的轨迹跳过（判断机和采样同时进行时，隔一段时间补判新采到的）
    try:
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("failed"):
                done.add(r["episode_id"])
    except FileNotFoundError:
        pass
    labs = [x for x in labs if x["episode_id"] not in done]
    print(f"JUDGE {len(labs)} to judge, {len(done)} already judged", flush=True)
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=2)

    def one(lab):
        ep = eps[lab["episode_id"]]
        msgs, idxs = build(ep, tasks[ep["task_id"]], policy)
        raws = []
        # 第一次开思考；解析不出来（实测多半是思考太长、把输出上限用完）第二次关思考直接答
        for t, think in enumerate([bool(a.think), False]):
            r = cl.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, max_tokens=a.max_tokens,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": think}})
            text, fin = r.choices[0].message.content or "", r.choices[0].finish_reason
            p = parse(text, len(idxs))
            if p:
                steps, fe = p
                return {"episode_id": lab["episode_id"], "task_id": ep["task_id"], "traj": lab["traj"], "first_error_step": fe, "think": think,
                        "steps": [{"idx": i, "step": s + 1, **steps[s + 1]} for s, i in enumerate(idxs)]}
            raws.append({"think": think, "finish": fin, "n_chars": len(text), "tail": text[-600:]})
        return {"episode_id": lab["episode_id"], "failed": True, "n_steps": len(idxs), "raw": raws}
    n_ok = 0
    with ThreadPoolExecutor(a.concurrency) as ex, open(a.out, "a", encoding="utf-8") as f:
        futs = [ex.submit(one, x) for x in labs]
        for j, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                r = {"failed": True, "error": repr(e)[:200]}
            n_ok += not r.get("failed")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if j % 50 == 0:
                print(f"  judged {j}/{len(labs)} ok {n_ok}", flush=True)
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    st = collections.Counter(s["label"] for r in rows if not r.get("failed") for s in r["steps"])
    print("JUDGE_DONE", json.dumps({"trajectories": len(rows), "parsed": n_ok, "step_labels": dict(st)}), flush=True)


if __name__ == "__main__":
    main()
