"""APIGen-MT 的 retail 对话 → GRPO 用的新任务（不需要标签）。

GRPO 的奖励（结果模型的 P(正确)、过程 critic）只看轨迹本身，新任务只差一样东西：用户模拟器的剧本。
这里让 9B 只读对话里用户说的话，按 τ² 的格式写出剧本（来电原因含有条件的偏好、已知信息、不知道的信息、性格），
不给它看客服查出来的东西（订单详情等），免得剧本泄露答案；用户自己报出的商品 id 也换成属性描述
（τ² 原题里用户只描述商品属性，让客服自己去查；剧本里带 id 的任务会简单很多），剧本里出现 10 位数字就重写。
另外把原对话里成功执行的写操作记成参考动作：τ² 的评测器可以重放它们得到参考的数据库状态，
只用来监控（看结果模型在没见过的任务上准不准），不进奖励。

  python -m cs_agent.apigen_tasks build --src outputs/rt/cs/sft_data/sft_B.jsonl --url <vLLM /v1> --n 50 --out outputs/rt/cs/apigen_tasks/pilot50.jsonl
  load_tasks_file(path) → τ² Task 列表（cs_agent.evaluate --tasks_file、cs_agent.grpo --extra_tasks 用）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cs_agent  # noqa: F401
from cs_agent.rules import WRITE

PROMPT = """Below are only the CUSTOMER's messages from a customer-service conversation with an online retail store. Write the instructions for a user simulator that will play this same customer in a new conversation, in the style of these fields:

- reason_for_call: what the customer wants done and why, in second person ("You want to ..."). Include every request the customer made, in order, and any conditions or preferences they stated (e.g. "if X is not available, you would rather Y"; which payment method they want a refund to). Describe every item by its product type and attributes (color, size, material, etc.). NEVER write an item id or product id (long numbers): if the customer used one, replace it with that item's attributes from the table below.
- known_info: identity details the customer gave (name, zip code, email, order numbers they mentioned), in second person ("You are ... in zip code ...").
- unknown_info: things the customer said they did not know or remember (empty string if none).
- task_instructions: a short note on how the customer behaves (tone, patience, whether they confirm details).

Only use what the customer said. Do not invent details. Reply with one JSON object with exactly these four string fields.

Item ids mentioned by the customer, with what they are (use these to describe the items; do not copy the ids):
{items}

Customer messages:
{turns}"""

ID_RE = re.compile(r"\b\d{10}\b")


def item_table(msgs: list[dict]) -> dict:
    """对话里工具返回的商品信息：item_id → 「商品名（属性）」。用来把用户嘴里的商品 id 换成属性描述。"""
    out = {}

    def walk(x, name=None):
        if isinstance(x, dict):
            nm = x.get("name") if isinstance(x.get("name"), str) else name
            if isinstance(x.get("item_id"), str) and isinstance(x.get("options"), dict):
                out[x["item_id"]] = f"{nm or 'item'} ({', '.join(f'{k}: {v}' for k, v in x['options'].items())})"
            for v in x.values():
                walk(v, nm)
        elif isinstance(x, list):
            for v in x:
                walk(v, name)
    for m in msgs:
        if m.get("role") == "tool":
            try:
                walk(json.loads(m.get("content") or ""))
            except ValueError:
                pass
    return out


def _hash(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def gold_writes(msgs: list[dict]) -> list[dict]:
    """原对话里成功执行的写操作（工具没报错的）。"""
    res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    out = []
    for m in msgs:
        for c in m.get("tool_calls") or []:
            t = res.get(c.get("id"), {})
            if c["name"] in WRITE and not str(t.get("content", "")).startswith("Error"):
                out.append({"action_id": f"a{len(out)}", "requestor": "assistant", "name": c["name"], "arguments": c.get("arguments") or {}})
    return out


def build(a) -> None:
    from openai import OpenAI
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if gold_writes(r["messages"])]   # 要有写操作：只查询的对话没法用数据库状态核对
    rows.sort(key=lambda r: _hash(f"{a.seed}:{r['id']}"))
    rows = rows[: a.n]
    cl = OpenAI(base_url=a.url, api_key=a.key, timeout=600, max_retries=2)

    def one(r):
        turns = "\n".join(f"- {m['content']}" for m in r["messages"] if m["role"] == "user" and (m.get("content") or "").strip())
        tab = item_table(r["messages"])
        used = sorted(set(ID_RE.findall(turns)))
        items = "\n".join(f"- {i}: {tab.get(i, 'unknown item')}" for i in used) or "- (none)"
        for t in range(4):
            out = cl.chat.completions.create(model="base", messages=[{"role": "user", "content": PROMPT.format(turns=turns, items=items)}], temperature=0.0 if t == 0 else 0.7, max_tokens=1200,
                                             extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content or ""
            m = re.search(r"\{.*\}", out, re.S)
            try:
                d = json.loads(m.group(0)) if m else None
            except ValueError:
                d = None
            if isinstance(d, dict) and all(isinstance(d.get(k), str) for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions")) \
                    and not ID_RE.search(" ".join(d[k] for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions"))):   # 剧本里不许有商品 id
                break
        else:
            return None
        return {"id": f"apigen:{r['id']}", "source_id": r["id"],
                "user_scenario": {"persona": None, "instructions": {"domain": "retail", "reason_for_call": d["reason_for_call"], "known_info": d["known_info"],
                                                                     "unknown_info": d["unknown_info"] or None, "task_instructions": d["task_instructions"]}},
                "evaluation_criteria": {"actions": gold_writes(r["messages"]), "reward_basis": ["DB"]}}
    with ThreadPoolExecutor(a.concurrency) as ex:
        tasks = [t for t in ex.map(one, rows) if t]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"APIGEN_TASKS {len(tasks)} / {len(rows)} -> {a.out}", flush=True)


def load_tasks_file(path: str) -> list:
    from tau2.data_model.tasks import Task
    return [Task.model_validate({k: v for k, v in json.loads(l).items() if k != "source_id"}) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build"])
    ap.add_argument("--src", default="outputs/rt/cs/sft_data/sft_B.jsonl")
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default="x")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", required=True)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
