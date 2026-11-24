"""客服 Agent 的 SFT 数据：APIGen-MT-5k 的 retail 对话 → 和 rollout 同一套 token 格式的训练样本。

  python -m cs_agent.sft_data build --src <apigen-mt_5k.json> --out outputs/rt/cs/sft_data [--tok <分词器>]
  python -m cs_agent.sft_data show --dir outputs/rt/cs/sft_data --n 2 --tok <分词器>   # 打印转换后的样本，人工核对格式

处理步骤：
  1. 只取 retail（3,411 条）；ShareGPT 的 human / gpt / function_call / observation 转成 τ² 的消息（user / 客服文字 / 工具调用 / 工具结果），
     开头补上 τ² 固定的开场白（rollout 也是这样开始的，不算 loss）；
  2. 系统提示和工具定义换成 τ² 现在的版本（和评测、GRPO 完全一样），APIGen 原来带的旧版 τ-bench 政策不用；
  3. 去泄漏：对话里出现 τ² 测试任务标准动作里的用户 id、订单号、邮箱、姓名的，整条去掉；
  4. 规则库（cs_agent.rules）逐条检查，记下违规；A 版不按规则筛，B 版去掉有 S1 / S2 违规的（R5 在这份数据里不会出现：每次工具调用都单独一轮）；
  5. 长度：按训练时的 token 序列算，超过 --max_len 的丢掉。
输出 sft_A.jsonl / sft_B.jsonl（每行：id、messages、违规、token 数），stats.json；token 化在训练时用 cs_agent.sft_data.encode 做，和 rollout 共用 qwen_fmt.Conv。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cs_agent  # noqa: F401
from cs_agent.qwen_fmt import Conv
from cs_agent.rollout import GREETING, agent_system_prompt


def convert(conv: list[dict]) -> tuple[list[dict], str | None]:
    """ShareGPT 对话 → τ² 风格的消息（rules.check 和 encode 都吃这个格式）。格式不对返回 (None, 原因)。"""
    msgs = [{"role": "assistant", "content": GREETING, "greeting": True}]
    k = 0
    for t in conv:
        f, v = t["from"], t["value"]
        if f == "human":
            msgs.append({"role": "user", "content": v})
        elif f == "gpt":
            msgs.append({"role": "assistant", "content": v})
        elif f == "function_call":
            try:
                c = json.loads(v)
            except ValueError:
                return None, "bad function_call json"
            k += 1
            args = c.get("arguments") or {}
            if isinstance(args, str):   # 少数样本的参数是又编码了一层的 JSON 字符串
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    return None, "bad arguments json"
            if not isinstance(args, dict):
                return None, "arguments not a dict"
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{k}", "name": c["name"], "arguments": args}]})
        elif f == "observation":
            if not msgs or msgs[-1].get("role") != "assistant" or not msgs[-1].get("tool_calls"):
                return None, "observation without a call"
            msgs.append({"role": "tool", "id": msgs[-1]["tool_calls"][0]["id"], "content": v, "error": str(v).startswith("Error")})
        else:
            return None, f"unknown role {f}"
    while msgs and msgs[-1]["role"] != "assistant":   # 末尾是用户或工具结果：后面没有要学的客服输出，截掉
        msgs.pop()
    if len(msgs) < 3:
        return None, "too short"
    return msgs, None


def call_text(name: str, args: dict) -> str:
    """和聊天模板写工具调用的方式一致：字符串原样写，其余写成 JSON（模板的 tojson）。"""
    body = "".join(f"<parameter={a}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n" for a, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def encode(tok, msgs: list[dict], system: str, tools: list[dict]) -> dict:
    """消息 → 和 rollout 完全相同的 token 序列：{prompt_ids, ids, lmask}；客服的每一轮（文字或工具调用）+ 结尾的 <|im_end|> 算 loss。"""
    c = Conv(tok, system, tools)
    pending = []
    for m in msgs:
        if m["role"] == "tool":
            pending.append(m["content"])
            continue
        if pending:
            c.add_tool_results(pending)
            pending = []
        if m["role"] == "user":
            c.add_user(m["content"] or "")
        elif m.get("greeting"):
            c.add_assistant_text(m["content"])
        else:
            text = call_text(m["tool_calls"][0]["name"], m["tool_calls"][0]["arguments"]) if m.get("tool_calls") else (m["content"] or "").strip()
            ids = c.enc(text) + [c.im_end]
            c.add_policy(ids, [0.0] * len(ids), "stop")
    return {"prompt_ids": c.prefix_ids, "ids": c.ids, "lmask": c.lmask}


def test_keys() -> set[str]:
    """τ² 测试任务标准动作里出现的用户 id、订单号、邮箱、姓名（小写，订单号去掉 #）。"""
    from tau2.domains.retail.environment import get_tasks
    ks = set()
    for t in get_tasks("test"):
        for a in t.evaluation_criteria.actions or []:
            for k in ("user_id", "order_id", "email"):
                if k in a.arguments:
                    ks.add(str(a.arguments[k]).lower().lstrip("#"))
            if a.name == "find_user_id_by_name_zip":
                ks.add(f"{a.arguments['first_name']} {a.arguments['last_name']}".lower())
    return ks


def build(a) -> None:
    from transformers import AutoTokenizer
    from tau2.domains.retail.environment import get_environment
    from cs_agent.rules import check
    tok = AutoTokenizer.from_pretrained(a.tok)
    env = get_environment()
    system, tools = agent_system_prompt(env), [t.openai_schema for t in env.get_tools()]
    keys = test_keys()
    raw = [x for x in json.load(open(a.src, encoding="utf-8")) if "etail" in x["system"][:300]]
    st = collections.Counter(retail=len(raw))
    rule_eps = collections.Counter()
    A, B = [], []
    for i, x in enumerate(raw):
        txt = json.dumps(x["conversations"], ensure_ascii=False).lower()
        if any(k in txt for k in keys):
            st["drop_test_overlap"] += 1
            continue
        msgs, why = convert(x["conversations"])
        if msgs is None:
            st[f"drop_{why}"] += 1
            continue
        n = len(encode(tok, msgs, system, tools)["ids"]) + 0
        if n + 5100 > a.max_len:   # 开头（系统提示 + 工具定义）约 5,066 token
            st["drop_too_long"] += 1
            continue
        vs = check(msgs)
        rule_eps.update({v["rule"] for v in vs})
        bad = [v for v in vs if v["sev"] in ("S1", "S2") and v["rule"] != "R5"]
        row = {"id": f"apigen_retail_{i}", "messages": msgs, "violations": vs, "n_tokens": n + 5066}
        A.append(row)
        if bad:
            st["B_drop_rule"] += 1
        else:
            B.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_A", A), ("sft_B", B)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = sorted(r["n_tokens"] for r in A)
    stats = {**st, "A": len(A), "B": len(B), "A_tokens_total": sum(toks), "A_tokens_p50": toks[len(toks) // 2] if toks else 0,
             "A_tokens_max": toks[-1] if toks else 0, "A_episodes_with_rule": {k: round(v / max(1, len(A)), 4) for k, v in sorted(rule_eps.items())},
             "max_len": a.max_len, "src": a.src}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SFT_DATA", json.dumps(stats, ensure_ascii=False), flush=True)


def show(a) -> None:
    from transformers import AutoTokenizer
    from tau2.domains.retail.environment import get_environment
    tok = AutoTokenizer.from_pretrained(a.tok)
    env = get_environment()
    system, tools = agent_system_prompt(env), [t.openai_schema for t in env.get_tools()]
    rows = [json.loads(l) for l, _ in zip(open(Path(a.dir) / "sft_B.jsonl", encoding="utf-8"), range(a.n))]
    for r in rows:
        e = encode(tok, r["messages"], system, tools)
        print(f"===== {r['id']}  tokens {len(e['prompt_ids']) + len(e['ids'])}  (prefix {len(e['prompt_ids'])}, loss tokens {sum(e['lmask'])})")
        # 按是否算 loss 分段打印：[LOSS] 是模型要学的部分
        seg, cur = [], None
        for t, m in zip(e["ids"], e["lmask"]):
            if m != cur:
                seg.append([m, []])
                cur = m
            seg[-1][1].append(t)
        for m, ts in seg:
            s = tok.decode(ts)
            print(("[LOSS] " if m else "[ctx]  ") + (s if len(s) < 400 else s[:250] + " …… " + s[-120:]).replace("\n", "⏎"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "show"])
    ap.add_argument("--src", default="external_data/agent/apigen-mt-5k/apigen-mt_5k.json")
    ap.add_argument("--out", default="outputs/rt/cs/sft_data")
    ap.add_argument("--dir", default="outputs/rt/cs/sft_data")
    ap.add_argument("--tok", default="outputs/models/Qwen3.5-2B-tok")   # Qwen3.5 的 2B 和 9B 分词器、模板相同
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else show(a)


if __name__ == "__main__":
    main()
