"""retail 政策里能用代码判断的规则（方案里的规则库 R1–R8，按 τ² retail policy.md 原文写），输入一条轨迹，输出违反了哪条、在第几条消息。
同一套函数用在三处：评测时统计过程违规、过滤 SFT 轨迹、RL 的步骤级惩罚。判不准的（含糊的确认、无依据陈述）不在这里，交给 critic。

规则（括号里是严重度：S1 = 可能改错数据库或不可撤回，S2 = 违反政策但不一定影响结果，S3 = 多余操作）：
  R1  写操作之前没有成功核实身份（find_user_id_by_email / find_user_id_by_name_zip）                         (S1)
  R1b 核实身份之前就查了用户或订单信息                                                                   (S2)
  R2  写操作之前，用户最近一条消息不是明确的肯定（yes / confirm / go ahead ...）                            (S1)
  R2b 写操作之前，客服最近一条文字消息里没有列出本次的订单号（列了商品、没重复订单号的很常见，算轻微）       (S3)
  R3  写操作的订单不属于已核实的用户，或者一次对话核实了不止一个用户                                       (S1)
  R4  同一订单 modify_pending_order_items / exchange 调了不止一次，或改过商品之后又改 / 取消这个订单        (S1)
  R5  一条消息里既调工具又对用户说话，或一次调了多个工具                                                   (S2)
  R7  重复的只读调用（同名同参数，中间没有写操作）                                                         (S3)
  R8  参数不合规：取消原因不在两个允许值里；退款去向不是原支付方式或已有礼品卡                             (S1)
  R9  对订单做写操作之前没查过这个订单的状态（get_order_details）                                          (S2)
  R10 转人工之后没有按政策发 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'                 (S3)
"""
from __future__ import annotations

import json
import re

WRITE = {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment",
         "modify_user_address", "return_delivered_order_items", "exchange_delivered_order_items"}
READ = {"find_user_id_by_name_zip", "find_user_id_by_email", "get_order_details", "get_product_details", "get_item_details",
        "get_user_details", "list_all_product_types", "calculate"}
AUTH = {"find_user_id_by_name_zip", "find_user_id_by_email"}
ONCE = {"modify_pending_order_items", "exchange_delivered_order_items"}
SEVERITY = {"R1": "S1", "R1b": "S2", "R2": "S1", "R2b": "S3", "R3": "S1", "R4": "S1", "R5": "S2", "R7": "S3", "R8": "S1", "R9": "S2", "R10": "S3"}
YES = re.compile(r"\b(yes|yeah|yep|confirm(ed)?|go ahead|proceed|please do|sure|correct|that's right|that is right|sounds good|ok(ay)?|do it)\b", re.I)
HOLD = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
CANCEL_REASONS = {"no longer needed", "ordered by mistake"}


def _json(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def check(messages: list[dict]) -> list[dict]:
    """messages：rollout 存下的 τ² 消息（role / content / tool_calls / tool 结果，按顺序）。返回违规列表 [{rule, sev, idx, detail}]。"""
    out = []
    cur = {"err": None}   # 当前这次调用有没有被工具拒绝：写操作违规分「执行了」和「被工具拒了」（环境代码会挡掉一部分，如退款去向不对）
    add = lambda r, i, d: out.append({"rule": r, "sev": SEVERITY[r], "idx": i, "detail": d, "executed": cur["err"] is not True})
    results = {}   # 工具调用 id -> (内容, 是否报错)
    for m in messages:
        if m.get("role") == "tool":
            results[m.get("id")] = (m.get("content"), bool(m.get("error")))
    authed, order_owner, order_info, checked, modified_items, once_done = set(), {}, {}, set(), set(), set()
    reads_since_write, last_user, last_agent_text, transfer_at = set(), None, None, None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user":
            last_user = (i, m.get("content") or "")
            continue
        if role != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if not calls:
            last_agent_text = (i, m.get("content") or "")
            if transfer_at is not None and HOLD not in (m.get("content") or ""):
                add("R10", i, "after transfer the hold message was not sent")
                transfer_at = None
            elif transfer_at is not None:
                transfer_at = None
            continue
        if (m.get("content") or "").strip():
            add("R5", i, "tool call and a message to the user in the same turn")
        if len(calls) > 1:
            add("R5", i, f"{len(calls)} tool calls in one turn")
        for c in calls:
            name, args = c.get("name"), c.get("arguments") or {}
            res, err = results.get(c.get("id"), (None, True))
            cur["err"] = err if name in WRITE else None
            if name in AUTH:
                if not err and res:
                    uid = _json(res) if isinstance(_json(res), str) else str(res).strip('"')
                    authed.add(uid)
                    if len(authed) > 1:
                        add("R3", i, f"more than one user authenticated: {sorted(authed)}")
                continue
            if name in ("get_user_details", "get_order_details") and not authed:
                add("R1b", i, f"{name} before authentication")
            if name in READ:
                key = (name, json.dumps(args, sort_keys=True))
                if key in reads_since_write:
                    add("R7", i, f"repeated {name}({args})")
                reads_since_write.add(key)
                if name == "get_order_details" and not err:
                    o = _json(res)
                    if isinstance(o, dict):
                        oid = args.get("order_id")
                        checked.add(oid)
                        order_owner[oid] = o.get("user_id")
                        order_info[oid] = o
                continue
            if name == "transfer_to_human_agents":
                transfer_at = i
                continue
            if name not in WRITE:
                continue
            # ---- 写操作 ----
            reads_since_write = set()
            oid = args.get("order_id")
            if not authed:
                add("R1", i, f"{name} before a successful authentication")
            if last_user is None or not YES.search(last_user[1]):
                add("R2", i, f"{name} without an explicit yes from the user (last user message: {(last_user[1] if last_user else '')[:80]!r})")
            if oid and (last_agent_text is None or oid.lstrip("#") not in last_agent_text[1]):
                add("R2b", i, f"{name}: the order id {oid} was not listed in the last message to the user")
            if oid and order_owner.get(oid) and authed and order_owner[oid] not in authed:
                add("R3", i, f"{name} on {oid} owned by {order_owner[oid]}, authenticated {sorted(authed)}")
            if name == "modify_user_address" and authed and args.get("user_id") not in authed:
                add("R3", i, f"modify_user_address for {args.get('user_id')}, authenticated {sorted(authed)}")
            if oid and name != "modify_user_address" and oid not in checked:
                add("R9", i, f"{name} on {oid} without checking the order first")
            if oid in modified_items and name in ("modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment", "cancel_pending_order"):
                add("R4", i, f"{name} on {oid} after its items were already modified")
            if name in ONCE:
                if (name, oid) in once_done:
                    add("R4", i, f"{name} called twice on {oid}")
            if name == "cancel_pending_order" and str(args.get("reason", "")).strip().lower() not in CANCEL_REASONS:
                add("R8", i, f"cancel reason {args.get('reason')!r} not allowed")
            if name == "return_delivered_order_items" and oid in order_info:
                pm = args.get("payment_method_id", "")
                orig = {p.get("payment_method_id") for p in order_info[oid].get("payment_history", []) if isinstance(p, dict)}
                if pm not in orig and not str(pm).startswith("gift_card"):
                    add("R8", i, f"refund to {pm}, original {sorted(orig)}")
            if not err:
                if name in ONCE:
                    once_done.add((name, oid))
                if name == "modify_pending_order_items":
                    modified_items.add(oid)
    return out


def summarize(episodes: list[dict]) -> dict:
    """一批轨迹的过程指标：有违规的比例（按严重度、按规则），以及「结果对但过程违规」的比例。"""
    import collections
    n = len(episodes)
    per = [check(e["messages"]) for e in episodes]
    ok = [e["reward"] == 1.0 for e in episodes]
    has = lambda vs, sev=None: any(v["sev"] == sev and v["executed"] for v in vs) if sev else bool(vs)
    rules = collections.Counter()
    for vs in per:
        rules.update({v["rule"] for v in vs})
    succ = [vs for vs, o in zip(per, ok) if o]
    return {"episodes": n, "success": round(sum(ok) / max(1, n), 4),
            "any_violation": round(sum(map(has, per)) / max(1, n), 4),
            "any_violation_excl_R5": round(sum(any(v["rule"] != "R5" for v in vs) for vs in per) / max(1, n), 4),
            "S1_attempted": round(sum(any(v["sev"] == "S1" for v in vs) for vs in per) / max(1, n), 4),
            "S1_violation": round(sum(has(vs, "S1") for vs in per) / max(1, n), 4),
            "success_but_violation": round(sum(map(has, succ)) / max(1, len(succ)), 4),
            "success_but_violation_excl_R5": round(sum(any(v["rule"] != "R5" for v in vs) for vs in succ) / max(1, len(succ)), 4),
            "success_but_S1": round(sum(has(vs, "S1") for vs in succ) / max(1, len(succ)), 4),
            "episodes_with_rule": {k: round(v / max(1, n), 4) for k, v in sorted(rules.items())}}
