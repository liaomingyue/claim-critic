"""客服轨迹的错误卡片和合成算子：正确样本改错（inject，针对 critic 漏判）、错误样本改对（repair，针对误报），外加「看着像错、其实是对」（confuse）。

卡片体系按 critic 的错误分层（scripts/analysis/cs_critic_errors.py 的报告）：
  不一致 F1   和工具结果 / 前文对不上        I1–I3、R4
  缺失   F2   该做的没做                    I4–I8、R2、R5、R6
  理解   F3   意图 / 约束 / 政策            I9–I12、R3
  写参数 F4   写操作的参数                  I13–I14、R1
  合规   F5                                I15–I17、R8
  纠错         出错后自己纠正               R7
  像错实对     打结构捷径                    C1–C3
每张卡片：改哪里、做法、适用条件、为什么算错（对）、第一错误落在哪、怎么验证、对应的盲区。做法 / 适用条件 / 为什么是英文，后面直接进错误生成模型的提示词。

第一批实现了 8 张（代码能验证、盲区最大）：I1 I2 I4 I7 I13 I14 R1 R7。I6、R2 要续跑，先只有定义：
R2 只看前缀做不对——用户模拟器常偏离 gold（用户还在提问、要的动作和 gold 不同），代码判断不了「此刻执行 gold 写操作是对的」，
本地试跑 5 条全不对；要续跑完整条、重放比对数据库，再让 9B 核对用户确实同意了这个动作。
改错卡（I13 / I14 等）的孪生就是「同一位置的正确写操作」，「第一次写操作 = 错」这条捷径主要靠这些成对样本打，R1 产出少不影响。
都是「前缀样本」：轨迹截到标签那一步为止（critic 在这一步结尾读分，看不到之后的内容），所以不需要重新执行工具、不需要续跑；
需要续跑出 AFTER_ERR / FIX 后续步骤的版本放到有环境和推理服务的机器上做（rollout=True 的卡片）。

每个合成样本都带一个「孪生」：同一条源轨迹截在同一位置、不做改动（改错卡的孪生标签是对，改对卡的孪生标签是错），
两者用同一个编码器重新编码成 token（encode），编码方式和标签无关，critic 学不到「重新编码过的就是错」这种假特征。

  python -m cs_agent.error_cards synth --eps <rollout jsonl>... --tasks <τ² tasks.json> [--labels <steps_v2_clean.jsonl>] \
      [--tok <分词器目录>] --cards I1 I2 I4 I7 I13 I14 R1 R2 R7 --out_dir <目录>
  python -m cs_agent.error_cards cards --out docs/cs_error_cards.md [--examples <synth 目录>]
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from cs_agent.ground import check_replies
from cs_agent.rules import WRITE, YES, check

SOFT = {"R5", "R2b", "R10", "R7"}   # 不算错的规则（R7 重复查询是 S3，也不当错）
TRANSFER = "transfer_to_human_agents"
NEG = re.compile(r"\b(no|not|wait|hold on|don't|can't|cannot|won't|never)\b|n't\b", re.I)


def affirm(text: str) -> bool:
    """用户明确同意：有肯定词，且肯定词之前没有否定（「I can't say yes」不算）。"""
    m = YES.search(text or "")
    return bool(m) and not NEG.search(text[: m.start()])


DONE = re.compile(r"\b(has been|have been|successfully|is now|was (cancelled|modified|processed|updated|returned|exchanged)|processed|completed|done)\b", re.I)


# ================================================================ 卡片
@dataclass
class Card:
    id: str
    direction: str        # inject / repair / confuse
    layer: str
    name: str             # 中文名
    where: str            # 改哪里
    how: str              # 做法（英文，进提示词）
    when: str             # 适用条件（英文）
    why: str              # 为什么算错 / 算对（英文）
    lands: str            # 第一错误（或被改对的那一步）落在哪
    verify: str           # 怎么验证
    blind: str = ""       # 对应 critic 的哪个盲区
    rollout: bool = False  # 需要续跑才能做
    implemented: bool = False
    examples: list = field(default_factory=list)


C = Card
CARDS = {c.id: c for c in [
    # ---------------- 正确样本改错（inject）
    C("I1", "inject", "F1", "复述工具结果出错", "客服文字",
      "In a message to the customer, restate one value from an earlier tool result (a price, refund amount or total) with a wrong value.",
      "a text message from the agent quotes a dollar amount that appears in an earlier tool result",
      "the stated value contradicts what the tool returned", "被改的那条客服文字", "代码：ground.check_replies 在这一步报出新值、原版不报",
      "不一致类召回 0.13", implemented=True),
    C("I2", "inject", "F1", "工具结果变了，客服没跟上", "工具结果",
      "Change one value in a tool result (an item price, or an order status) and keep everything the agent said and did afterwards unchanged.",
      "a later agent message repeats that price, or the agent later acts on that order assuming the old status",
      "the agent's statement or action no longer matches what the tool returned", "第一条复述旧值（或按旧状态行事）的客服步骤",
      "代码：check_replies / 订单状态规则", "不一致类召回 0.13", implemented=True),
    C("I3", "inject", "F1", "张冠李戴", "客服文字",
      "Make the agent refer to the wrong object: say it found an item in order A when it looked up order B, or name a different product than the one it queried.",
      "the agent mentions an order id or product right after looking it up",
      "the object named in the message is not the one the tool call returned", "被改的那条客服文字", "代码：比对 id"),
    C("I4", "inject", "F2", "漏做写操作却说办好了", "删除一步",
      "Delete one database-changing call and its result; keep the agent's following message that tells the customer it was done.",
      "a successful write call is followed by an agent message reporting the change as done",
      "the agent claims a change that was never made", "那条「已办好」的客服文字", "代码：gold 写操作缺失 + 文字里有完成措辞",
      "缺失类召回 0.25", implemented=True),
    C("I5", "inject", "F2", "确认后迟迟不执行", "截断后插入一步",
      "After the customer says yes, make the agent ask again or look things up instead of executing, and end the conversation.",
      "the customer gives an explicit yes and the agent's next step is the write", "the confirmed change is never made",
      "用户说 yes 之后的那一步", "重放：数据库 ≠ gold", "缺失类召回 0.25", rollout=True),
    C("I6", "inject", "F2", "查找不全就下结论", "删除一步",
      "Remove the lookup that found the right order, so the agent settles on the first order it checked.",
      "the agent checked several orders before finding the one the customer meant", "the agent acts on the wrong order",
      "下结论的那一步", "重放 + gold 对照", "缺失类召回 0.25", rollout=True),
    C("I7", "inject", "F2", "能办却转人工", "插入一步",
      "Replace the agent's next step after the customer's confirmation with a transfer to a human agent.",
      "the customer has confirmed and the agent's next step is a write that matches the gold action",
      "the request could have been completed, the policy only allows a transfer when the agent cannot handle it", "插入的转人工那一步",
      "代码：被替换的写操作和 gold 一致（本来能办成）", "转人工召回 0.14", implemented=True),
    C("I8", "inject", "F2", "方案里漏掉一个请求", "客服文字",
      "When the agent lists the changes for confirmation, drop one of the customer's requests from the list.",
      "the confirmation message lists two or more changes", "one requested change will never be made", "确认那条客服文字", "gold 对照"),
    C("I9", "inject", "F3", "范围扩大", "用户的话 / 写操作",
      "Narrow the customer's request (only one item, only one order) while the agent still changes everything it originally changed.",
      "the customer's request covers several items or orders", "the agent changes more than the customer asked for", "确认文字或写操作",
      "gold 对照 + 9B"),
    C("I10", "inject", "F3", "无视用户的条件", "用户的话",
      "Add a condition to the customer's message (e.g. 'if this needs cancelling the whole order, don't do it') that the agent's later action violates.",
      "the agent later makes a change that such a condition would forbid", "the agent ignores an explicit condition from the customer",
      "违反条件的那一步", "9B 对照改动"),
    C("I11", "inject", "F3", "违反约束选错候选", "用户的话 / new_item_id",
      "Add a constraint to the customer's request (shorter, same capacity, cheapest) that the agent's chosen item does not meet, or swap the chosen item for one that breaks the stated constraint.",
      "the agent chooses a new item from a list of variants", "the chosen item does not satisfy what the customer asked for",
      "提出方案的文字", "代码：商品属性比对"),
    C("I12", "inject", "F3", "政策说错", "客服文字",
      "Make the agent state a policy rule wrongly (e.g. that a pending order cannot have its items changed), which sends the conversation the wrong way.",
      "the agent explains what can or cannot be done", "the statement contradicts the policy", "那一步", "9B 对照政策原文"),
    C("I13", "inject", "F4", "退款去向错", "写操作参数（及确认文字）",
      "Change the refund payment method of a return to another of the customer's payment methods that the order was not paid with (and not a gift card), also where the agent states it in the confirmation.",
      "a return call whose refund goes to the original payment method, and the customer has another non-gift-card payment method",
      "refunds must go to the original payment method or an existing gift card", "第一次说出新去向的客服步骤（确认文字或写操作）",
      "代码：规则 R8", "写参数召回 0.59（确认文字上更低）", implemented=True),
    C("I14", "inject", "F4", "写错对象", "写操作参数（及确认文字）",
      "Replace the order in a cancel / address change with another order of the same customer, or the new item of an exchange / item change with a sibling variant, also where the agent states it in the confirmation.",
      "the customer has another order, or the product has another available variant",
      "the change is applied to something the customer did not ask for", "第一次说出新对象的客服步骤", "代码：和 gold 写操作对不上",
      "写参数召回 0.59", implemented=True),
    C("I15", "inject", "F5", "泄露隐私", "客服文字",
      "Insert the customer's e-mail address or gift card balance into a reply where it was not asked for.",
      "the agent has looked up the user's details", "the policy forbids revealing private information that was not requested", "那一步", "代码"),
    C("I16", "inject", "F5", "没有确认就执行", "删除一步",
      "Delete the customer's explicit yes (or the agent's confirmation turn) before a write.", "a write preceded by a confirmation exchange",
      "the policy requires an explicit yes before any change", "那次写操作", "代码：规则 R2"),
    C("I17", "inject", "F5", "违反状态依赖", "工具结果 / 步骤顺序",
      "Make the agent change an order again after its items were modified, or act on an order whose status does not allow the change.",
      "an order is modified more than once, or a write follows a status lookup", "the policy forbids the change in that state", "那次写操作",
      "代码：规则 R4 + 状态"),
    # ---------------- 错误样本改对（repair）
    C("R1", "repair", "F4", "改对写操作参数", "写操作参数（及确认文字）",
      "Replace the wrong ids in a failed write (order, payment method, items) with the gold ones, also where the agent stated them in the confirmation.",
      "the trajectory's first error is a write whose arguments differ from a gold write of the same tool only in ids",
      "the write now matches the gold action and everything before it was already correct", "被改对的写操作（标 OK）",
      "代码：参数 = gold、改后前缀没有规则违规", "误报：第一次写操作 91% 被判 ERR", implemented=True),
    C("R2", "repair", "F2", "补上漏做的写操作", "替换一步",
      "Where the customer has confirmed a change that the agent then never made, put the gold write call as the agent's next step.",
      "a gold write never executed, the customer said yes right after the agent proposed it, and the agent's next step was something else",
      "the confirmed change is made at the right moment", "补上的写操作（标 OK）", "续跑后重放：数据库 = gold；9B 核对用户此刻确实同意了这个动作",
      "误报：写过之后 92% 被判 AFTER_ERR；缺失类", rollout=True),
    C("R3", "repair", "F3", "删掉多做的写操作", "删除一步", "Remove the write that went beyond what the customer asked for.",
      "a write that matches no gold action", "only the requested changes remain", "删除后的下一步", "重放：数据库 = gold", "", rollout=True),
    C("R4", "repair", "F1", "改对陈述", "客服文字", "Rewrite a wrong statement so it matches the tool results and the policy.",
      "a message flagged as a wrong statement", "the message is now supported by the tool results", "那一步（标 OK）", "代码 / 9B", ""),
    C("R5", "repair", "F2", "补查找或换方式核实", "插入一步",
      "Add the missing lookup (check the other order, ask for another e-mail) before the agent decides.", "an incomplete search or failed authentication",
      "the agent now has the information it needs", "插入的那一步", "重放", "", rollout=True),
    C("R6", "repair", "F2", "撤回转人工继续办", "替换一步", "Replace an unnecessary transfer with the agent completing the request.",
      "a transfer when the gold actions show the request was doable", "the request is handled", "替换的那一步", "重放", "", rollout=True),
    C("R7", "repair", "纠错", "出错后自己纠正", "插入一步",
      "Keep the error and add the agent's next step correcting it: restate the right value, or stop before executing and propose the correct refund method.",
      "an error the agent can still repair (a wrong statement, a wrong proposal not yet executed)",
      "the agent notices and repairs its own mistake", "错误那步标 ERR，纠正那步标 FIX", "代码：纠正后的值 = 工具结果 / 原支付方式",
      "FIX 类样本稀少（留出集 16 + 7 条）", implemented=True),
    C("R8", "repair", "F5", "补上确认", "插入一步", "Insert the confirmation exchange (details listed, customer says yes) before a write that lacked it.",
      "a write without an explicit yes", "the policy's confirmation step is now present", "那次写操作（标 OK）", "代码：规则 R2", ""),
    # ---------------- 看着像错、其实是对（confuse）
    C("C1", "confuse", "结构", "互不相关的写操作换顺序", "步骤顺序", "Swap two independent writes; the final database is the same.",
      "two writes on different orders", "order of independent changes does not matter", "两次写操作都标 OK", "重放：数据库 = gold",
      "误报：结构捷径", rollout=True),
    C("C2", "confuse", "结构", "多问一句澄清", "插入一步", "Insert one reasonable clarifying question before a write.",
      "a write after a confirmation", "a clarifying question is allowed", "写操作标 OK", "代码", "误报：结构捷径"),
    C("C3", "confuse", "结构", "工具报错后正确重试", "插入两步",
      "Insert a failed lookup with a typo and the agent's corrected retry.", "a successful lookup", "recovering from a tool error is correct behaviour",
      "重试标 OK", "代码", "误报：结构捷径"),
]}


# ================================================================ 轨迹工具
def agent_steps(msgs: list[dict]) -> list[int]:
    return [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and not (i == 0 and not m.get("tool_calls"))]


def _json(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def norm_write(name: str, args: dict) -> tuple:
    a = dict(args)
    if isinstance(a.get("item_ids"), list) and isinstance(a.get("new_item_ids"), list):
        a["_pairs"] = sorted(zip(map(str, a.pop("item_ids")), map(str, a.pop("new_item_ids"))))
    elif isinstance(a.get("item_ids"), list):
        a["item_ids"] = sorted(map(str, a["item_ids"]))
    return name, json.dumps({k: a[k] for k in sorted(a)}, sort_keys=True, ensure_ascii=False)


class Traj:
    """一条源轨迹 + 它的任务 gold 写操作 + 常用的查表。"""

    def __init__(self, ep: dict, gold: list[dict], label: dict | None = None):
        self.ep, self.msgs = ep, ep["messages"]
        self.turn_text = {t["msg_idx"]: t["text"] for t in ep.get("turns") or [] if "msg_idx" in t}
        self.gold_w = [a for a in gold if a["name"] in WRITE]
        self.steps = agent_steps(self.msgs)
        self.label = label   # steps_v2 的一行（有的话）
        self.ok = ep.get("reward") == 1.0
        self.res = {m.get("id"): m for m in self.msgs if m.get("role") == "tool"}
        self.user, self.orders, self.products = {}, {}, {}
        for m in self.msgs:
            for c in m.get("tool_calls") or []:
                r = self.res.get(c.get("id"))
                if not r or r.get("error"):
                    continue
                o = _json(r.get("content"))
                if not isinstance(o, dict):
                    continue
                if c["name"] == "get_user_details":
                    self.user = o
                elif c["name"] == "get_order_details":
                    self.orders[o.get("order_id")] = o
                elif c["name"] == "get_product_details":
                    self.products[o.get("product_id")] = o

    def call(self, i):
        tc = self.msgs[i].get("tool_calls") or []
        return tc[0] if len(tc) == 1 else None

    def gold_match(self, name, args) -> bool:
        return norm_write(name, args) in {norm_write(g["name"], g["arguments"]) for g in self.gold_w}

    def step_of(self, i) -> int:
        return self.steps.index(i) + 1

    def code_errors(self, msgs=None, upto=None) -> list[int]:
        """代码能看到的错误位置（规则 S1/S2 非软违规、无依据的数字 / id、和 gold 对不上的写操作），只看 upto（含）之前的消息。"""
        msgs = msgs if msgs is not None else self.msgs
        sub = msgs[: (upto + 1) if upto is not None else None]
        out = {v["idx"] for v in check(sub) if v["rule"] not in SOFT and v["sev"] in ("S1", "S2")}
        out |= {x["idx"] for x in check_replies(sub)}
        for i, m in enumerate(sub):
            for c in m.get("tool_calls") or []:
                if c["name"] in WRITE and not self.gold_match(c["name"], c.get("arguments") or {}):
                    out.add(i)
        return sorted(out)

    def clean_prefix(self, upto: int) -> bool:
        """upto 之前（不含）都是对的：代码查不出错，且有标签时标签都是 OK。"""
        if self.code_errors(upto=upto - 1):
            return False
        if self.label:
            k = self.step_of(upto) if upto in self.steps else len([s for s in self.steps if s < upto]) + 1
            if any(s["cls"] != "OK" for s in self.label["steps"] if s["step"] < k):
                return False
        return True

    def last_agent_text(self, before: int) -> int | None:
        for j in range(before - 1, -1, -1):
            m = self.msgs[j]
            if m.get("role") == "assistant" and not m.get("tool_calls") and j > 0:
                return j
        return None


def render(m: dict, text: str | None = None) -> str:
    """一条客服消息 → 策略输出的文本格式（文字 + 工具调用）。"""
    from cs_agent.sft_data import call_text
    if text is not None:
        return text
    body = (m.get("content") or "").strip()
    for c in m.get("tool_calls") or []:
        body = (body + "\n\n" if body else "") + call_text(c["name"], c.get("arguments") or {})
    return body


def replace_in(msgs: list[dict], a: int, b: int, old: str, new: str, roles=("assistant", "user")) -> list[int]:
    """把 [a, b] 区间内客服文字 / 用户的话（以及客服调用参数）里的 old 换成 new，返回改过的消息位置。"""
    changed = []
    for j in range(a, b + 1):
        m = msgs[j]
        if m.get("role") not in roles:
            continue
        hit = False
        if old in (m.get("content") or ""):
            m["content"] = m["content"].replace(old, new)
            hit = True
        for c in m.get("tool_calls") or []:
            s = json.dumps(c.get("arguments") or {}, ensure_ascii=False)
            if old in s:
                c["arguments"] = json.loads(s.replace(old, new))
                hit = True
        if hit:
            m["_edited"] = True
            changed.append(j)
    return changed


# ================================================================ 合成结果
@dataclass
class Sample:
    card: str
    msgs: list            # 截断后的消息（带 _src / _edited / _text 标记）
    labels: dict          # 消息位置 → 类别（只列非 OK 的；截断点之前其余步骤都是 OK）
    target: int           # 标签那一步（截断点）
    info: dict
    twin_cut: int | None = None      # 孪生（源轨迹不改）截在哪条消息
    twin_labels: dict | None = None  # 孪生的非 OK 标签


def _base(t: Traj) -> list[dict]:
    ms = copy.deepcopy(t.msgs)
    for i, m in enumerate(ms):
        m["_src"] = i
    return ms


# ---------------------------------------------------------------- I1 复述工具结果出错
MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})")


def _shift_money(v: float, rng) -> float:
    for _ in range(20):
        k = rng.choice(["digit", "pct", "cents"])
        if k == "pct":
            w = round(v * (1 + rng.choice([-1, 1]) * rng.uniform(0.08, 0.3)), 2)
        elif k == "cents":
            w = round(v + rng.choice([-1, 1]) * rng.choice([0.1, 1.0, 10.0]), 2)
        else:
            s = f"{v:.2f}"
            ds = [i for i, ch in enumerate(s) if ch.isdigit()]
            i = rng.choice(ds)
            w = float(s[:i] + str((int(s[i]) + rng.randint(1, 8)) % 10) + s[i + 1:])
        if w > 0 and abs(w - v) > 0.004:
            return w
    return round(v * 1.1, 2)


def op_I1(t: Traj, rng) -> list[Sample]:
    out = []
    for i in t.steps:
        m = t.msgs[i]
        if m.get("tool_calls") or not t.clean_prefix(i):
            continue
        vals = MONEY.findall(m.get("content") or "")
        if not vals:
            continue
        raw = rng.choice(vals)
        v = float(raw.replace(",", ""))
        w = _shift_money(v, rng)
        ms = _base(t)[: i + 1]
        new = f"{w:,.2f}" if "," in raw else f"{w:.2f}"
        ms[i]["content"] = ms[i]["content"].replace("$" + raw, "$" + new, 1).replace("$ " + raw, "$ " + new, 1)
        ms[i]["_edited"] = True
        flags = {x["idx"] for x in check_replies(ms)}
        if i not in flags or flags - {i}:
            continue
        out.append(Sample("I1", ms, {i: "ERR"}, i, {"old": raw, "new": new}, twin_cut=i, twin_labels={}))
        break
    return out


# ---------------------------------------------------------------- I2 工具结果变了，客服没跟上
def op_I2(t: Traj, rng) -> list[Sample]:
    out = []
    # (a) 价格：改工具结果里的一个价格，客服之后复述它的那条消息成为第一错误
    for i in t.steps:
        m = t.msgs[i]
        if m.get("tool_calls"):
            continue
        for raw in MONEY.findall(m.get("content") or ""):
            v = raw.replace(",", "")
            src = [j for j in range(i) if t.msgs[j].get("role") == "tool" and re.search(rf"\b{re.escape(v)}\b", t.msgs[j].get("content") or "")]
            if len(src) != 1 or not t.clean_prefix(i):
                continue
            j = src[0]
            w = f"{_shift_money(float(v), rng):.2f}"
            ms = _base(t)[: i + 1]
            ms[j]["content"] = re.sub(rf"\b{re.escape(v)}\b", w, ms[j]["content"])
            ms[j]["_edited"] = True
            flags = sorted({x["idx"] for x in check_replies(ms)})
            if not flags or flags[0] < j:
                continue
            k = flags[0]
            if k not in t.steps:
                continue
            ms = ms[: k + 1]
            out.append(Sample("I2", ms, {k: "ERR"}, k, {"kind": "price", "tool_msg": j, "old": v, "new": w}, twin_cut=k, twin_labels={}))
            return out
    # (b) 状态：get_order_details 返回 pending，之后对这个订单做了 modify_pending_* / cancel → 改成 processed
    for i in t.steps:
        c = t.call(i)
        if not c or c["name"] != "get_order_details":
            continue
        r = t.res.get(c.get("id"))
        o = _json((r or {}).get("content"))
        if not isinstance(o, dict) or o.get("status") != "pending":
            continue
        oid = o.get("order_id")
        later = [k for k in t.steps if k > i and (t.call(k) or {}).get("name", "").startswith(("modify_pending", "cancel_pending"))
                 and (t.call(k).get("arguments") or {}).get("order_id") == oid]
        if not later:
            continue
        ri = t.msgs.index(r)
        # 第一个按旧状态说话或行事的客服步骤：说了 pending 的文字，或者那次写操作
        k = next((s for s in t.steps if ri < s <= later[0] and "pending" in (t.msgs[s].get("content") or "").lower()), later[0])
        if not t.clean_prefix(k):
            continue
        ms = _base(t)[: k + 1]
        ms[ri]["content"] = ms[ri]["content"].replace('"status": "pending"', '"status": "processed"')
        ms[ri]["_edited"] = True
        out.append(Sample("I2", ms, {k: "ERR"}, k, {"kind": "status", "tool_msg": ri, "order": oid}, twin_cut=k, twin_labels={}))
        return out
    return out


# ---------------------------------------------------------------- I4 漏做写操作却说办好了
def op_I4(t: Traj, rng) -> list[Sample]:
    if not t.ok:
        return []
    for i in t.steps:
        c = t.call(i)
        if not c or c["name"] not in WRITE:   # 策略多数一边说话一边调用（R5），这条消息连文字一起删
            continue
        r = t.res.get(c.get("id"))
        if not r or r.get("error") or not t.gold_match(c["name"], c.get("arguments") or {}):
            continue
        ri = t.msgs.index(r)
        nxt = ri + 1
        if nxt >= len(t.msgs) or t.msgs[nxt].get("role") != "assistant" or t.msgs[nxt].get("tool_calls"):
            continue
        if not DONE.search(t.msgs[nxt].get("content") or "") or not t.clean_prefix(nxt):
            continue
        ms = _base(t)[: nxt + 1]
        del ms[i: ri + 1]
        k = len(ms) - 1
        return [Sample("I4", ms, {k: "ERR"}, k, {"deleted": c["name"], "args": c.get("arguments")}, twin_cut=nxt, twin_labels={})]
    return []


# ---------------------------------------------------------------- I7 能办却转人工
TRANSFER_TEXT = ["I'm not able to complete this change myself, so I'll transfer you to a human agent who can help.",
                 "This request needs to be handled by a specialist. Let me transfer you to a human agent.",
                 "I'll transfer you to a human agent to complete this for you."]


def op_I7(t: Traj, rng) -> list[Sample]:
    for i in t.steps:
        c = t.call(i)
        if not c or c["name"] not in WRITE or not t.gold_match(c["name"], c.get("arguments") or {}):
            continue
        u = i - 1
        if t.msgs[u].get("role") != "user" or not affirm(t.msgs[u].get("content") or "") or not t.clean_prefix(i):
            continue
        ms = _base(t)[: i + 1]
        uid = (t.user or {}).get("user_id", "the customer")
        oid = (c.get("arguments") or {}).get("order_id", "")
        verb = c["name"].split("_")[0]
        summary = f"Customer {uid} wants to {verb} {('order ' + oid) if oid else 'their account details'}. Transferring for further assistance."
        ms[i] = {"role": "assistant", "content": rng.choice(TRANSFER_TEXT), "_edited": True, "_src": None,
                 "tool_calls": [{"id": "syn_tr", "name": TRANSFER, "arguments": {"summary": summary}, "requestor": "assistant"}]}
        return [Sample("I7", ms, {i: "ERR"}, i, {"replaced": c["name"]}, twin_cut=i, twin_labels={})]
    return []


# ---------------------------------------------------------------- I13 退款去向错 / I14 写错对象
def _swap_write(t: Traj, i: int, key: str, old: str, new: str, card: str, info: dict) -> Sample | None:
    """把写操作 i 的参数 old → new，连同之前最后一条客服文字（确认）和其后的用户回答里的同一个值；第一错误 = 最早改动的客服步骤。"""
    c = t.call(i)
    a = t.last_agent_text(i)
    start = a if a is not None and old in (t.msgs[a].get("content") or "") else i
    if not t.clean_prefix(start):
        return None
    ms = _base(t)[: i + 1]
    changed = replace_in(ms, start, i, old, new)
    if i not in changed:
        return None
    if (ms[i]["tool_calls"][0].get("arguments") or {}).get(key) is None and key not in json.dumps(ms[i]["tool_calls"][0].get("arguments")):
        return None
    k = min(j for j in changed if j in t.steps)
    if t.gold_match(c["name"], ms[i]["tool_calls"][0]["arguments"]):
        return None
    if card == "I13" and not any(v["rule"] == "R8" and v["idx"] == i for v in check(ms)):
        return None   # 改完的退款去向要真的违反 R8
    return Sample(card, ms[: k + 1] if k < i else ms, {k: "ERR"}, k, {**info, "new": new, "first": "confirm" if k < i else "write", "write_idx": i},
                  twin_cut=k, twin_labels={})


def op_I13(t: Traj, rng) -> list[Sample]:
    pms = list(((t.user or {}).get("payment_methods") or {}).keys())
    for i in t.steps:
        c = t.call(i)
        if not c or c["name"] != "return_delivered_order_items" or not t.gold_match(c["name"], c.get("arguments") or {}):
            continue
        args = c.get("arguments") or {}
        o = t.orders.get(args.get("order_id")) or {}
        orig = {p.get("payment_method_id") for p in o.get("payment_history") or [] if isinstance(p, dict)}
        cur = args.get("payment_method_id")
        alt = [p for p in pms if p not in orig and p != cur and not p.startswith("gift_card")]
        if not alt or not cur:
            continue
        s = _swap_write(t, i, "payment_method_id", cur, rng.choice(alt), "I13", {"old": cur})
        if s is not None:
            return [s]
    return []


def op_I14(t: Traj, rng) -> list[Sample]:
    gold_orders = {(g["arguments"] or {}).get("order_id") for g in t.gold_w}
    others = [o for o in (t.user or {}).get("orders") or [] if o not in gold_orders]
    for i in t.steps:
        c = t.call(i)
        if not c or c["name"] not in WRITE or not t.gold_match(c["name"], c.get("arguments") or {}):
            continue
        args = c.get("arguments") or {}
        if c["name"] in ("cancel_pending_order", "modify_pending_order_address") and others:
            s = _swap_write(t, i, "order_id", args["order_id"], rng.choice(others), "I14", {"kind": "order", "old": args["order_id"]})
            if s:
                return [s]
        if c["name"] in ("exchange_delivered_order_items", "modify_pending_order_items") and args.get("new_item_ids"):
            new = str(rng.choice(args["new_item_ids"]))
            pid = next((p for p, d in t.products.items() if new in (d.get("variants") or {})), None)
            if not pid:
                continue
            sib = [v for v, d in t.products[pid]["variants"].items() if v != new and d.get("available") and v not in map(str, args.get("item_ids") or [])]
            if not sib:
                continue
            s = _swap_write(t, i, "new_item_ids", new, rng.choice(sib), "I14", {"kind": "variant", "old": new})
            if s:
                return [s]
    return []


# ---------------------------------------------------------------- R1 改对写操作参数
ID_KEYS = ("order_id", "payment_method_id", "item_ids", "new_item_ids")
CONFIRM = re.compile(r"confirm|proceed|would you like|shall I|do you want|is (this|that) correct|\(yes\)|yes/no", re.I)
STRONG_YES = re.compile(r"^\W*(yes|yeah|yep|sure|correct|confirm(ed)?|i confirm|please (do|proceed|go ahead)|go ahead|proceed)\b", re.I)


def _pm_kind(v: str) -> str | None:
    return {"gift_card": "gift card", "paypal": "paypal", "credit_card": "card"}.get(v.rsplit("_", 1)[0]) if "_" in v else None


def op_R1(t: Traj, rng, allow_items: bool = False) -> list[Sample]:
    if t.ok:
        return []
    errs = t.code_errors()
    if not errs:
        return []
    i = errs[0]
    c = t.call(i)
    if not c or c["name"] not in WRITE or i not in t.steps:
        return []
    if t.label and (t.label["meta"].get("first_error_step") not in (None, t.step_of(i))):
        return []   # 判断模型认为第一错误在别处（改对这一步不能让前缀变对）
    args = c.get("arguments") or {}
    cands = [g for g in t.gold_w if g["name"] == c["name"] and not any(
        norm_write(g["name"], g["arguments"]) == norm_write(x["name"], x.get("arguments") or {}) for j in t.steps if j < i for x in t.msgs[j].get("tool_calls") or [])]
    for g in cands:
        ga = g["arguments"]
        if set(ga) != set(args):
            continue
        diffs = [k for k in ga if ga[k] != args[k]]
        if not diffs or any(k not in ID_KEYS for k in diffs) or (not allow_items and any(k in ("item_ids", "new_item_ids") for k in diffs)):
            continue
        a = t.last_agent_text(i)
        ms = _base(t)[: i + 1]
        pairs = []
        for k in diffs:
            if isinstance(args[k], list):
                pairs += [(str(x), str(y)) for x, y in zip(args[k], ga[k]) if x != y]
            else:
                pairs.append((str(args[k]), str(ga[k])))
        users = " ".join((m.get("content") or "") for m in t.msgs[:i] if m.get("role") == "user").lower()
        if any(o.lower() in users or (_pm_kind(o) and _pm_kind(o) in users) for o, _ in pairs):
            continue   # 用户自己点了旧值（或旧值那一类支付方式）：不是客服单方面写错，用户的话不改
        # 只改确认那一句（之前最后一条客服文字，且它看起来是在请求确认）和写操作本身，用户的话不动
        start = a if a is not None and CONFIRM.search(t.msgs[a].get("content") or "") and any(o in (t.msgs[a].get("content") or "") for o, _ in pairs) else i
        changed = set()
        for o, n in pairs:
            changed |= set(replace_in(ms, start, i, o, n, roles=("assistant",)))
        ms[i]["tool_calls"][0]["arguments"] = dict(ga)
        ms[i]["_edited"] = True
        said = " ".join((m.get("content") or "") + json.dumps(m.get("tool_calls") or [], ensure_ascii=False) for m in ms if m.get("role") != "tool")
        if any(o in said for o, _ in pairs):
            continue   # 改完前缀里还提到旧值（更早就提出了）：根因在更早的地方，改这一步不能让前缀自洽
        if not t.gold_match(c["name"], ms[i]["tool_calls"][0]["arguments"]):
            continue
        bad = [x for x in t.code_errors(ms, upto=i) if x <= i]
        if bad:
            continue
        twin_k = min([j for j in changed if j in t.steps] + [i])
        return [Sample("R1", ms, {}, i, {"diff": diffs, "pairs": pairs}, twin_cut=i, twin_labels={twin_k: "ERR"})]
    return []


# ---------------------------------------------------------------- R2 补上漏做的写操作
PROCEED = ["Thank you for confirming. I'll process this now.", "Great, I'll proceed with this change now.", "Thank you. Let me process that for you."]
def _flat(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _flat(x)
    elif isinstance(v, list):
        for x in v:
            yield from _flat(x)
    else:
        yield v


def op_R2(t: Traj, rng) -> list[Sample]:
    if t.ok:
        return []
    done = {norm_write(x["name"], x.get("arguments") or {}) for m in t.msgs for x in m.get("tool_calls") or [] if x["name"] in WRITE}
    for g in t.gold_w:
        if norm_write(g["name"], g["arguments"]) in done:
            continue
        oid = (g["arguments"] or {}).get("order_id") or (g["arguments"] or {}).get("user_id")
        for u, m in enumerate(t.msgs):
            if m.get("role") != "user" or not STRONG_YES.search(m.get("content") or "") or not affirm(m.get("content") or ""):
                continue   # 用户开头就明确同意（「Okay... I guess I'll wait」不算）
            a = t.last_agent_text(u)
            j = u + 1
            if a is None or not oid or j not in t.steps:
                continue
            ids = [str(v) for v in _flat(g["arguments"]) if re.fullmatch(r"#?W\d{7}|\d{10}|(credit_card|gift_card|paypal)_\d+", str(v))]
            if not ids or not all(x.lstrip("#") in (t.msgs[a].get("content") or "") for x in ids):
                continue   # 客服上一句要把 gold 写操作的所有 id 都提出来过，用户的同意才是对这个动作的同意
            nx = t.call(j)
            if nx and nx["name"] in WRITE:
                continue   # 下一步本来就是写操作（参数错的归 R1）
            if not t.clean_prefix(j):
                continue
            if t.label and t.label["meta"].get("first_error_step") not in (None, t.step_of(j)):
                continue
            ms = _base(t)[: j + 1]
            ms[j] = {"role": "assistant", "content": rng.choice(PROCEED), "_edited": True, "_src": None,
                     "tool_calls": [{"id": "syn_w", "name": g["name"], "arguments": dict(g["arguments"]), "requestor": "assistant"}]}
            bad = [v for v in check(ms) if v["idx"] == j and v["rule"] not in SOFT and v["sev"] in ("S1", "S2")]
            if bad:
                continue
            twin = {j: "ERR"} if t.label and t.label["meta"].get("first_error_step") == t.step_of(j) else None
            return [Sample("R2", ms, {}, j, {"gold": g["name"]}, twin_cut=j if twin else None, twin_labels=twin)]
    return []


# ---------------------------------------------------------------- R7 出错后自己纠正（在 I1 / I13 的错误上接一步纠正）
FIX_MONEY = ["Sorry, I need to correct what I said: the amount is ${right}, not ${wrong}.",
             "Correction to my previous message: it should be ${right}, not ${wrong}. I apologize for the confusion.",
             "I made a mistake above — the correct amount is ${right} (not ${wrong})."]
FIX_REFUND = ["Before I process this, I need to correct one detail: refunds for returns can only go to the original payment method ({orig}) "
              "or an existing gift card, not {wrong}. Would you like the refund to go to {orig}?",
              "Sorry, I misspoke about the refund. It has to go back to the original payment method, {orig}, or to a gift card. "
              "Shall I process the return with the refund to {orig}?"]


def op_R7(t: Traj, rng) -> list[Sample]:
    out = []
    for s in op_I1(t, rng):   # 文字说错了数，用户回了一句，客服下一步先纠正再接着原来的话
        k = s.target
        u = k + 1
        if u + 1 >= len(t.msgs) or t.msgs[u].get("role") != "user" or (u + 1) not in t.steps:
            continue
        ms = s.msgs + [dict(copy.deepcopy(t.msgs[u]), _src=u)]
        nxt = dict(copy.deepcopy(t.msgs[u + 1]), _src=None, _edited=True)
        fix = rng.choice(FIX_MONEY).format(right=s.info["old"], wrong=s.info["new"])
        nxt["content"] = fix + ("\n\n" + nxt["content"] if (nxt.get("content") or "").strip() else "")
        ms.append(nxt)
        out.append(Sample("R7", ms, {k: "ERR", len(ms) - 1: "FIX"}, len(ms) - 1, {"base": "I1", **s.info}, twin_cut=None, twin_labels=None))
        break
    for s in op_I13(t, rng):   # 确认时说错了退款去向、用户说 yes：执行之前先纠正
        if s.info.get("first") != "confirm":
            continue
        k, i = s.target, s.info["write_idx"]
        ms = _base(t)[: i + 1]
        replace_in(ms, k, i - 1, s.info["old"], s.info["new"])
        ms[i] = {"role": "assistant", "content": rng.choice(FIX_REFUND).format(orig=s.info["old"], wrong=s.info["new"]), "_edited": True, "_src": None}
        out.append(Sample("R7", ms, {k: "ERR", i: "FIX"}, i, {"base": "I13", **s.info}, twin_cut=None, twin_labels=None))
        break
    return out


OPS = {"I1": op_I1, "I2": op_I2, "I4": op_I4, "I7": op_I7, "I13": op_I13, "I14": op_I14, "R1": op_R1, "R2": op_R2, "R7": op_R7}


# ================================================================ 编码
def encode(tok, src_ep: dict, msgs: list[dict]) -> dict:
    """消息 → token（和 rollout 同一种拼法）：系统提示沿用源轨迹的 prompt_ids；客服消息用原始生成文本（没改过的）或按格式重写（改过 / 新加的）。
    原版和改版用同一个函数编码。"""
    from cs_agent.qwen_fmt import Conv
    c = Conv.__new__(Conv)
    c.tok, c.im_end = tok, tok.convert_tokens_to_ids("<|im_end|>")
    c.prefix_ids, c.ids, c.lmask, c.old_lp, c.turns = list(src_ep["prompt_ids"]), [], [], [], []
    turn_text = {t["msg_idx"]: t["text"] for t in src_ep.get("turns") or [] if "msg_idx" in t}
    pending = []
    for n, m in enumerate(msgs):
        if m["role"] == "tool":
            pending.append(m.get("content") or "")
            continue
        if pending:
            c.add_tool_results(pending)
            pending = []
        if m["role"] == "user":
            c.add_user(m.get("content") or "")
        elif n == 0 and not m.get("tool_calls"):
            c.add_assistant_text(m.get("content") or "")
        else:
            src = m.get("_src")
            text = turn_text.get(src) if (src is not None and not m.get("_edited")) else None
            text = (text if text is not None else render(m)).split("<|im_end|>")[0]
            ids = c.enc(text) + [c.im_end]
            a, b = c.add_policy(ids, [0.0] * len(ids), "stop")
            c.turns.append({"a": a, "b": b, "msg_idx": n})
    if pending:
        c.add_tool_results(pending)
    return {"prompt_ids": c.prefix_ids, "ids": c.ids, "lmask": c.lmask, "turns": c.turns}


# ================================================================ 命令
def _clean(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def load_gold(paths: list[str]) -> dict:
    gold = {}
    for p in paths:
        if p.endswith(".json"):
            for t in json.load(open(p, encoding="utf-8")):
                gold[str(t["id"])] = [a for a in (t.get("evaluation_criteria") or {}).get("actions") or [] if (a.get("requestor") or "assistant") == "assistant"]
        else:   # apigen_tasks 的 jsonl：每行一个 τ² Task 的 dict
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    t = json.loads(l)
                    gold[str(t["id"])] = [a for a in (t.get("evaluation_criteria") or {}).get("actions") or [] if (a.get("requestor") or "assistant") == "assistant"]
    return gold


def synth(a) -> None:
    rng = random.Random(a.seed)
    gold = load_gold(a.tasks)
    labels = {}
    for l in open(a.labels, encoding="utf-8") if a.labels else []:
        if l.strip():
            r = json.loads(l)
            labels[(r["policy"], r["episode_id"])] = r
    tok = None
    if a.tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tok)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_msg = open(out / "syn_msgs.jsonl", "w", encoding="utf-8")
    f_eps = open(out / "syn_eps.jsonl", "w", encoding="utf-8") if tok else None
    f_lab = open(out / "syn_steps.jsonl", "w", encoding="utf-8")
    st = collections.Counter()
    prompt_ids = None
    if a.prompt_from:   # APIGen 对话没有 token：系统提示 + 工具定义和 rollout 完全一样（sft_data 换成了 τ² 现行版本），借一条 rollout 的
        prompt_ids = json.loads(open(a.prompt_from, encoding="utf-8").readline())["prompt_ids"]
    for path in a.eps:
        pol = Path(path).name.split("_")[0]
        for line in open(path, encoding="utf-8"):
            try:
                ep = json.loads(line)
            except ValueError:
                continue
            if "reward" not in ep and "id" in ep:   # APIGen-MT 的成功对话（sft_data 的 A / B 版）：本身就是正确示范，gold = 它自己执行成功的写操作
                ep = {"episode_id": ep["id"], "task_id": ep["id"], "messages": ep["messages"], "reward": 1.0, "prompt_ids": prompt_ids, "turns": []}
                res = {m.get("id"): m for m in ep["messages"] if m.get("role") == "tool"}
                gold[ep["task_id"]] = [{"name": c["name"], "arguments": c.get("arguments") or {}} for m in ep["messages"] for c in m.get("tool_calls") or []
                                       if c["name"] in WRITE and not (res.get(c.get("id")) or {}).get("error")]
                pol = "apigen"
            tid = str(ep.get("task_id"))
            if tid not in gold or (a.labels and pol != "apigen" and (pol, ep.get("episode_id")) not in labels):
                st["skip_no_gold_or_label"] += 1
                continue
            t = Traj(ep, gold[tid], labels.get((pol, ep.get("episode_id"))))
            st["src_ok" if t.ok else "src_fail"] += 1
            for cid in a.cards:
                try:
                    samples = OPS[cid](t, rng)
                except Exception as e:  # noqa: BLE001  个别轨迹格式怪，记数跳过
                    st[f"{cid}_exception"] += 1
                    if st[f"{cid}_exception"] <= 3:
                        print("EXC", cid, ep.get("episode_id"), repr(e)[:200])
                    continue
                for s in samples:
                    st[f"{cid}_made"] += 1
                    sid = f"{cid}-{pol}-{ep['episode_id']}"
                    variants = [("syn", s.msgs[: s.target + 1], s.labels)]
                    if s.twin_cut is not None:
                        variants.append(("orig", _base(t)[: s.twin_cut + 1], s.twin_labels or {}))
                        st[f"{cid}_twin"] += 1
                    for var, ms, labs in variants:
                        eid = f"{sid}-{var}"
                        steps = [{"step": n + 1, "idx": i, "cls": labs.get(i, "OK"), "fine": labs.get(i, "OK"), "train_mask": 1}
                                 for n, i in enumerate(agent_steps(ms))]
                        meta = {"card": cid, "direction": CARDS[cid].direction, "variant": var, "pair": sid, "src_policy": pol,
                                "src_episode": ep["episode_id"], "info": s.info}
                        f_lab.write(json.dumps({"episode_id": eid, "policy": "syn", "task_id": tid, "db_ok": None, "split": a.split,
                                                "meta": meta, "steps": steps}, ensure_ascii=False) + "\n")
                        f_msg.write(json.dumps({"episode_id": eid, "meta": meta, "labels": {str(k): v for k, v in labs.items()},
                                                "messages": [dict(_clean(m), _edited=bool(m.get("_edited"))) for m in ms]}, ensure_ascii=False) + "\n")
                        if f_eps:
                            enc = encode(tok, ep, ms)
                            f_eps.write(json.dumps({"episode_id": eid, "task_id": tid, **enc, "messages": [_clean(m) for m in ms]}, ensure_ascii=False) + "\n")
    for f in (f_msg, f_eps, f_lab):
        if f:
            f.close()
    print(json.dumps(dict(sorted(st.items())), ensure_ascii=False, indent=1))


def dump_cards(a) -> None:
    ex = collections.defaultdict(list)
    if a.examples:
        for l in open(Path(a.examples) / "syn_msgs.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["meta"]["variant"] == "syn" and len(ex[r["meta"]["card"]]) < 2:
                ex[r["meta"]["card"]].append(r)
    L = ["# 客服错误卡片", "",
         "合成 critic 训练样本用的卡片：正确样本改错（inject，针对漏判）、错误样本改对（repair，针对误报和纠错），外加看着像错、其实是对（confuse，打结构捷径）。",
         "这一页由 `python -m cs_agent.error_cards cards` 从代码（`cs_agent/error_cards.py`）生成；做法 / 适用条件 / 为什么是英文，因为直接进错误生成模型的提示词。", "",
         "| 编号 | 方向 | 层 | 卡片 | 改哪里 | 第一错误落在哪 | 验证 | 状态 |", "|---|---|---|---|---|---|---|---|"]
    for c in CARDS.values():
        state = "已实现" if c.implemented else ("需续跑" if c.rollout else "待实现")
        L.append(f"| [{c.id}](#{c.id.lower()}) | {c.direction} | {c.layer} | {c.name} | {c.where} | {c.lands} | {c.verify} | {state} |")
    for c in CARDS.values():
        L += ["", f'<a id="{c.id.lower()}"></a>', "", f"### {c.id} {c.name}", "", "| | |", "|---|---|",
              f"| 做法 | {c.how} |", f"| 适用条件 | {c.when} |", f"| 为什么 | {c.why} |", f"| 第一错误落在哪 | {c.lands} |",
              f"| 怎么验证 | {c.verify} |", f"| 对应盲区 | {c.blind or '—'} |"]
        for r in ex.get(c.id, []):
            ed = [m for m in r["messages"] if m.get("_edited")]
            if ed:
                m = ed[0]
                body = (m.get("content") or "")[:300] + (" " + json.dumps(m.get("tool_calls"), ensure_ascii=False)[:200] if m.get("tool_calls") else "")
                L += ["", f"- 例子（改动后的那条消息，{json.dumps(r['meta']['info'], ensure_ascii=False)[:160]}）：`{body.replace('`', '')}`"]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("CARDS", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--eps", nargs="+", required=True)
    s.add_argument("--tasks", nargs="+", default=[])
    s.add_argument("--prompt_from", default="", help="APIGen 对话借这条 rollout 的 prompt_ids（系统提示 + 工具定义）")
    s.add_argument("--labels", default="")
    s.add_argument("--tok", default="")
    s.add_argument("--cards", nargs="+", default=[k for k in OPS if CARDS[k].implemented])
    s.add_argument("--split", default="syn")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out_dir", required=True)
    d = sub.add_parser("cards")
    d.add_argument("--out", required=True)
    d.add_argument("--examples", default="")
    a = ap.parse_args()
    {"synth": synth, "cards": dump_cards}[a.cmd](a)


if __name__ == "__main__":
    main()
