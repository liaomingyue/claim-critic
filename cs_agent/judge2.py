"""带特权信息的 9B 判断，第二版：按轨迹成败分两种问法，并标注出错之后的纠错行为。

  失败的轨迹（数据库比对不通过）：找第一个关键错误（第几步、哪类、违背了哪个标准动作 / 政策条款），
    再把它之后和这个错误有关的步骤标成四类：FIX 有效纠正 / TRY 尝试纠正（察觉了、在补救，但方法不对或没成功）/
    LOOP 无效重复（同样的做法反复试）/ IGNORE 忽略错误（没察觉，或装作没事，如工具报错了还说已办好）；
    最后说该在哪一步、怎么补救，或者写操作已经执行、无法挽回。
  成功的轨迹（数据库比对通过）：查数据库比对看不到的部分——过程违规（含泄露隐私、政策外的承诺）、回答用户的问题答错 / 漏答（E）、
    靠用户指出才改对的步骤（仍算客服的错）、侥幸成功（没核对、碰巧对）、靠模拟器放水的成功（这类不进正样本），以及纠错步骤（FIX / TRY）。
    效率（多余查询、重复追问、轮数）用代码统计，不在这里问。
  输出都是一个 JSON 对象，只列需要列的步骤，其余步骤默认正常：比第一版「每步都列」短得多，也不再要求步数对上。

  python -m cs_agent.judge2 --eps <rollout jsonl> --labels <cs_agent.label 输出> --url http://127.0.0.1:8001/v1 --out <jsonl> [--only <第一版判断的输出>]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import cs_agent  # noqa: F401
from cs_agent.judge import SYSTEM, build

ERR = "B|C|D|E"
FAILED = """The final database state is WRONG. Find the FIRST critical error: the earliest agent step that made the outcome wrong or broke the policy (use the gold actions to see what should have happened).

Then look at the agent steps AFTER that error that relate to it, and label each:
FIX = the agent noticed the problem and repaired it correctly (re-checked, corrected the arguments, asked again, told the user honestly)
TRY = the agent noticed and tried to repair it, but the method was wrong or it did not work (e.g. the change could not be undone, so it told the user and offered a transfer)
LOOP = the agent repeated the same failing attempt instead of changing approach
IGNORE = the agent did not notice, or acted as if nothing was wrong (e.g. told the user it was done after a tool error)
Only list steps that relate to the error; all other steps are treated as normal.
Think briefly: go straight to the gold actions and the steps that differ from them; do not re-read the whole conversation step by step.

Reply with ONE JSON object only:
{{"first_error_step": <step number>, "label": "<{err}>", "violated": "<the gold action or policy rule it broke>", "reason": "<short>",
 "after": [{{"step": <n>, "label": "FIX|TRY|LOOP|IGNORE", "reason": "<short>"}}],
 "should_have": "<at which step and how the agent should have recovered, or 'unrecoverable: <why>'>"}}"""
# 第四版失败问法（--prompt v4）：和 Claude 盲判对比后发现三类分歧——漏做的失败没有天然的落点、「根因」和「出事的那一步」两边各挑一个、
# 9B 自己数的步骤编号常差一两步。所以：根因步和出事步分开问；漏做统一落在「信息已经齐了却没去确认执行」的那一步；
# 每一步都要引用那一步的原文（工具名或开头几个词），由代码按原文找回实际步骤，不信模型数的编号。
FAILED4 = """The final database state is WRONG. Identify two steps (use the gold actions to see what should have happened):

ROOT step = the EARLIEST agent step such that, if that step had been done correctly, the outcome would have been right. It may be a statement to the customer (e.g. a false claim that misled the customer, suggesting the wrong option), a lookup, or an action.
DAMAGE step = the agent step that actually executed a wrong database-changing call (wrong object, wrong arguments, or a change that should not have been made). Use null if no wrong change was executed.
If the failure is an OMISSION (a required change was never made), the ROOT step is the first agent step taken when the agent already had all the information it needed to propose that change to the customer but did something else instead (more lookups, an unrelated question, ending the conversation).

For each step give "quote": copy 3-12 words EXACTLY from that step as shown in the conversation (for a call, the tool name plus an argument value, e.g. cancel_pending_order #W2575533; for a message, its opening words). The quote is used to find the step, so it must come from that step only.

Then look at the agent steps AFTER the ROOT step that relate to the error, and label each:
FIX = the agent noticed the problem and repaired it correctly (re-checked, corrected the arguments, asked again, told the user honestly)
TRY = the agent noticed and tried to repair it, but the method was wrong or it did not work
LOOP = the agent repeated the same failing attempt instead of changing approach
IGNORE = the agent did not notice, or acted as if nothing was wrong (e.g. told the user it was done after a tool error)
Only list steps that relate to the error.
Think briefly: go straight to the gold actions and the steps that differ from them; do not re-read the whole conversation step by step.

Reply with ONE JSON object only:
{{"kind": "wrong_change|omission|other", "root_step": <step number>, "root_quote": "<exact words>", "label": "<{err}>", "violated": "<the gold action or policy rule it broke>", "reason": "<short>",
 "damage_step": <step number or null>, "damage_quote": "<exact words or empty>",
 "after": [{{"step": <n>, "label": "FIX|TRY|LOOP|IGNORE", "reason": "<short>"}}],
 "should_have": "<at which step and how the agent should have recovered, or 'unrecoverable: <why>'>"}}"""

# 第五版（--prompt v5）：v4 的「根因步」让 9B 往前追得太远（和 Claude 一致率从 39% 掉到 27%），回到第三版「第一个关键错误」的问法；
# 保留 v4 的引文定位和单独的出事步；漏做落在「信息已齐、该向用户确认执行却没做」的那一步。输出字段和 v4 相同（root_* = 第一个关键错误）。
FAILED5 = """The final database state is WRONG. Find the FIRST critical error: the earliest agent step that made the outcome wrong or broke the policy (use the gold actions to see what should have happened).
If the failure is an OMISSION (a required change was never made), the first critical error is the agent step where the agent already had all the information needed to present that change to the customer for confirmation, but did something else instead (another lookup, an unrelated question, ending the conversation).
Also give the DAMAGE step: the agent step that actually executed a wrong database-changing call (wrong object, wrong arguments, or a change that should not have been made); null if no wrong change was executed. It may be the same step as the first critical error.

For each step give "quote": copy 3-12 words EXACTLY from that step as shown in the conversation (for a call, the tool name plus an argument value, e.g. cancel_pending_order #W2575533; for a message, its opening words). The quote is used to find the step, so it must come from that step only.

Then look at the agent steps AFTER the first critical error that relate to it, and label each:
FIX = the agent noticed the problem and repaired it correctly (re-checked, corrected the arguments, asked again, told the user honestly)
TRY = the agent noticed and tried to repair it, but the method was wrong or it did not work
LOOP = the agent repeated the same failing attempt instead of changing approach
IGNORE = the agent did not notice, or acted as if nothing was wrong (e.g. told the user it was done after a tool error)
Only list steps that relate to the error.
Think briefly: go straight to the gold actions and the steps that differ from them; do not re-read the whole conversation step by step.

Reply with ONE JSON object only:
{{"kind": "wrong_change|omission|other", "root_step": <step number of the first critical error>, "root_quote": "<exact words>", "label": "<{err}>", "violated": "<the gold action or policy rule it broke>", "reason": "<short>",
 "damage_step": <step number or null>, "damage_quote": "<exact words or empty>",
 "after": [{{"step": <n>, "label": "FIX|TRY|LOOP|IGNORE", "reason": "<short>"}}],
 "should_have": "<at which step and how the agent should have recovered, or 'unrecoverable: <why>'>"}}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9#]+", " ", s.lower()).strip()


def step_texts(ep: dict, idxs: list[int]) -> list[str]:
    """每个客服步骤的原文（和 judge.render 里显示的内容一致：工具名 + 参数，或说的话），已规范化，供按引文找步骤。"""
    out = []
    for i in idxs:
        m = ep["messages"][i]
        parts = [m.get("content") or ""]
        for c in m.get("tool_calls") or []:
            parts.append(c["name"] + " " + json.dumps(c.get("arguments") or {}, ensure_ascii=False))
        out.append(_norm(" ".join(parts)))
    return out


def anchor(quote: str, stated, texts: list[str]):
    """按引文找回步骤：先找整段引文出现的步骤，多个时取离模型所给编号最近的；找不到就按词重合度（≥0.6）；还不行沿用模型的编号。
    返回 (步骤, 怎么定的：exact / fuzzy / stated)。"""
    q = _norm(quote or "")
    st = stated if isinstance(stated, int) else None
    near = lambda cands: min(cands, key=lambda s: (abs(s - st) if st else 0, s))
    if q:
        hit = [n for n, t in enumerate(texts, 1) if q in t]
        if hit:
            return near(hit), "exact"
        qw = set(q.split())
        if len(qw) >= 2:
            sc = [(len(qw & set(t.split())) / len(qw), n) for n, t in enumerate(texts, 1)]
            best = max(s for s, _ in sc)
            if best >= 0.6:
                return near([n for s, n in sc if s == best]), "fuzzy"
    return st, "stated"


def parse4(text: str, n: int, texts: list[str]):
    o = _obj(text)
    if not isinstance(o, dict) or str(o.get("label", ""))[:1] not in "BCDE":
        return None
    root, how = anchor(str(o.get("root_quote", "")), o.get("root_step"), texts)
    if not (isinstance(root, int) and 1 <= root <= n):
        return None
    dmg, dhow = (None, "none")
    if o.get("damage_step") not in (None, "null", 0) or o.get("damage_quote"):
        dmg, dhow = anchor(str(o.get("damage_quote", "")), o.get("damage_step"), texts)
        if not (isinstance(dmg, int) and 1 <= dmg <= n):
            dmg, dhow = None, "none"
    ok = lambda s: isinstance(s, int) and 1 <= s <= n
    after = [{"step": x["step"], "label": str(x.get("label", "")).upper(), "reason": str(x.get("reason", ""))[:300]}
             for x in o.get("after") or [] if isinstance(x, dict) and ok(x.get("step")) and x["step"] > root
             and str(x.get("label", "")).upper() in ("FIX", "TRY", "LOOP", "IGNORE")]
    return {"first_error_step": root, "root_stated": o.get("root_step"), "root_anchor": how, "root_quote": str(o.get("root_quote", ""))[:200],
            "damage_step": dmg, "damage_stated": o.get("damage_step"), "damage_anchor": dhow,
            "kind": str(o.get("kind", ""))[:20], "label": str(o["label"])[:1], "violated": str(o.get("violated", ""))[:300],
            "reason": str(o.get("reason", ""))[:300], "after": after, "should_have": str(o.get("should_have", ""))[:400]}


SUCCESS = """The final database state is CORRECT. Now check what the database check cannot see:
1. violations: agent steps that broke the policy or were wrong even though the outcome came out right (label {err}; see the label definitions above). Include revealing private information (e.g. reading out the user's email or another user's data) and promises or procedures the policy does not allow.
2. info_errors: agent messages that answered the customer's questions wrongly or incompletely, or left a question unanswered (tracking numbers, refund amounts or timing, price differences, order status, item options). These are label E.
3. user_corrected: agent steps that were wrong and only got fixed because the CUSTOMER pointed it out (the agent's step is still an error).
4. lucky: true if the right outcome came by luck rather than by checking (e.g. the agent picked an item or payment method without verifying and happened to be right).
5. sim_leak: true if the success relied on the customer simulator misbehaving (revealing information its instructions say to hide, or giving in without reason).
6. recoveries: steps where the agent noticed an earlier mistake of its own (or a tool error it caused) and repaired it (FIX) or tried to (TRY).
Most correct conversations have empty lists and false flags. Think briefly: check the listed points only.

Reply with ONE JSON object only:
{{"violations": [{{"step": <n>, "label": "<{err}>", "reason": "<short>"}}],
 "info_errors": [{{"step": <n>, "reason": "<short>"}}],
 "user_corrected": [{{"step": <n>, "reason": "<short>"}}],
 "lucky": false, "lucky_reason": "", "sim_leak": false, "sim_leak_reason": "",
 "recoveries": [{{"step": <n>, "label": "FIX|TRY", "about_step": <n>, "reason": "<short>"}}]}}"""


# 第六版成功问法（--prompt v6，只判成功轨迹）：在 SUCCESS 上加第 7 项 overfix——原本是对的，客服硬去「纠正」并且真的改了 / 发了改动请求。
SUCCESS6 = SUCCESS.replace(
    "Most correct conversations have empty lists",
    "7. overfix: agent steps that \"corrected\" something that was already right: a database-changing call (or an attempt at one) that changed, "
    "redid or undid an item, address, payment method, order or user detail that was already correct, or pressed the customer to change a correct choice. "
    "A reasonable confirmation question is NOT overfix; only a step that actually acted (or pushed to act) on something already correct.\n"
    "Most correct conversations have empty lists").replace(
    ' "recoveries": [{{"step": <n>, "label": "FIX|TRY", "about_step": <n>, "reason": "<short>"}}]}}',
    ' "recoveries": [{{"step": <n>, "label": "FIX|TRY", "about_step": <n>, "reason": "<short>"}}],\n "overfix": [{{"step": <n>, "reason": "<short>"}}]}}')
assert "7. overfix" in SUCCESS6 and '"overfix": [' in SUCCESS6

# 失败轨迹的 OVERFIX 补问（--prompt ofx）：第一个错误已由第三版定好（--ref），只问这一步是不是「把原本对的改错了」。
OFX = """The final database state is WRONG, and the first critical error has been located at step [S{step}]:
{desc}

Question: was step [S{step}] an OVERFIX, i.e. the agent "corrected" or changed something that was ALREADY RIGHT (an item, address, payment method, order, user detail, or a correct earlier action of its own; or it told the customer a correct choice was wrong and pushed a change), rather than a mistake on something not yet done right (wrong lookup, wrong new arguments, skipped step, false statement, omission)?
Think briefly.

Reply with ONE JSON object only:
{{"overfix": true|false, "reason": "<short>"}}"""

def _obj(text: str):
    text = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def parse(text: str, n: int, failed: bool):
    o = _obj(text)
    if not isinstance(o, dict):
        return None
    ok = lambda s: isinstance(s, int) and 1 <= s <= n
    if failed:
        fe = o.get("first_error_step")
        if not ok(fe) or str(o.get("label", ""))[:1] not in "BCDE":
            return None
        after = [{"step": x["step"], "label": str(x.get("label", "")).upper(), "reason": str(x.get("reason", ""))[:300]}
                 for x in o.get("after") or [] if isinstance(x, dict) and ok(x.get("step")) and x["step"] > fe
                 and str(x.get("label", "")).upper() in ("FIX", "TRY", "LOOP", "IGNORE")]
        return {"first_error_step": fe, "label": str(o["label"])[:1], "violated": str(o.get("violated", ""))[:300],
                "reason": str(o.get("reason", ""))[:300], "after": after, "should_have": str(o.get("should_have", ""))[:400]}
    viol = [{"step": x["step"], "label": str(x.get("label", ""))[:1], "reason": str(x.get("reason", ""))[:300]}
            for x in o.get("violations") or [] if isinstance(x, dict) and ok(x.get("step")) and str(x.get("label", ""))[:1] in "BCDE"]
    rec = [{"step": x["step"], "label": str(x.get("label", "")).upper(), "about_step": x.get("about_step"), "reason": str(x.get("reason", ""))[:300]}
           for x in o.get("recoveries") or [] if isinstance(x, dict) and ok(x.get("step")) and str(x.get("label", "")).upper() in ("FIX", "TRY")]
    if "violations" not in o and "recoveries" not in o:
        return None
    steps_only = lambda k: [{"step": x["step"], "reason": str(x.get("reason", ""))[:300]} for x in o.get(k) or [] if isinstance(x, dict) and ok(x.get("step"))]
    return {"violations": viol, "recoveries": rec, "info_errors": steps_only("info_errors"), "user_corrected": steps_only("user_corrected"), "overfix": steps_only("overfix"),
            "lucky": bool(o.get("lucky")), "lucky_reason": str(o.get("lucky_reason", ""))[:300],
            "sim_leak": bool(o.get("sim_leak")), "sim_leak_reason": str(o.get("sim_leak_reason", ""))[:300]}


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
    ap.add_argument("--only", default="", help="只判这个文件里出现过的轨迹（如第一版判断的输出，便于两版对比）")
    ap.add_argument("--succ_frac", type=float, default=1.0, help="成功的轨迹只判这个比例（按 episode_id 的哈希固定抽样，续跑时去留不变）；失败的全部判")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--think_budget", type=int, default=2048, help="思考最多这么多 token（0 = 不限）")
    ap.add_argument("--answer_tokens", type=int, default=1500)
    ap.add_argument("--tok", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt", default="v3", choices=["v3", "v4", "v5", "v6", "ofx"], help="v4：失败轨迹的新问法（根因步 + 出事步 + 引文定位），只判失败的轨迹")
    ap.add_argument("--ref", default="", help="--prompt ofx：第三版判断的输出（取每条失败轨迹的第一个错误）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t.id: t for t in load_tasks(a.split)}
    policy = get_environment().policy
    eps = {}
    for l in open(a.eps, encoding="utf-8"):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        eps[e["episode_id"]] = e
    labs = [x for x in (json.loads(l) for l in open(a.labels, encoding="utf-8")) if x["episode_id"] in eps]
    if a.succ_frac < 1.0:
        import hashlib
        frac = lambda eid: int(hashlib.md5(eid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        labs = [x for x in labs if eps[x["episode_id"]].get("reward") != 1.0 or frac(x["episode_id"]) < a.succ_frac]
    if a.prompt in ("v4", "v5"):
        labs = [x for x in labs if eps[x["episode_id"]].get("reward") != 1.0]
    if a.prompt == "v6":
        labs = [x for x in labs if eps[x["episode_id"]].get("reward") == 1.0]
    ref = {}
    if a.prompt == "ofx":
        for l in open(a.ref, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("failed") and r.get("mode") == "failed":
                ref[r["episode_id"]] = r
        labs = [x for x in labs if x["episode_id"] in ref]
    if a.only:
        keep = {json.loads(l)["episode_id"] for l in open(a.only, encoding="utf-8")}
        labs = [x for x in labs if x["episode_id"] in keep]
    done = set()
    try:
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("failed") and not (r.get("mode") == "success" and r.get("pv", 2) < 3):   # 成功轨迹的旧版问法（没有答错 / 侥幸 / 放水字段）重判
                done.add(r["episode_id"])
    except FileNotFoundError:
        pass
    labs = [x for x in labs if x["episode_id"] not in done]
    print(f"JUDGE2 {len(labs)} to judge, {len(done)} already judged", flush=True)
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)

    def one(lab):
        ep = eps[lab["episode_id"]]
        failed = ep.get("reward") != 1.0
        msgs, idxs = build(ep, tasks[ep["task_id"]], policy)
        msgs[0] = {"role": "system", "content": msgs[0]["content"]}
        if a.prompt == "ofx":
            rf = ref[lab["episode_id"]]
            q = OFX.format(step=rf["first_error_step"], desc=f"label {rf['label']}; broke: {rf.get('violated', '')}; why: {rf.get('reason', '')}")
        else:
            q = ({"v4": FAILED4, "v5": FAILED5, "v6": SUCCESS6}.get(a.prompt) or (FAILED if failed else SUCCESS)).format(err=ERR)
        texts = step_texts(ep, idxs)
        msgs[1] = {"role": "user", "content": msgs[1]["content"].split("Label every agent step")[0] + q}
        raws = []
        for think in (True, False):
            if think and a.think_budget:   # 思考预算（Qwen 官方做法）：思考最多 think_budget token，没想完就补一句「现在必须作答」并手动结束思考，再续写答案
                r = cl.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, max_tokens=a.think_budget,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": True}})
                text, fin = r.choices[0].message.content or "", r.choices[0].finish_reason
                if "</think>" not in text and fin == "length":
                    pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    stop = "\n\nConsidering the limited time, I have to give the final answer now.\n</think>\n\n"
                    r2 = cl.completions.create(model=a.model, prompt=pre + text + stop, temperature=0.0, max_tokens=a.answer_tokens)
                    text, fin = text + stop + (r2.choices[0].text or ""), r2.choices[0].finish_reason
                elif "</think>" in text and fin == "length":   # 想完了但答案没写完：接着写
                    pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    r2 = cl.completions.create(model=a.model, prompt=pre + text, temperature=0.0, max_tokens=a.answer_tokens)
                    text, fin = text + (r2.choices[0].text or ""), r2.choices[0].finish_reason
            else:
                r = cl.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, max_tokens=a.max_tokens,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": think}})
                text, fin = r.choices[0].message.content or "", r.choices[0].finish_reason
            if a.prompt == "ofx":
                ob = _obj(text)
                p = {"first_error_step": ref[lab["episode_id"]]["first_error_step"], "overfix": bool(ob.get("overfix")), "overfix_reason": str(ob.get("reason", ""))[:300]} \
                    if isinstance(ob, dict) and isinstance(ob.get("overfix"), bool) else None
            else:
                p = parse4(text, len(idxs), texts) if a.prompt in ("v4", "v5") else parse(text, len(idxs), failed)
            if p is not None:
                return {"episode_id": lab["episode_id"], "task_id": ep["task_id"], "mode": "failed" if failed else "success", "think": think, "pv": {"v4": 4, "v5": 5, "v6": 6, "ofx": 7}.get(a.prompt, 3), "budget": a.think_budget if think else 0,
                        "n_steps": len(idxs), "step_idx": idxs, **p}
            raws.append({"think": think, "finish": fin, "n_chars": len(text), "tail": text[-600:]})
        return {"episode_id": lab["episode_id"], "failed": True, "mode": "failed" if failed else "success", "raw": raws}
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
            f.flush()   # 每条立即落盘：中途重启不丢已判的结果
            if j % 25 == 0:
                print(f"  judged {j}/{len(labs)} ok {n_ok}", flush=True)
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    good = [r for r in rows if not r.get("failed")]
    st = {"trajectories": len(rows), "parsed": len(good), "modes": dict(collections.Counter(r["mode"] for r in good)),
          "first_error_labels": dict(collections.Counter(r.get("label") for r in good if r["mode"] == "failed")),
          "after_labels": dict(collections.Counter(x["label"] for r in good if r["mode"] == "failed" for x in r.get("after") or [])),
          "success_with_violation": sum(bool(r.get("violations")) for r in good if r["mode"] == "success"),
          "recoveries": dict(collections.Counter(x["label"] for r in good if r["mode"] == "success" for x in r.get("recoveries") or [])),
          "success_info_error": sum(bool(r.get("info_errors")) for r in good if r["mode"] == "success"),
          "success_user_corrected": sum(bool(r.get("user_corrected")) for r in good if r["mode"] == "success"),
          "success_lucky": sum(bool(r.get("lucky")) for r in good if r["mode"] == "success"),
          "success_sim_leak": sum(bool(r.get("sim_leak")) for r in good if r["mode"] == "success"),
          "success_overfix": sum(bool(r.get("overfix")) for r in good if r["mode"] == "success"),
          "failed_overfix": sum(r.get("overfix") is True for r in good if r["mode"] == "failed")}
    print("JUDGE2_DONE", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
