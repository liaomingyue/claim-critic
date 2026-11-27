"""客服 critic 的错误体系：步骤模型（过程）和结果模型各自漏判 / 误报了什么，按机制 × 结构拆开。

两个方向：
  漏判  实际错、模型判对（步骤：第一错误没判成 ERR；结果：数据库错的轨迹 P(好) > 0.5）
  误报  实际对、模型判错（步骤：标 OK 的步骤判成 ERR / AFTER_ERR；结果：干净成功轨迹 P(好) <= 0.5）
P(好) = GOOD / (GOOD + FAIL)，和 GRPO 奖励同一个口径。

三个子命令：
  extract  在有原始文件的机器上（共享盘）把留出集轨迹、标签、两个 critic 的分数、判断理由抽成一个精简 jsonl
           python scripts/analysis/cs_critic_errors.py extract --dir outputs/rt/cs/sample_r0 --steps outputs/rt/cs/sample_r0/steps_cls_v1.jsonl \
             --step_scores outputs/rt/cs/critic_step/scores.jsonl --outcome_scores outputs/rt/cs/critic_outcome/scores.jsonl \
             --claude outputs/rt/cs/claude_arb/claude.jsonl --out outputs/rt/cs/blind/blind_ex.jsonl
  report   读精简 jsonl，打印错误体系的各张表，并写 json
           python scripts/analysis/cs_critic_errors.py report --ex outputs/rt/cs/blind/blind_ex.jsonl --out outputs/rt/cs/blind/report.json
  online   读 GRPO 存的采样（策略自己的分布，带两个 critic 的分数和数据库结果）：题内对比数、结果模型题内 AUC、步骤误报的结构
           python scripts/analysis/cs_critic_errors.py online --samples '<runs>/samples/*.jsonl.gz' --out <json>

标签噪声（report 里单独计数，并给出剔除后的召回）：
  R5     判断把「说话的同时调工具」当成违规（规则库里 R5 只记 tag，不算错）
  SIM    核实身份的调用报错，但参数全部出自用户说过的话：查不到是用户模拟器给错了信息（剧本里有两个邮箱只报一个、报了新地址的邮编），
         不是客服的错
机制是按 9B 判断理由的关键词粗分的（MECH 表，按顺序取第一条命中），够看分布；要精确的分机制召回，后面换成按机制提问的 9B 标注。
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs_agent.rules import AUTH, WRITE  # noqa: E402

TRANSFER = "transfer_to_human_agents"
R5_RE = re.compile(r"at the same time|simultaneously|while (executing|making) a tool call|while responding", re.I)
# (机制, 所属大类, 理由里的关键词)。大类：F1 不一致 / F2 缺失 / F3 理解 / F4 写参数 / F5 合规
MECH = [
    ("隐私泄露", "F5", r"private|reveal|privacy"),
    ("转人工不当", "F2", r"transferr?ed|transfer to|human agent"),
    ("遗漏：信息齐了 / 已确认却没执行", "F2", r"never (executed|proceeded|processed|performed|asked|confirmed|made|calculated)|did not (execute|perform|process)|failed to (execute|complete|process)|without (executing|performing)|not executed|but never|only executed|only processed"),
    ("范围误解：多做 / 少做对象", "F3", r"not part of|beyond|scope|all (4 )?items|both (items|orders)|entire order|not mentioned|not requested|only .{0,40}should be|ONLY want|only want"),
    ("查找不全 / 身份核实", "F2", r"other email|second email|another (valid )?email|two (valid )?email|alternative|user not found|zip code|only checked|only tried|without (first )?checking|should have checked|incomplete lookup"),
    ("退款 / 支付方式参数", "F4", r"payment method|original payment|gift card"),
    ("约束下选错候选", "F3", r"instead of|wrong (item|order|new_item|variant)|cheapest|most expensive|shorter|capacity|which order|wrong address"),
    ("政策误述 / 错误拒绝", "F3", r"incorrectly stated|policy allows|cannot be (cancelled|modified)|refused|misstat|misunderstanding"),
    ("顺序 / 状态依赖违规", "F5", r"after modifying|item modified|only be called once|once per order"),
]
GROUP = {"F1": "不一致（和工具结果 / 前文对不上）", "F2": "缺失（该做的没做）", "F3": "理解（意图 / 约束 / 政策）", "F4": "写操作参数", "F5": "合规", "F0": "未归类"}


def rd(p):
    out = []
    op = gzip.open if str(p).endswith(".gz") else open
    try:
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return out


def clip(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def agent_steps(msgs):
    return [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and not (i == 0 and not m.get("tool_calls"))]


def sim_auth(msgs, idx) -> bool:
    """核实身份的调用报错，且每个参数值都在之前的用户消息里出现过 → 查不到是模拟器给的信息不对。"""
    m = msgs[idx]
    tc = m.get("tool_calls") or []
    if not tc or tc[0]["name"] not in AUTH or idx + 1 >= len(msgs):
        return False
    if not str(msgs[idx + 1].get("content", "")).lower().startswith("error"):
        return False
    said = " ".join(str(x.get("content", "")) for x in msgs[:idx] if x.get("role") == "user").lower()
    args = tc[0].get("arguments") or {}
    vals = [str(v).strip().lower() for v in args.values()]
    return bool(vals) and all(v and v in said for v in vals)


# ---------------------------------------------------------------- extract
def extract(a) -> None:
    D = Path(a.dir)
    ds = {(x["policy"], x["episode_id"]): x for x in rd(a.steps) if x["split"] in a.splits}
    sc = collections.defaultdict(dict)
    for r in rd(a.step_scores):
        sc[(r["policy"], r["episode_id"])][r["step"]] = r["probs"]
    oc = {(r["policy"], r["episode_id"]): r["probs"] for r in rd(a.outcome_scores)}
    judge = collections.defaultdict(dict)
    for p in a.pols:
        for name in ("judge2", "judge6", "judgeofx"):
            for r in rd(D / f"{p}_{name}.jsonl"):
                judge[(p, r.get("episode_id"))][name] = {k: r.get(k) for k in ("mode", "first_error_step", "label", "violated", "reason", "after",
                                                                               "should_have", "violations", "info_errors", "overfix") if k in r}
    claude = {}
    for r in rd(a.claude) if a.claude else []:
        pol, eid = r["id"].split(":")
        claude[(pol, eid)] = r["claude"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for p in a.pols:
            for e in rd(D / f"{p}_train.jsonl"):
                k = (p, e["episode_id"])
                if k not in ds:
                    continue
                x, msgs = ds[k], e["messages"]
                steps = []
                for st in x["steps"]:
                    m = msgs[st["idx"]]
                    tc = m.get("tool_calls") or []
                    res = msgs[st["idx"] + 1] if tc and st["idx"] + 1 < len(msgs) else None
                    prev_user = next((msgs[j]["content"] for j in range(st["idx"] - 1, -1, -1) if msgs[j]["role"] == "user"), "")
                    steps.append(dict(step=st["step"], cls=st["cls"], fine=st.get("fine"),
                                      call=[{"name": t["name"], "args": t.get("arguments")} for t in tc] or None,
                                      text=clip(m.get("content"), 600), result=clip(res.get("content"), 300) if res else None,
                                      prev_user=clip(prev_user, 300), sim_auth=sim_auth(msgs, st["idx"]), probs=sc[k].get(st["step"])))
                out.write(json.dumps(dict(policy=p, episode_id=e["episode_id"], task_id=e["task_id"], split=x["split"], db_ok=x["db_ok"],
                                          meta=x["meta"], outcome=oc.get(k), judge=judge.get(k), claude=claude.get(k), steps=steps,
                                          n_tool_errors=e.get("n_tool_errors"), term=e.get("term")), ensure_ascii=False) + "\n")
                n += 1
    print("EXTRACT", n, a.out)


# ---------------------------------------------------------------- report
def tool_of(s) -> str:
    if not s.get("call"):
        return "text"
    c = s["call"][0]["name"]
    return "write:" + c if c in WRITE else "auth" if c in AUTH else "transfer" if c == TRANSFER else "read:" + c


def pgood(p) -> float:
    return p["GOOD"] / (p["GOOD"] + p["FAIL"])


def why(e, step) -> str:
    j = e.get("judge") or {}
    if not e["db_ok"] and j.get("judge2"):
        s = j["judge2"]
        return f'{s.get("violated")} {s.get("reason")}'
    src = j.get("judge6") or j.get("judge2") or {}
    txt = [v.get("reason", "") for key in ("violations", "info_errors", "overfix") for v in src.get(key) or []
           if isinstance(v, dict) and v.get("step") == step]
    return " ".join(txt)


def mechanism(e, s) -> tuple[str, str, str | None]:
    """→ (机制, 大类, 噪声类型或 None)。"""
    w = why(e, s["step"])
    t = tool_of(s)
    if s.get("sim_auth"):
        return "核实身份：模拟器给错信息", "F0", "SIM"
    if e["db_ok"] and R5_RE.search(w):
        return "R5 说话同时调工具", "F0", "R5"
    if s.get("fine") == "INFO_ERR":
        return "事实误述（和工具结果 / 前文不符）", "F1", None
    if t == "transfer":
        return "转人工不当", "F2", None
    for name, g, pat in MECH:
        if re.search(pat, w, re.I):
            return name, g, None
    return "未归类", "F0", None


def auc(pos, neg) -> float | None:
    if not pos or not neg:
        return None
    w = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return round(w / (len(pos) * len(neg)), 3)


def within_task_auc(items) -> tuple[float | None, int]:
    """items: [(task, db_ok, score)] → 同一道题内 db 对 vs 错的 AUC 和对数。"""
    by = collections.defaultdict(lambda: ([], []))
    for t, ok, q in items:
        by[t][0 if ok else 1].append(q)
    w = n = 0
    for P, N in by.values():
        for a in P:
            for b in N:
                n += 1
                w += (a > b) + 0.5 * (a == b)
    return (round(w / n, 3) if n else None), n


def table(title, rows, key, metric_names, metrics, out, min_n=1):
    g = collections.defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    print(f"\n== {title}")
    res = []
    for k, v in sorted(g.items(), key=lambda t: -len(t[1])):
        if len(v) < min_n:
            continue
        vals = [m(v) for m in metrics]
        res.append({"key": k if isinstance(k, (str, int, float, bool)) or k is None else list(k), "n": len(v), **dict(zip(metric_names, vals))})
        print(f"  {str(k):40s} n={len(v):5d}  " + "  ".join(f"{nm}={x:.2f}" if isinstance(x, float) else f"{nm}={x}" for nm, x in zip(metric_names, vals)))
    out[title] = res


def report(a) -> None:
    E = rd(a.ex)
    out = {"n_traj": len(E)}
    # ---- 步骤模型 漏判：第一错误（ERR）步
    err = []
    for e in E:
        st = [s for s in e["steps"] if s.get("probs")]
        pe = [s["probs"]["ERR"] for s in st]
        for i, s in enumerate(st):
            if s["cls"] != "ERR":
                continue
            m, g, noise = mechanism(e, s)
            pred = max(s["probs"], key=s["probs"].get)
            err.append(dict(split=e["split"], db=e["db_ok"], tool=tool_of(s), kind=tool_of(s).split(":")[0], mech=m, group=g, noise=noise,
                            hit=pred == "ERR", pred=pred, top1=pe[i] == max(pe), p=s["probs"]["ERR"], pos=min(2, int(3 * i / len(st)))))
    rec = [lambda v: sum(r["hit"] for r in v) / len(v), lambda v: sum(r["top1"] for r in v) / len(v), lambda v: sum(r["p"] for r in v) / len(v)]
    names = ["召回", "轨迹内最高", "平均P(ERR)"]
    clean = [r for r in err if not r["noise"]]
    print(f"步骤模型 第一错误步 {len(err)}，其中标签噪声 {len(err) - len(clean)}（R5 {sum(r['noise'] == 'R5' for r in err)}，SIM {sum(r['noise'] == 'SIM' for r in err)}）")
    print(f"召回：全部 {sum(r['hit'] for r in err) / max(1, len(err)):.3f}  剔除噪声后 {sum(r['hit'] for r in clean) / max(1, len(clean)):.3f}")
    out["step_fn_recall"] = {"all": round(sum(r["hit"] for r in err) / max(1, len(err)), 3), "clean": round(sum(r["hit"] for r in clean) / max(1, len(clean)), 3)}
    table("步骤漏判 × 大类（剔除噪声）", clean, lambda r: GROUP[r["group"]], names, rec, out)
    table("步骤漏判 × 机制（含噪声行）", err, lambda r: r["mech"], names, rec, out)
    table("步骤漏判 × 步骤种类（剔除噪声）", clean, lambda r: r["tool"], names, rec, out)
    table("步骤漏判 × 位置（前 / 中 / 后 1/3，剔除噪声）", clean, lambda r: r["pos"], names, rec, out)
    table("步骤漏判时判成了什么", [r for r in clean if not r["hit"]], lambda r: r["pred"], ["占比"], [lambda v: len(v) / max(1, sum(not r["hit"] for r in clean))], out)
    # ---- 步骤模型 误报：标 OK 的步
    ok = []
    for e in E:
        nw = 0
        st = [s for s in e["steps"] if s.get("probs")]
        for i, s in enumerate(st):
            t = tool_of(s)
            if s["cls"] == "OK":
                pred = max(s["probs"], key=s["probs"].get)
                ok.append(dict(db=e["db_ok"], tool=t, kind=t.split(":")[0], nw=min(nw, 2), pos=min(2, int(3 * i / len(st))), pred=pred))
            if t.startswith("write"):
                nw += 1
    fpm = [lambda v: sum(r["pred"] in ("ERR", "AFTER_ERR") for r in v) / len(v), lambda v: sum(r["pred"] == "ERR" for r in v) / len(v),
           lambda v: sum(r["pred"] == "AFTER_ERR" for r in v) / len(v)]
    fpn = ["误报", "判ERR", "判AFTER"]
    succ = [r for r in ok if r["db"]]
    table("步骤误报（成功轨迹）× 种类", succ, lambda r: r["tool"], fpn, fpm, out)
    table("步骤误报（成功轨迹）× (种类, 之前写操作数)", succ, lambda r: (r["kind"] == "write", r["nw"]), fpn, fpm, out)
    table("步骤误报（成功轨迹）× 位置", succ, lambda r: r["pos"], fpn, fpm, out)
    table("步骤误报（失败轨迹第一错误之前）× 种类", [r for r in ok if not r["db"]], lambda r: r["kind"], fpn, fpm, out)
    # ---- 结果模型
    first = {}
    for e in E:
        for s in e["steps"]:
            if s["cls"] == "ERR" and s.get("probs"):
                first[(e["policy"], e["episode_id"])] = mechanism(e, s)
                break
    O = []
    for e in E:
        if not e.get("outcome"):
            continue
        ws = sorted({tool_of(s).split(":")[1] for s in e["steps"] if tool_of(s).startswith("write")})
        nwr = sum(tool_of(s).startswith("write") for s in e["steps"])
        cleanok = e["db_ok"] and not any(s["cls"] == "ERR" for s in e["steps"])
        m = first.get((e["policy"], e["episode_id"]), ("没有第一错误标签", "F0", None))
        O.append(dict(task=e["task_id"], db=e["db_ok"], clean=cleanok, q=pgood(e["outcome"]), mech=m[0], noise=m[2], nw=min(nwr, 3), ws="+".join(ws) or "-"))
    fail = [o for o in O if not o["db"]]
    good = [o for o in O if o["clean"]]
    print(f"\n结果模型：失败轨迹判好 {sum(o['q'] > .5 for o in fail) / max(1, len(fail)):.3f}（n={len(fail)}），干净成功轨迹判坏 {sum(o['q'] <= .5 for o in good) / max(1, len(good)):.3f}（n={len(good)}）")
    wa, wn = within_task_auc([(o["task"], o["db"], o["q"]) for o in O])
    allauc = auc([o["q"] for o in O if o["db"]], [o["q"] for o in fail])
    print(f"结果模型 AUC：整体 {allauc}，题内 {wa}（对数 {wn}）")
    out["outcome"] = {"fail_judged_good": round(sum(o["q"] > .5 for o in fail) / max(1, len(fail)), 3),
                      "clean_judged_bad": round(sum(o["q"] <= .5 for o in good) / max(1, len(good)), 3), "auc_all": allauc, "auc_within_task": wa, "pairs": wn}
    table("结果漏判（失败轨迹判好）× 第一错误机制", fail, lambda o: o["mech"], ["判好", "平均P(好)"],
          [lambda v: sum(o["q"] > .5 for o in v) / len(v), lambda v: sum(o["q"] for o in v) / len(v)], out)
    table("结果误报（干净成功轨迹判坏）× 写操作次数", good, lambda o: o["nw"], ["判坏", "平均P(好)"],
          [lambda v: sum(o["q"] <= .5 for o in v) / len(v), lambda v: sum(o["q"] for o in v) / len(v)], out)
    table("结果误报（干净成功轨迹判坏）× 写操作组合", good, lambda o: o["ws"], ["判坏", "平均P(好)"],
          [lambda v: sum(o["q"] <= .5 for o in v) / len(v), lambda v: sum(o["q"] for o in v) / len(v)], out, min_n=5)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\nREPORT", a.out)


# ---------------------------------------------------------------- online（GRPO 采样）
def online(a) -> None:
    files = sorted({f for pat in a.samples for f in glob.glob(pat)})
    groups = collections.defaultdict(list)
    fp = collections.Counter()
    tot = collections.Counter()
    n = 0
    for f in files:
        step = int(re.search(r"step(\d+)", f).group(1))
        for d in rd(f):
            n += 1
            q = d["p_out"][0] / (d["p_out"][0] + d["p_out"][2])
            groups[(f.split("/samples")[0], step, d["task_id"])].append((bool(d["db"]), q))
            if not d["db"] or d["s1"]:
                continue
            nw = 0
            for k, i in enumerate(agent_steps(d["messages"])):
                if k >= len(d["p_step"]):
                    break
                tc = d["messages"][i].get("tool_calls") or []
                isw = bool(tc) and tc[0]["name"] in WRITE
                ps = d["p_step"][k]
                key = f"{'写操作' if isw else '其他'}, 之前写过 {min(nw, 2)} 次"
                tot[key] += 1
                fp[key] += max(range(len(ps)), key=lambda j: ps[j]) in (1, 2)   # 类别顺序 OK ERR AFTER_ERR FIX TRY
                nw += isw
    mixed = [g for g in groups.values() if len({x[0] for x in g}) == 2]
    wa, wn = within_task_auc([(k, ok, q) for k, g in groups.items() for ok, q in g])
    res = {"trajectories": n, "groups": len(groups), "mixed_groups": len(mixed), "mixed_success": sum(x[0] for g in mixed for x in g),
           "mixed_fail": sum(not x[0] for g in mixed for x in g), "outcome_auc_within_task": wa, "pairs": wn,
           "step_fp_success": {k: {"n": tot[k], "fp": round(fp[k] / tot[k], 3)} for k in sorted(tot)}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("--dir", required=True)
    x.add_argument("--steps", required=True)
    x.add_argument("--step_scores", required=True)
    x.add_argument("--outcome_scores", required=True)
    x.add_argument("--claude", default="")
    x.add_argument("--pols", nargs="+", default=["base9", "sftB"])
    x.add_argument("--splits", nargs="+", default=["test", "claude"])
    x.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--ex", required=True)
    r.add_argument("--out", default="")
    o = sub.add_parser("online")
    o.add_argument("--samples", nargs="+", required=True)
    o.add_argument("--out", default="")
    a = ap.parse_args()
    {"extract": extract, "report": report, "online": online}[a.cmd](a)


if __name__ == "__main__":
    main()
