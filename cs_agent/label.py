"""客服轨迹的自动标注（不调 LLM）：轨迹级标签给结果模型，步骤级标签给步骤模型。

  python -m cs_agent.label --eps <rollout 的 jsonl> [--split train] --out <标注 jsonl>   # 末尾打印统计

轨迹级（结果模型的标签）：
  正样本 = 数据库比对通过，且没有严重违规（S1，执行了或被工具拒了都算）；
  负样本 = 数据库比对不通过，或者结果对但有严重违规。其余违规（S2/S3）记在 tags 里，不改标签。
步骤级（步骤模型的标签，方案里的 A–E）：
  A 正常 / B 查询错误 / C 操作错误 / D 不合规 / E 无依据陈述（代码判不了，这里不出，留给 critic 和人工）
  错误步骤从三处来：
    1. 规则库命中的步骤（cs_agent.rules：R7 → B，其余 S1/S2 → D；R5、R2b、R10 太常见或太轻，只记 tag 不当错）；
    2. 和任务的标准动作对照：写操作的名字和参数对不上任何一个标准写操作 → C（标准动作里根本没有这个写操作也算）；
    3. 写操作被工具拒了（报错）→ C；同时命中规则的按规则（D）。
  第一个错误步骤之前的步骤标 A；之后的标 U（前面已经错了，上下文不干净，先不用）。
  结果不对、但上面三处都没定位到错误步骤的轨迹（多半是漏做了写操作，或错在对话里），步骤全标 U，轨迹级照常是负样本，
  这部分正是规则和对照都看不见的，后面单独看（候选的盲区）。
"""
from __future__ import annotations

import argparse
import collections
import json

import cs_agent  # noqa: F401
from cs_agent.rules import WRITE, check

RULE_LABEL = {"R7": "B"}
SOFT = {"R5", "R2b", "R10"}   # 只记 tag


def _norm(name: str, args: dict) -> tuple:
    """写操作参数的比较形式：item_ids / new_item_ids 按位置配对后排序（顺序不同不算错）。"""
    a = dict(args)
    if "item_ids" in a and "new_item_ids" in a and isinstance(a["item_ids"], list) and isinstance(a["new_item_ids"], list):
        pairs = sorted(zip(map(str, a.pop("item_ids")), map(str, a.pop("new_item_ids"))))
        a["_pairs"] = pairs
    elif "item_ids" in a and isinstance(a["item_ids"], list):
        a["item_ids"] = sorted(map(str, a["item_ids"]))
    return name, json.dumps({k: a[k] for k in sorted(a)}, sort_keys=True, ensure_ascii=False)


def label_episode(ep: dict, task) -> dict:
    msgs = ep["messages"]
    gold = [a for a in (task.evaluation_criteria.actions or []) if (a.requestor or "assistant") == "assistant"]
    gold_w = collections.Counter(_norm(a.name, a.arguments) for a in gold if a.name in WRITE)
    results = {m.get("id"): bool(m.get("error")) for m in msgs if m.get("role") == "tool"}
    viol = check(msgs)
    by_idx = collections.defaultdict(list)
    for v in viol:
        by_idx[v["idx"]].append(v)
    steps, first_err = [], None
    used = collections.Counter()
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or m.get("greeting") or (i == 0 and not m.get("tool_calls")):
            continue
        calls = m.get("tool_calls") or []
        lab, why = "A", []
        tags = sorted({v["rule"] for v in by_idx.get(i, [])})
        hard = [v for v in by_idx.get(i, []) if v["rule"] not in SOFT]
        if hard:
            lab = "B" if all(RULE_LABEL.get(v["rule"]) == "B" for v in hard) else "D"
            why += [f"{v['rule']}: {v['detail'][:120]}" for v in hard]
        for c in calls:
            if c.get("name") not in WRITE:
                continue
            key = _norm(c["name"], c.get("arguments") or {})
            if used[key] < gold_w[key]:
                used[key] += 1
                if results.get(c.get("id")):   # 和标准一致却被工具拒了：多半是前面状态已经不对
                    why.append(f"write {c['name']} matches the gold action but was rejected")
                    lab = lab if lab != "A" else "C"
                continue
            why.append(f"write {c['name']} does not match any gold write action" + (" (rejected by the tool)" if results.get(c.get("id")) else ""))
            lab = "D" if lab == "D" else "C"
        steps.append({"idx": i, "kind": "call" if calls else "text", "label": lab, "tags": tags, "reason": why})
        if lab != "A" and first_err is None:
            first_err = i
    ok = ep.get("reward") == 1.0
    s1 = [v for v in viol if v["sev"] == "S1"]
    traj = "pos" if ok and not s1 else "neg"
    traj_why = "ok" if traj == "pos" else ("db_mismatch" if not ok else "s1_violation")
    if first_err is not None:
        for s in steps:
            if s["idx"] > first_err:
                s["label"] = "U"
    elif not ok:   # 结果不对却没定位到错误步骤：全标 U，留作候选盲区
        for s in steps:
            s["label"] = "U"
    return {"episode_id": ep.get("episode_id"), "task_id": ep["task_id"], "trial": ep.get("trial"), "traj": traj, "traj_why": traj_why,
            "db": ep.get("reward"), "term": ep.get("term"), "first_err": first_err, "unlocated": (not ok) and first_err is None,
            "violations": [{k: v[k] for k in ("rule", "sev", "idx", "executed")} for v in viol], "steps": steps}


def main() -> None:
    from cs_agent.rollout import load_tasks
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", required=True)
    ap.add_argument("--split", default="base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t.id: t for t in load_tasks(a.split)}
    rows = []
    for f in a.eps:
        for l in open(f, encoding="utf-8"):
            try:   # 采样还在写的文件，最后一行可能不完整
                ep = json.loads(l)
            except ValueError:
                continue
            if ep["task_id"] in tasks:
                rows.append(label_episode(ep, tasks[ep["task_id"]]))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    traj = collections.Counter((r["traj"], r["traj_why"]) for r in rows)
    st = collections.Counter(s["label"] for r in rows for s in r["steps"])
    st_kind = collections.Counter((s["kind"], s["label"]) for r in rows for s in r["steps"])
    print(json.dumps({"episodes": len(rows), "traj": {f"{k[0]}/{k[1]}": v for k, v in traj.items()},
                      "neg_unlocated": sum(r["unlocated"] for r in rows), "steps": dict(st),
                      "steps_by_kind": {f"{k[0]}/{k[1]}": v for k, v in sorted(st_kind.items())}}, indent=1))


if __name__ == "__main__":
    main()
