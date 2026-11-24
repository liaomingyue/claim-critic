"""客服轨迹的步骤级标签数据集（步骤模型 / 过程奖励的训练数据）。

标签体系：
  数据库对不上（失败）的轨迹：
    第一个出错的位置标出来（FIRST_ERR）；它之前的步骤正常（OK）。
    之后的步骤，除了纠错动作，全算错（AFTER_ERR）——在错的基础上继续往下走。纠错动作按效果给分：
      FIX       纠正确了：所有动作里得分最高；
      HANDOFF   出错后转人工（9B 判为在补救，或之前已经试过纠错）：加分；meta.handoff 里记之前有没有试过纠错；
      TRY       试图纠正但没成功：至少不扣分（略加分）；
      LOOP      同样的错误做法反复试：不扣分也不加分。
  数据库对得上（成功）的轨迹：结果分之外还有过程分——违规 / 答错漏答 / 靠用户指出才改对的步骤扣分；
    侥幸做对的那一步不给正分；自己发现并纠正的步骤（FIX / TRY）照样加分。
    靠用户模拟器放水做成的轨迹不进数据集。
  「原本是正确的、非要去改、而且改了」（OVERFIX）要大扣分：失败轨迹里它就是第一个错误（只是分更低，判断 --prompt ofx 补问），
    成功轨迹里（改了又改回来、或被工具拒了）也扣同样的分（判断 --prompt v6 的第 7 项）。

来源：
  失败轨迹的第一个错误：9B 特权判断第三版（cs_agent.judge2 默认问法，和 Claude 盲判最一致）；
    代码标注（规则 + 标准动作对照）或第五版（引文定位）指到同一步的记 conf=high，否则 conf=low。
    之后步骤的 FIX / TRY / LOOP / IGNORE 来自第三版的 after；TRY 或 LOOP 之后调用了 transfer_to_human_agents 的那一步记 HANDOFF。
  成功轨迹：优先用第六版（全部成功轨迹，含 overfix），没有时用第三版（按哈希抽的一部分），conf=judge；都没有的只用代码规则，conf=code。

  python -m cs_agent.step_dataset --dir outputs/rt/cs/sample_r0 --out outputs/rt/cs/sample_r0/steps_v1.jsonl [--pols base9 sftB]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SCORE = {"OK": 1.0, "FIRST_ERR": -2.0, "OVERFIX": -3.0, "AFTER_ERR": -1.0,
         "FIX": 3.0, "HANDOFF": 1.5, "TRY": 0.5, "LOOP": 0.0,
         "VIOLATION": -1.0, "VIOLATION_S1": -2.0, "INFO_ERR": -1.0, "USER_CORRECTED": -1.0, "LUCKY": 0.0}
SOFT = {"R5", "R2b", "R10"}
TRANSFER = "transfer_to_human_agents"


def _load(path: Path, key="episode_id", keep=lambda r: True) -> dict:
    out = {}
    if not path.exists():
        return out
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if keep(r):
            out[r[key]] = r
    return out


def agent_steps(ep: dict) -> list[int]:
    """和 cs_agent.judge.render 一样的步骤编号：开场问候不算，其余每条客服消息一步。"""
    return [i for i, m in enumerate(ep["messages"]) if m.get("role") == "assistant" and not (i == 0 and not m.get("tool_calls"))]


def build_failed(ep, idxs, j3, code, j5, ofx=None):
    n = len(idxs)
    fe = j3["first_error_step"]
    code_step = idxs.index(code["first_err"]) + 1 if code and code.get("first_err") in idxs else None
    conf = "high" if fe in (code_step, (j5 or {}).get("first_error_step")) else "low"
    after = {x["step"]: x["label"] for x in j3.get("after") or []}
    labels = {}
    tried, handoff = False, []
    for s in range(1, n + 1):
        m = ep["messages"][idxs[s - 1]]
        calls = [c["name"] for c in m.get("tool_calls") or []]
        if s < fe:
            lab = "OK"
        elif s == fe:
            lab = "OVERFIX" if (ofx or {}).get("overfix") else "FIRST_ERR"
        else:
            a = after.get(s)
            if TRANSFER in calls and (a in ("FIX", "TRY") or tried):   # 出错后转人工：9B 判成在补救，或之前已经试过纠错 → HANDOFF
                lab = "HANDOFF"
                handoff.append({"step": s, "after_attempt": tried})
            elif a in ("FIX", "TRY", "LOOP"):
                lab = a
                tried = True
            else:
                lab = "AFTER_ERR"
        labels[s] = lab
    return labels, {"first_error_step": fe, "first_error_type": j3.get("label"), "code_step": code_step,
                    "v5_step": (j5 or {}).get("first_error_step"), "v5_damage_step": (j5 or {}).get("damage_step"), "conf": conf,
                    "overfix": (ofx or {}).get("overfix"), "handoff": handoff, "overfix_reason": (ofx or {}).get("overfix_reason", "")}


def build_success(ep, idxs, js, code):
    n = len(idxs)
    labels = {s: "OK" for s in range(1, n + 1)}
    if js:
        for x in js.get("violations") or []:
            labels[x["step"]] = "VIOLATION"
        for x in js.get("info_errors") or []:
            labels[x["step"]] = "INFO_ERR"
        for x in js.get("user_corrected") or []:
            labels[x["step"]] = "USER_CORRECTED"
        for x in js.get("recoveries") or []:
            labels[x["step"]] = x["label"]
        for x in js.get("overfix") or []:
            labels[x["step"]] = "OVERFIX"
        conf = "judge"
    else:
        conf = "code"
    for v in (code or {}).get("violations") or []:   # 代码规则的严重违规两种来源都用（S1 再加重）
        if v["rule"] in SOFT or v["sev"] not in ("S1", "S2") or v["idx"] not in idxs:
            continue
        s = idxs.index(v["idx"]) + 1
        labels[s] = "VIOLATION_S1" if v["sev"] == "S1" else (labels[s] if labels[s] != "OK" else "VIOLATION")
    meta = {"conf": conf, "judge_pv": (js or {}).get("pv"), "lucky": bool(js and js.get("lucky")), "lucky_reason": (js or {}).get("lucky_reason", "")}
    return labels, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pols", nargs="+", default=["base9", "sftB"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    rows, st = [], collections.Counter()
    for p in a.pols:
        eps = _load(D / f"{p}_train.jsonl")
        code = _load(D / f"{p}_labels.jsonl") or _load(D / f"{p}_labels_live.jsonl")
        j3 = _load(D / f"{p}_judge2.jsonl", keep=lambda r: not r.get("failed") and r.get("pv", 0) >= 3)
        j5 = _load(D / f"{p}_judge5.jsonl", keep=lambda r: not r.get("failed"))
        j6 = _load(D / f"{p}_judge6.jsonl", keep=lambda r: not r.get("failed") and r.get("mode") == "success")
        jo = _load(D / f"{p}_judgeofx.jsonl", keep=lambda r: not r.get("failed"))
        for eid, ep in eps.items():
            idxs = agent_steps(ep)
            ok = ep.get("reward") == 1.0
            if ok:
                js = j6.get(eid) or (j3.get(eid) if (j3.get(eid) or {}).get("mode") == "success" else None)
                if js and js.get("sim_leak"):
                    st["skip_sim_leak"] += 1
                    continue
                labels, meta = build_success(ep, idxs, js, code.get(eid))
            else:
                j = j3.get(eid)
                if not j or j.get("mode") != "failed" or j.get("step_idx") != idxs:
                    st["skip_failed_unjudged"] += 1
                    continue
                labels, meta = build_failed(ep, idxs, j, code.get(eid), j5.get(eid), jo.get(eid))
            st[f"traj_{'ok' if ok else 'fail'}_{meta['conf']}"] += 1
            steps = [{"step": s, "idx": idxs[s - 1], "kind": "call" if ep["messages"][idxs[s - 1]].get("tool_calls") else "text",
                      "label": labels[s], "score": SCORE[labels[s]]} for s in range(1, len(idxs) + 1)]
            for x in steps:
                st["step_" + x["label"]] += 1
            rows.append({"episode_id": eid, "policy": p, "task_id": ep["task_id"], "db_ok": ok, "meta": meta, "steps": steps,
                         "traj_score": sum(x["score"] for x in steps) / max(1, len(steps))})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"trajectories": len(rows), **dict(sorted(st.items()))}, indent=1))


if __name__ == "__main__":
    main()
