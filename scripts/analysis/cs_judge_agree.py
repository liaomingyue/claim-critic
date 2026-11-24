"""客服轨迹：9B 特权判断（cs_agent.judge2）和代码标注（cs_agent.label：规则 + 标准动作对照 + 工具报错）的一致性。
  python -m scripts.analysis.cs_judge_agree --dir outputs/rt/cs/sample_r0 [--pols base9 sftB]
失败的轨迹：比第一个错误的位置（同一步 / 9B 更早 / 9B 更晚 / 只有一边定位到）和类型；
成功的轨迹：比违规（9B 的 violations 对规则的 S1/S2 违规，不算 R5 / R2b / R10），按轨迹算重合。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SOFT = {"R5", "R2b", "R10"}


def load_last(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if not r.get("failed"):
            out[r["episode_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pols", nargs="+", default=["base9", "sftB"])
    a = ap.parse_args()
    D = Path(a.dir)
    for p in a.pols:
        J = load_last(D / f"{p}_judge2.jsonl")
        L = {}
        for f in (D / f"{p}_labels.jsonl", D / f"{p}_labels_live.jsonl"):
            if f.exists():
                for l in open(f, encoding="utf-8"):
                    r = json.loads(l)
                    L[r["episode_id"]] = r
                break
        pos = collections.Counter()
        typ = collections.Counter()
        succ = collections.Counter()
        for eid, j in J.items():
            c = L.get(eid)
            if c is None:
                continue
            if j["mode"] == "failed":
                jidx = j["step_idx"][j["first_error_step"] - 1]
                cidx = c["first_err"]
                if cidx is None:
                    pos["code_none"] += 1
                    typ[("9B:" + j["label"], "code:-")] += 1
                    continue
                pos["same" if jidx == cidx else ("9B_earlier" if jidx < cidx else "9B_later")] += 1
                cl = next(s["label"] for s in c["steps"] if s["idx"] == cidx)
                if jidx == cidx:
                    typ[("9B:" + j["label"], "code:" + cl)] += 1
            else:
                jv = {j["step_idx"][v["step"] - 1] for v in j["violations"]}
                cv = {v["idx"] for v in c["violations"] if v["rule"] not in SOFT and v["sev"] in ("S1", "S2")}
                key = ("9B" if jv else "-") + "/" + ("rule" if cv else "-")
                succ[key] += 1
                if jv and cv:
                    succ["same_step_overlap"] += bool(jv & cv)
        n_f = sum(v for k, v in pos.items())
        print(f"== {p}: failed trajectories judged {n_f}")
        for k in ("same", "9B_earlier", "9B_later", "code_none"):
            print(f"   {k:11s} {pos[k]:4d}  ({pos[k] / max(1, n_f):.0%})")
        print("   type at the same step:", sorted(((k, v) for k, v in typ.items() if not k[1].endswith("-")), key=lambda x: -x[1])[:8])
        print("   9B type where code found nothing:", sorted(((k[0], v) for k, v in typ.items() if k[1].endswith("-")), key=lambda x: -x[1]))
        print(f"   success trajectories judged {sum(v for k, v in succ.items() if '/' in k)}: {dict(succ)}")


if __name__ == "__main__":
    main()
