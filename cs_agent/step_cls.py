"""步骤级标签（cs_agent.step_dataset 的输出）→ 步骤模型的分类训练数据：合并类别、抽样正常步骤、按任务划分训练 / 测试。

类别（细标签保留在 fine 字段，合成数据补够以后可以再拆开）：
  OK         正常
  ERR        出错：失败轨迹的第一个错误（含 OVERFIX）、成功轨迹里的违规 / 答错漏答 / 靠用户指出才改对的步骤
  AFTER_ERR  在错的基础上继续走（出错后、不是纠错的动作）
  FIX        纠正确了
  TRY        纠错的尝试：没成功（TRY）、反复试（LOOP）、纠不成转人工（HANDOFF）
奖励 = Σ p(类别) × 分值（REWARD，可以改，不用重训）。

模型一次前向读整条轨迹、在每一步结尾读出分类；训练时只有 train_mask=1 的步骤算损失：
  非 OK 的步骤全部算；OK 步骤在训练集里只按固定种子抽 n_ok 个（其余 OK 步骤不算损失，但仍在上下文里）。
测试集 test：按 task_id 的哈希留出约 test_frac 的任务（同一个任务的所有轨迹都在测试集，和训练集不共享任务）；
另把 Claude 盲判过、但任务在训练集里的轨迹单独放 claude（不训练，只评测，和 Claude 的判断对照用）。两个评测集的步骤全部算。

  python -m cs_agent.step_cls --steps outputs/rt/cs/sample_r0/steps_v1.jsonl --out outputs/rt/cs/sample_r0/steps_cls_v1.jsonl \
      [--n_ok 4000] [--test_frac 0.15] [--holdout outputs/rt/cs/claude_arb/bundle.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

MERGE = {"OK": "OK", "FIRST_ERR": "ERR", "OVERFIX": "ERR", "VIOLATION": "ERR", "VIOLATION_S1": "ERR", "INFO_ERR": "ERR",
         "USER_CORRECTED": "ERR", "AFTER_ERR": "AFTER_ERR", "FIX": "FIX", "TRY": "TRY", "LOOP": "TRY", "HANDOFF": "TRY",
         "LUCKY": "OK"}
CLASSES = ["OK", "ERR", "AFTER_ERR", "FIX", "TRY"]
REWARD = {"OK": 1.0, "ERR": -2.0, "AFTER_ERR": -1.0, "FIX": 3.0, "TRY": 0.5}


def _h(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_ok", type=int, default=4000)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--holdout", default="outputs/rt/cs/claude_arb/bundle.jsonl", help="这些轨迹（id = 策略:episode_id）一律进测试集")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for l in open(a.steps, encoding="utf-8"):
        l = l.strip()
        if l:
            rows.append(json.loads(l))
    hold = set()
    if a.holdout and Path(a.holdout).exists():
        hold = {json.loads(l)["id"] for l in open(a.holdout, encoding="utf-8") if l.strip()}
    for r in rows:
        r["split"] = "test" if _h("task:" + str(r["task_id"])) < a.test_frac else ("claude" if f"{r['policy']}:{r['episode_id']}" in hold else "train")
        for s in r["steps"]:
            s["fine"] = s["label"]
            s["cls"] = MERGE[s["label"]]
            s["train_mask"] = 1 if (r["split"] != "train" or s["cls"] != "OK") else 0
    ok_train = sorted(((r["policy"], r["episode_id"], s["step"]) for r in rows if r["split"] == "train" for s in r["steps"] if s["cls"] == "OK"))
    random.Random(a.seed).shuffle(ok_train)
    pick = set(ok_train[:a.n_ok])
    for r in rows:
        if r["split"] == "train":
            for s in r["steps"]:
                if s["cls"] == "OK" and (r["policy"], r["episode_id"], s["step"]) in pick:
                    s["train_mask"] = 1
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = {}
    for sp in ("train", "test", "claude"):
        rs = [r for r in rows if r["split"] == sp]
        c = collections.Counter(s["cls"] for r in rs for s in r["steps"] if s["train_mask"])
        fine = collections.Counter(s["fine"] for r in rs for s in r["steps"] if s["train_mask"] and s["cls"] != s["fine"])
        st[sp] = {"trajectories": len(rs), "tasks": len({r["task_id"] for r in rs}), "fail": sum(not r["db_ok"] for r in rs),
                  "steps": {k: c[k] for k in CLASSES}, "merged_from": dict(fine)}
    print(json.dumps({"classes": CLASSES, "reward": REWARD, **st}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
