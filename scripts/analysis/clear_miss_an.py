"""critic 漏掉的显然错有什么共同特征。Claude 金标三套验收（去拒答），显然错按 critic 是否 ≥ 0.5 分成漏判 / 抓到，比较各种特征。"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from common.schema import read_jsonl
from ragtruth.critic_cap import critic_file, scores
from ragtruth.critic_cap3 import DIRS

CR = sys.argv[1] if len(sys.argv) > 1 else "rm1__core_r2_bfcl"
rows, ans = [], {}
for d in DIRS:
    for a in read_jsonl(Path(d) / "answers.jsonl"):
        ans[a["id"]] = a
    sc = scores(critic_file(d, CR)) or {}
    for x in read_jsonl(Path(d) / "final.jsonl"):
        if x["id"] in sc and "unable to answer" not in x["sentence"].lower():
            rows.append({**x, "p": sc[x["id"]]})
clear = [x for x in rows if x["is_error"] and x["severity"] == "clear"]
neg = [x for x in rows if not x["is_error"]]
thr = np.quantile([x["p"] for x in neg], 0.9)   # 误报 10% 的阈值
tok = lambda s: re.findall(r"\w+", s.lower())


def feats(x):
    a = ans[x["resp_id"]]
    s, src = x["sentence"], a["source"]
    w = tok(s)
    srcw = tok(src)
    grams = {tuple(srcw[i:i + 4]) for i in range(len(srcw) - 3)}
    g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    copy = sum(g in grams for g in g4) / max(1, len(g4))
    n_sent = len(a["sentences"])
    return {
        "words": len(w),
        "clauses": s.count(",") + s.count(";") + s.count(" and ") + s.count(" while ") + s.count(" which "),
        "digits": bool(re.search(r"\d", s)),
        "quote": '"' in s or "“" in s,
        "copy4": copy,
        "pos_rel": (x["idx"] - 1) / max(1, n_sent - 1),
        "first": x["idx"] == 1,
        "last": x["idx"] == n_sent,
        "n_sent": n_sent,
        "src_words": len(srcw),
        "task": x["task_type"],
        "type": x["type"],
        "mech": x["mechanism"],
    }


F = [(x, feats(x)) for x in clear]
miss = [(x, f) for x, f in F if x["p"] < 0.5]
caught = [(x, f) for x, f in F if x["p"] >= 0.5]
miss10 = [(x, f) for x, f in F if x["p"] < thr]
print(f"critic {CR}: 显然错 {len(clear)}，0.5 阈值漏判 {len(miss)}，误报 10% 阈值（{thr:.3f}）漏判 {len(miss10)}")


def num(name, fmt="{:.2f}"):
    m = np.mean([f[name] for _, f in miss]); c = np.mean([f[name] for _, f in caught])
    mm = np.median([f[name] for _, f in miss]); cm = np.median([f[name] for _, f in caught])
    print(f"  {name:10s} 漏判 均值 {fmt.format(m)} 中位 {fmt.format(mm)} | 抓到 均值 {fmt.format(c)} 中位 {fmt.format(cm)}")


print("\n数值特征（漏判 vs 抓到）")
for k in ("words", "clauses", "copy4", "pos_rel", "n_sent", "src_words"):
    num(k)
print("\n布尔特征（占比）")
for k in ("digits", "quote", "first", "last"):
    print(f"  {k:10s} 漏判 {np.mean([f[k] for _, f in miss]):.0%} | 抓到 {np.mean([f[k] for _, f in caught]):.0%}")


def rate(key, groups=None):
    c = collections.defaultdict(lambda: [0, 0])
    for x, f in F:
        k = f[key]
        if groups:
            k = groups(f[key])
        c[k][0] += x["p"] < 0.5
        c[k][1] += 1
    print(f"\n按 {key} 的漏判率（0.5 阈值）")
    for k, (m, n) in sorted(c.items(), key=lambda kv: -kv[1][1]):
        print(f"  {str(k):28s} {m}/{n} = {m / n:.0%}")


rate("mech")
rate("type")
rate("task")
rate("words", lambda w: "<15" if w < 15 else "15-25" if w < 25 else "25-35" if w < 35 else ">=35")
rate("copy4", lambda c: "<0.3" if c < 0.3 else "0.3-0.6" if c < 0.6 else ">=0.6")
rate("clauses", lambda c: "0-1" if c <= 1 else "2-3" if c <= 3 else ">=4")
rate("pos_rel", lambda p: "开头" if p < 0.34 else "中间" if p < 0.67 else "结尾")
print("\n漏判的显然错（按 critic 分数从低到高，前 25 条）")
for x, f in sorted(miss, key=lambda t: t[0]["p"])[:25]:
    print(f"- p={x['p']:.2f} [{f['task']}/{f['mech']}/{f['type']}] {f['words']}词 copy{f['copy4']:.2f} :: {x['sentence'][:220]}")
    print(f"    证据: {(x.get('evidence') or '')[:160]} | 说明: {(x.get('note') or '')[:200]}")
