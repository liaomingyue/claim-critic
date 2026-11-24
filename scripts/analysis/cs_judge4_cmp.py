"""第四版 9B 失败问法（根因步 + 出事步 + 引文定位）对 Claude 盲判，和第三版同一批轨迹并排比。
  python -m scripts.analysis.cs_judge4_cmp --arb outputs/rt/cs/claude_arb --j4 outputs/rt/cs/sample_r0/base9_judge4.jsonl outputs/rt/cs/sample_r0/sftB_judge4.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arb", required=True)
    ap.add_argument("--j4", nargs="+", required=True)
    a = ap.parse_args()
    D = Path(a.arb)
    ans = {json.loads(l)["id"]: json.loads(l) for l in open(D / "answers.jsonl", encoding="utf-8")}
    cla = {json.loads(l)["id"]: json.loads(l)["claude"] for l in open(D / "claude.jsonl", encoding="utf-8")}
    j4 = {}
    for f in a.j4:
        pol = Path(f).name.split("_")[0]
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if not r.get("failed"):
                j4[f"{pol}:{r['episode_id']}"] = r
    rows = collections.defaultdict(collections.Counter)
    anch = collections.Counter()
    kinds = collections.Counter()
    for i, c in cla.items():
        x = ans[i]
        if x["stratum"] == "succ9b" or i not in j4 or not c.get("first_error_step"):
            continue
        cs, v3, r = c["first_error_step"], x["judge9b"], j4[i]
        for s in (x["stratum"], "all"):
            R = rows[s]
            R["n"] += 1
            R["v3_exact"] += v3["first_error_step"] == cs
            R["v3_within1"] += abs(v3["first_error_step"] - cs) <= 1
            R["v3_label"] += v3["first_error_step"] == cs and v3["label"] == c.get("label")
            R["v4_root_exact"] += r["first_error_step"] == cs
            R["v4_root_within1"] += abs(r["first_error_step"] - cs) <= 1
            R["v4_root_or_damage"] += cs in (r["first_error_step"], r.get("damage_step"))
            R["v4_label"] += r["first_error_step"] == cs and r["label"] == c.get("label")
        anch[r["root_anchor"]] += 1
        anch["moved_by_anchor"] += r["root_anchor"] != "stated" and r["root_stated"] != r["first_error_step"]
        kinds[r.get("kind")] += 1
    for s, R in rows.items():
        n = max(1, R["n"])
        print(s, json.dumps({k: (v if k == "n" else f"{v} ({v / n:.0%})") for k, v in R.items()}, ensure_ascii=False))
    print("root anchor:", dict(anch), "| kind:", dict(kinds))
    allj = list(j4.values())
    print("all v4 judged:", len(allj), "anchor:", dict(collections.Counter(r["root_anchor"] for r in allj)),
          "moved:", sum(r["root_anchor"] != "stated" and r["root_stated"] != r["first_error_step"] for r in allj),
          "kind:", dict(collections.Counter(r.get("kind") for r in allj)), "label:", dict(collections.Counter(r["label"] for r in allj)))


if __name__ == "__main__":
    main()
