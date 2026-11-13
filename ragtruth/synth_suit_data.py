"""合成适合度数据：给定（原文, 回答, 错误类型），按这个类型去改写，能不能造成这一类的真错。

来源是历次送 Claude 核实的合成样本（outputs/rt/core/gen_v2_label*、scope_label）。每次尝试一条样本：
- requested：要求的类型，标签 = Claude 判为真错且类型与要求一致；
- alt：造成了别的类型的真错，给实际类型记一条正样本（这条回答其实适合那一类）。
没试过的（回答, 类型）不出现在数据里。按原文划分 train / holdout，评测只用 holdout 里的 requested。

  python -m ragtruth.synth_suit_data        # 写 outputs/rt/synth_suit/{train,holdout}.jsonl 和 build_report.json
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path

SETS = ["gen_v2_label", "gen_v2_label_r2", "gen_v2_label_r3", "gen_v2_label_r4", "gen_v2_label_t1", "gen_v2_label_t2", "scope_label"]
MECHS = ["certainty_or_scope_shift", "near_synonym_shift", "contradicts_source", "added_specific_fact", "unsupported_inference",
         "wrong_attribution", "vague_evaluation_or_addon", "number_or_date", "merged_across_passages"]
OUT = Path("outputs/rt/synth_suit")


def load(core: str = "outputs/rt/core") -> list[dict]:
    rows = []
    for d in SETS:
        items = {x["aid"]: x for x in map(json.loads, open(f"{core}/{d}/items.jsonl", encoding="utf-8"))}
        for y in map(json.loads, open(f"{core}/{d}/labels.jsonl", encoding="utf-8")):
            c = y["claude"]
            c = ast.literal_eval(c) if isinstance(c, str) else c
            if y["aid"] in items and isinstance(c, dict):
                rows.append({**items[y["aid"]], "c": c, "set": d})
    return rows


def is_holdout(source_id: str) -> bool:
    return int(hashlib.md5(f"synth_suit|{source_id}".encode()).hexdigest(), 16) % 5 == 0


def main() -> None:
    out = {"train": [], "holdout": []}
    for r in load():
        so = r["sentences_orig"]
        so = ast.literal_eval(so) if isinstance(so, str) else so
        real = r["c"].get("severity") in ("clear", "minor")
        got = r["c"].get("mechanism")
        base = {"source_id": str(r["source_id"]), "resp_id": r["resp_id"], "task_type": r["task_type"], "source": r["source"], "answer": " ".join(s.strip() for s in so),
                "set": r["set"], "group": r["group"], "idx": r["idx"], "severity": r["c"].get("severity"), "produced": got}
        split = "holdout" if is_holdout(base["source_id"]) else "train"
        out[split].append({"id": f"{r['set']}/{r['aid']}", "mech": r["mechanism"], "label": int(real and got == r["mechanism"]), "kind": "requested", **base})
        if real and got != r["mechanism"] and got in MECHS:
            out[split].append({"id": f"{r['set']}/{r['aid']}#alt", "mech": got, "label": 1, "kind": "alt", **base})
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for n, rows in out.items():
        with open(OUT / f"{n}.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        req = [x for x in rows if x["kind"] == "requested"]
        rep[n] = {"rows": len(rows), "requested": len(req), "alt": len(rows) - len(req), "sources": len({x["source_id"] for x in rows}),
                  "pos_rate_requested": round(sum(x["label"] for x in req) / len(req), 4),
                  "by_mech": {m: [sum(x["mech"] == m for x in req), round(sum(x["label"] for x in req if x["mech"] == m) / max(1, sum(x["mech"] == m for x in req)), 3),
                                  sum(x["mech"] == m and x["kind"] == "alt" for x in rows)] for m in MECHS}}
    assert not {x["source_id"] for x in out["train"]} & {x["source_id"] for x in out["holdout"]}
    (OUT / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
