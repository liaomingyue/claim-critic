"""2B critic 漏的明显错，是找不到 / 对不上证据，还是证据摆在眼前也不认为是错（2026-11-04，用户：核心是确认 2B 为什么识别不出来）。
验收回答（outputs/rt/minicpm/grpo300all/final.jsonl，Claude 标，带 evidence = 原文里相关的那段）每句做两份 prompt：
  |full：原样（完整原文）；|ev：原文只换成 Claude 摘的证据段。拒答句和没有证据的句子跳过。
2B / 9B critic 都打两份：证据段上 2B 判得出 → 问题在长原文里定位 / 对齐；仍判不出 → 它真的不认为是错（理解 / 口径）。
非错句同样两份，用来看只给证据会不会把所有句子都抬高。

  python -m ragtruth.evidence_probe --out outputs/rt/minicpm/grpo300all/probe_ev.jsonl
"""
from __future__ import annotations

import argparse
import json

from ragtruth.critic_refuse import REF_RE
from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.task import render_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/rt/minicpm/grpo300all/final.jsonl")
    ap.add_argument("--answers", nargs="*", default=["outputs/rt/minicpm/grpo300/answers.jsonl", "outputs/rt/minicpm/grpo300b/answers.jsonl"])
    ap.add_argument("--out", default="outputs/rt/minicpm/grpo300all/probe_ev.jsonl")
    a = ap.parse_args()
    ans = {x["id"]: x for f in a.answers for x in map(json.loads, open(f, encoding="utf-8"))}
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in map(json.loads, open(a.labels, encoding="utf-8")):
            ev = (r.get("evidence") or "").strip()
            if not ev or ev.lower() == "none" or REF_RE.search(r["sentence"]):
                continue
            x = ans[r["resp_id"]]
            for tag, src in (("full", x["source"]), ("ev", ev)):
                p = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": src, "sentences": x["sentences"], "idx": r["idx"]})}]
                f.write(json.dumps({"id": f"{r['id']}|{tag}", "prompt": p, "is_error": r["is_error"], "severity": r["severity"], "mechanism": r["mechanism"]}, ensure_ascii=False) + "\n")
                n += 1
    print("PROBE", a.out, n, flush=True)


if __name__ == "__main__":
    main()
