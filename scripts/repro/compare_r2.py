"""R2 报告：重训的两组 RAG vs 原实验。
  python -m scripts.repro.compare_r2 --repro <pol9b_cmp 输出> --orig results/loop2_cont5_ctl_cmp.json
原实验：s2（起点 RAG-S2）、r1_100（对照：奖励仍是 rm1__core_r1）、bf_100（奖励换成补盲后的 rm1__core_r2_bfcl）。
重训：rp_r1、rp_bf 对应后两组。判据（核心结论"提升来自新 critic"）：rp_bf − rp_r1 在含显然错 / 含任意错比例上都 < 0，且 90% 区间不含 0。
"""
from __future__ import annotations

import argparse
import json

M = {"any_clear": "含显然错", "any_err": "含任意错"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", required=True)
    ap.add_argument("--orig", required=True)
    a = ap.parse_args()
    r = json.load(open(a.repro, encoding="utf-8")); o = json.load(open(a.orig, encoding="utf-8"))
    print(f"配对回答：原 {o['answers_paired']}，重训 {r['answers_paired']}")
    print(f"{'':10s}" + "".join(f"{v:>16s}" for v in M.values()) + f"{'句数':>10s}")
    for on, rn, lab in (("s2", "s2", "起点 S2"), ("r1_100", "rp_r1", "对照 r1"), ("bf_100", "rp_bf", "新 critic")):
        cells = "".join(f"{o['arms'][on]['all'][m]:7.1%} → {r['arms'][rn]['all'][m]:5.1%}" for m in M)
        print(f"{lab:10s}{cells}  {o['arms'][on]['all']['sents']:.2f} → {r['arms'][rn]['all']['sents']:.2f}")
    print("\n新 critic − 对照（配对 bootstrap，90% 区间）")
    po = o["pairs"].get("r1_100->bf_100"); pr = r["pairs"]["rp_r1->rp_bf"]
    ok = True
    for m, lab in M.items():
        x = pr[m]; good = x["ci"][1] < 0; ok &= good
        os_ = f"{po[m]['delta'] * 100:+.1f} [{po[m]['ci'][0] * 100:+.1f},{po[m]['ci'][1] * 100:+.1f}]" if po else "—"
        print(f"  {lab}：原 {os_}  重训 {x['delta'] * 100:+.1f} [{x['ci'][0] * 100:+.1f},{x['ci'][1] * 100:+.1f}]  {'成立' if good else '不成立'}")
    print("\nR2_CONCLUSION_HOLDS" if ok else "\nR2_CONCLUSION_DIFFERS")


if __name__ == "__main__":
    main()
