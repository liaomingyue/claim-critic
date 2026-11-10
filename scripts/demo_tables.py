"""L0 演示：不需要 GPU，从 results/ 里发布的评测结果重出技术分享详细版（docs/tech_share_details.md）里的表 1–9。

  pip install -r requirements.txt
  python -m scripts.demo_tables            # 打印全部表（Markdown）
  python -m scripts.demo_tables --only 2   # 只打印第 2 张

results/ 里每个文件的来源：
  rwexp_cmp.json      ragtruth.pol9b_cmp：R0–R3 奖励对照，各 50 步，验收 a / b 两套（200 条配对回答），9B 金标 critic 逐句打分
  loop2_cmp.json      ragtruth.pol9b_cmp：分段 GRPO（SFT → 第一段 → cont → cont2 / cont3），三套验收 595 条配对回答
  critic_cap_*.json   ragtruth.critic_cap3：critic 在 Claude 金标三套验收上（去拒答 2061 句 / 显然错 179）的 AUC、召回，按原文 bootstrap
  gen_probe.json      ragtruth.gen_v2_grpo probe：错误生成器各版本在 64 个留出提示 × 32 上的试采漏斗
  blind_check.json    ragtruth.gen_v2_label：盲区候选每类 10 条的 Claude 核实
  critic_cap_blindfill*.json  ragtruth.critic_cap3：多类型一起补盲，核实组 / 不核实组（相对 r2，以及两组之差）
  loop2_cont4_cmp.json / loop2_cont5_ctl_cmp.json  ragtruth.pol9b_cmp：第一轮闭环（RAG-S3-bf）、对照（奖励模型不换的 RAG-S3-r1）
  loop2_cont6_cmp.json / critic_cap_rd2.json  第二轮闭环（RAG-S4-rd2）和第二轮 critic 的静态评测
有原始数据包（scripts.pack_data）和打分文件时，可用上面的模块从头重算这些 JSON。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def ci(x: dict, k: float = 1.0, nd: int = 3) -> str:
    return f"{x['delta'] * k:+.{nd}f} [{x['ci'][0] * k:+.{nd}f}, {x['ci'][1] * k:+.{nd}f}]"


def t1() -> str:
    d = load("rwexp_cmp.json")
    names = {"v0": "训练前", "R0": "R0 整段一个分（旧公式）", "R1": "R1 整段：去底噪 + 按错句计数", "R2": "R2 + 句子层优势", "R3": "R3 + 往前传一句（γ 0.3）"}
    out = [f"### 表 1 RAG 奖励粒度（R0–R3，各 50 步；{d['answers_paired']} 条配对回答 / {d['sources']} 原文；9B 金标 p ≥ 0.9 为显然错）", "",
           "| 奖励 | 含显然错 | 显然错句 / 条 | 拒答 | 显然错句数 vs R0 [90%] |", "|---|---|---|---|---|"]
    for k, n in names.items():
        a = d["arms"][k]["all"]
        diff = ci(d["vs_R0"][k]["clear"]) if k in d["vs_R0"] else "—"
        out.append(f"| {n} | {a['any_clear']:.1%} | {a['clear']:.3f} | {a['refuse']:.1%} | {diff} |")
    return "\n".join(out)


def t2() -> str:
    d = load("loop2_cmp.json")
    rows = [("sft", "RAG SFT"), ("core200", "第一段：critic rm1__core，200 步"), ("main200", "一次训完：SFT 起点，新 critic r1，200 步"),
            ("cont100", "第二段：第一段起点，新 critic r1，100 步"), ("cont2_100", "第三段：cont 起点，critic r2，100 步"),
            ("cont3_100", "第三段：cont 起点，critic r3（长句增强），100 步")]
    out = [f"### 表 2 分段 GRPO（{d['answers_paired']} 条配对回答 / {d['sources']} 原文）", "",
           "| 模型 | 含显然错 | 显然错句 / 条 | 含任意错 | 拒答 |", "|---|---|---|---|---|"]
    for k, n in rows:
        a = d["arms"][k]["all"]
        out.append(f"| {n} | {a['any_clear']:.1%} | {a['clear']:.3f} | {a['any_err']:.1%} | {a['refuse']:.1%} |")
    p = d["pairs"]
    out += ["", f"- 分段 vs 一次训完（cont100 − main200）：含显然错 {ci(p['main200->cont100']['any_clear'], 100, 1)} 个点，显然错句 {ci(p['main200->cont100']['clear'])}",
            f"- 一次训完第 100 → 200 步：显然错句 {ci(p['main100->main200']['clear'])}（已基本不降）",
            f"- cont → cont2（r2）：显然错句 {ci(p['cont100->cont2_100']['clear'])}；cont → cont3（r3）：{ci(p['cont100->cont3_100']['clear'])}"]
    return "\n".join(out)


def t3() -> str:
    main, hs, sev = load("critic_cap_main.json"), load("critic_cap_humsyn.json"), load("critic_cap_sev.json")
    out = ["### 表 3 critic 加什么数据有用（Claude 金标三套验收，去拒答，显然错 vs 没错）", "",
           "| 底座 | critic | 加了什么 | 显然错 AUC | vs 底座 [90%] | 误报 10% 召回 |", "|---|---|---|---|---|---|"]
    rows = [(main, "ref", "rm1__mcgold_ref", "—（底座 6330 条）"),
            (main, "ref", "rm1__mcgold_h3ctl", "生成器样本 554 对（sev 门槛 + 自然度）"),
            (main, "ref", "rm1__mcgold_ref_z1500", "生成器样本 1500 对（不过滤自然度）"),
            (main, "ref", "rm1__mcgold_ref_z6000", "生成器样本 6000 对"),
            (hs, "ref", "rm1__mcgold_hum", "只用人工标注（RAGTruth 原版，3030 条）"),
            (hs, "ref", "rm1__mcgold_syn", "只用合成（3030 条）"),
            (sev, "core", "rm1__core", "—（核心集底座 4604 条）"),
            (sev, "core", "rm1__core_sevrnd", "9B 生成器样本，随机 733"),
            (sev, "core", "rm1__core_sevcl", "9B 生成器样本，Claude 判为真错的 811"),
            (sev, "core", "rm1__core_r2", "critic 拿不准（P < 0.7）+ Claude 核实为真错的 581")]
    for d, base, k, what in rows:
        v = d.get(k) or d.get("clear", {}).get(k)
        if not v:
            continue
        diff = "—" if v.get("d", 0) == 0 and v.get("dlo", 0) == 0 else f"{v['d']:+.3f} [{v['dlo']:+.3f}, {v['dhi']:+.3f}]"
        out.append(f"| {base} | {k} | {what} | {v['auc']:.3f} | {diff} | {v['r10']:.2f} |")
    return "\n".join(out)


def t4() -> str:
    sc, no = load("critic_cap_scope.json"), load("critic_cap_scope_noorig.json")
    views = [("clear", "显然错"), ("any", "任意错（含细微）"), ("type:certainty_or_scope_shift", "范围偏移"), ("type:contradicts_source", "直接矛盾"),
             ("type:wrong_attribution", "张冠李戴"), ("type:near_synonym_shift", "近义偏移")]
    arms = [(sc, "rm1__core_r2_sccl", "+ 范围偏移 285 条，Claude 核实，配原句"), (sc, "rm1__core_r2_scnc", "+ 范围偏移 285 条，不核实，配原句"),
            (no, "rm1__core_r2_sccl_no", "+ 范围偏移 285 条，Claude 核实，不配原句")]
    out = ["### 表 4 定向补一个类型（范围偏移）：各口径 AUC 相对 r2 的变化 [90%]", "",
           "| critic | " + " | ".join(n for _, n in views) + " |", "|---|" + "---|" * len(views)]
    out.append("| r2（AUC） | " + " | ".join(f"{sc[v]['rm1__core_r2']['auc']:.3f}" for v, _ in views) + " |")
    for d, k, what in arms:
        out.append(f"| {what} | " + " | ".join(f"{d[v][k]['d']:+.3f} [{d[v][k]['dlo']:+.3f}, {d[v][k]['dhi']:+.3f}]" if k in d.get(v, {}) else "—" for v, _ in views) + " |")
    return "\n".join(out)


def t5() -> str:
    d = load("gen_probe.json")
    names = {"ge": "SFT 起点", "v2h": "v2h（难样本 SFT）", "v2h2": "v2h2（第二轮）", "g1s50": "g1（v2h2 + GRPO 50 步）", "v2h3": "v2h3（大提示池难样本 SFT）"}
    out = ["### 表 5 错误生成器迭代（64 个留出提示 × 32；门槛 = sev P(有错) ≥ 0.9 且 RM2 自然 ≥ 0.5）", "",
           "| 生成器 | 过门槛 | 过门槛里 critic < 0.5 | ≥ 0.9 | 过门槛且 critic 0.1–0.5 | 其中 sev 判要求类型 | 没有这类样本的组 | 盲区自然样本（critic 0.1–0.5）| 其中 sev ≥ 0.5 / ≥ 0.9 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, n in names.items():
        x = d[k]
        g = max(x["gated"], 1)
        out.append(f"| {n} | {x['gated'] / x['samples']:.1%} | {x['gated_lt05'] / g:.1%} | {x['gated_ge09'] / g:.1%} | {x['gated_01_05']} | {x['gated_01_05_req']} | "
                   f"{x['groups_zero_01_05']} / {x['groups']} | {x['band_nat']} | {x['band_nat_sev05']} / {x['band_nat_sev09']} |")
    return "\n".join(out)


def t6() -> str:
    rows = load("blind_check.json")["rows"]
    zh = {"certainty_or_scope_shift": "范围偏移", "near_synonym_shift": "近义偏移", "wrong_attribution": "张冠李戴", "contradicts_source": "直接矛盾",
          "merged_across_passages": "跨段拼接", "unsupported_inference": "无依据推断", "number_or_date": "数字日期",
          "vague_evaluation_or_addon": "附加评价", "added_specific_fact": "添加事实"}
    real = lambda r: r["severity"] in ("clear", "minor")
    out = [f"### 表 6 盲区候选的 Claude 核实（critic 0.1–0.5、自然、sev ≥ 0.5；每类 10 条，共 {len(rows)} 条）", "",
           "| 要求类型 | n | 真错 | 显然 | 类型一致 | 真错 + 自然 |", "|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["mechanism"]].append(r)
    for m, rs in sorted(by.items(), key=lambda kv: -sum(map(real, kv[1])) / len(kv[1])):
        out.append(f"| {zh.get(m, m)} | {len(rs)} | {sum(map(real, rs))} | {sum(r['severity'] == 'clear' for r in rs)} | {sum(r['matches'] for r in rs)} | "
                   f"{sum(real(r) and r['natural'] == 'yes' for r in rs)} |")
    out += ["", "| sev P(有错) | n | 真错率 |", "|---|---|---|"]
    for lo, hi in ((0.5, 0.8), (0.8, 0.9), (0.9, 1.01)):
        rs = [r for r in rows if lo <= r["p_err"] < hi]
        out.append(f"| {lo}–{min(hi, 1.0)} | {len(rs)} | {sum(map(real, rs)) / max(len(rs), 1):.0%} |")
    return "\n".join(out)


def _views():
    return [("clear", "显然错"), ("any", "任意错"), ("type:certainty_or_scope_shift", "范围偏移"), ("type:near_synonym_shift", "近义偏移"),
            ("type:wrong_attribution", "张冠李戴"), ("type:merged_across_passages", "跨段拼接"), ("type:contradicts_source", "直接矛盾")]


def _d(x):
    return f"{x['d']:+.3f} [{x['dlo']:+.3f}, {x['dhi']:+.3f}]"


def t7() -> str:
    d, cn = load("critic_cap_blindfill.json"), load("critic_cap_blindfill_clnc.json")
    out = ["### 表 7 多类型一起补，要不要人工核实（AUC 相对 r2 的变化 [90%]）", "",
           "| 口径 | r2 的 AUC | 核实组 cl | 不核实组 nc | cl − nc |", "|---|---|---|---|---|"]
    for v, n in _views():
        if v not in d:
            continue
        r = d[v]
        out.append(f"| {n} | {r['rm1__core_r2']['auc']:.3f} | {_d(r['rm1__core_r2_bfcl'])} | {_d(r['rm1__core_r2_bfnc'])} | {_d(cn[v]['rm1__core_r2_bfcl'])} |")
    return "\n".join(out)


def t8() -> str:
    d, c = load("loop2_cont4_cmp.json"), load("loop2_cont5_ctl_cmp.json")
    rows = [("sft", "RAG-SFT"), ("core200", "RAG-S1 第 1 段强化"), ("cont100", "RAG-S2 第 2 段强化"), ("cont2_100", "RAG-S3-r2（critic r2）"),
            ("cont3_100", "RAG-S3-r3（critic r3）"), ("cont4_100", "RAG-S3-bf（补过盲区的 critic，第一轮闭环）")]
    out = [f"### 表 8 闭环转了一圈：RAG 继续变好（{d['answers_paired']} 条配对回答 / {d['sources']} 原文）", "",
           "| RAG 模型 | 含显然错 | 显然错句 / 条 | 含任意错 | 拒答 | 句数 |", "|---|---|---|---|---|---|"]
    for k, n in rows:
        a = d["arms"][k]["all"]
        out.append(f"| {n} | {a['any_clear']:.1%} | {a['clear']:.3f} | {a['any_err']:.1%} | {a['refuse']:.1%} | {a['sents']:.2f} |")
    a = c["arms"]["r1_100"]["all"]
    out.append(f"| 对照 RAG-S3-r1（从 S2 出发、奖励模型不换、同样 100 步） | {a['any_clear']:.1%} | {a['clear']:.3f} | {a['any_err']:.1%} | {a['refuse']:.1%} | {a['sents']:.2f} |")
    p, q = d["pairs"]["cont4_v0->cont4_100"], c["pairs"]["r1_100->bf_100"]
    out += ["", f"- S3-bf 对自己的起点：含显然错 {ci(p['any_clear'], 100, 1)} 个点，显然错句 {ci(p['clear'])}，含任意错 {ci(p['any_err'], 100, 1)} 个点",
            f"- S3-bf 对同样多训 100 步、奖励模型不换的 S3-r1：含显然错 {ci(q['any_clear'], 100, 1)} 个点，含任意错 {ci(q['any_err'], 100, 1)} 个点"]
    return "\n".join(out)


def t9() -> str:
    d, cr = load("loop2_cont6_cmp.json"), load("critic_cap_rd2.json")
    rows = [("s2", "RAG-S2"), ("s3bf", "RAG-S3-bf（第一轮）"), ("s4_v0", "S4 起点（S3-bf 重新生成）"), ("s4_50", "S4 第 50 步"), ("s4_100", "RAG-S4-rd2 第 100 步（第二轮）")]
    out = [f"### 表 9 第二轮闭环：收益递减（{d['answers_paired']} 条配对回答）", "",
           "| RAG 模型 | 含显然错 | 显然错句 / 条 | 含任意错 | 句数 |", "|---|---|---|---|---|"]
    for k, n in rows:
        a = d["arms"][k]["all"]
        out.append(f"| {n} | {a['any_clear']:.1%} | {a['clear']:.3f} | {a['any_err']:.1%} | {a['sents']:.2f} |")
    p = d["pairs"]["s4_v0->s4_100"]
    out += ["", f"- S4 对自己的起点：含显然错 {ci(p['any_clear'], 100, 1)} 个点，含任意错 {ci(p['any_err'], 100, 1)} 个点",
            "", "第二轮 critic（rd2）相对第一轮（bfcl）的 AUC 变化：", "", "| 口径 | bfcl | rd2 | 变化 [90%] |", "|---|---|---|---|"]
    for v, n in _views():
        if v in cr:
            out.append(f"| {n} | {cr[v]['rm1__core_r2_bfcl']['auc']:.3f} | {cr[v]['rm1__core_rd2']['auc']:.3f} | {_d(cr[v]['rm1__core_rd2'])} |")
    return "\n".join(out)


TABLES = [t1, t2, t3, t4, t5, t6, t7, t8, t9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="只打印第几张（1 起）")
    a = ap.parse_args()
    for k, f in enumerate(TABLES, 1):
        if not a.only or a.only == k:
            print(f(), end="\n\n")


if __name__ == "__main__":
    main()
