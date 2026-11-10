"""开源发布清单（让别人能复现全部结论；L0 + L1 + L2 全部公开，模型和数据放 Hugging Face）。

在 GPU 机（共享盘 /root/autodl-fs）上跑：列出要发布的文件、是否存在、大小，写 release/inventory.json。
  python -m scripts.release_inventory [--fs /root/autodl-fs]
分组：
  data/eval      L1：评测集（三套验收回答 + Claude 金标、RAGTruth++ 校准 / 测试）、各 critic 逐句打分、各版 RAG 的 9B 评测
  data/core      L2：核心集（2B 回答 + Claude 改对 / 标注）及派生数据
  data/synth     L2：错误合成（生成器 v2 的 Claude 标注、sev / RM2 标注、两轮盲区补样本的候选和 Claude 核实）
  data/rm        L2：各打分模型的训练 / 校准 / 测试集（由 data/core、data/synth 派生，打包方便直接训）
  models         L1 / L2：LoRA adapter（只发 LoRA，不发基座）+ 各打分模型的校准温度（report.json）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

C = "code/critic_model/"
CRITICS = ["rm1__core", "rm1__core_r1", "rm1__core_r2", "rm1__core_r2_bfcl", "rm1__core_r2_bfnc", "rm1__core_rd2", "rm1__core_r3",
           "rm1__core_r2_sccl", "rm1__core_r2_scnc", "rm1__core_r2_sccl_no", "rm1__core_sevs1", "rm1__core_sevs2", "rm1__core_sevrnd", "rm1__core_sevcl",
           "rm1__mcgold_ref", "rm1__mcgold_h3ctl", "rm1__mcgold_ref_z1500", "rm1__mcgold_ref_z6000", "rm1__mcgold_hum", "rm1__mcgold_syn"]
RMS = ["sev__v3c0", "sev__v3d", "rm2__v3b", "rm2__v3d", "sat_pair__core", "refuse__core"]
RAG = {"rag-sft-core": "adapters/rag-sft-core/adapter", "RAG-S1": "adapters/mc-grpo-r3_core_coreonly/step200",
       "RAG-S2": "adapters/mc-grpo-r3_core_cont/step100", "RAG-S3-r2": "adapters/mc-grpo-r3_core_cont2/step100",
       "RAG-S3-r3": "adapters/mc-grpo-r3_core_cont3/step100", "RAG-S3-bf": "adapters/mc-grpo-r3_core_cont4/step100",
       "RAG-S3-r1": "adapters/mc-grpo-r3_core_cont5/step100", "RAG-S4-rd2": "adapters/mc-grpo-r3_core_cont6/step100",
       "RAG-S1x": "adapters/mc-grpo-r3_core_main/step200"}
GEN = {"gen-v2-mc": "adapters/gen-v2-mc/adapter", "gen-v2h2-mc": "adapters/gen-v2h2-mc/adapter", "gen-v2h3-mc": "adapters/gen-v2h3-mc/adapter", "gen-g1": C + "outputs/rt/gen_v2_grpo/g1/ckpt/step50"}
EVAL_RAG = ["loop2_coreonly", "loop2_main", "loop2_cont", "loop2_cont2", "loop2_cont3", "loop2_cont4", "loop2_cont5", "loop2_cont6"]


def entries() -> list[tuple[str, str, str]]:
    """(分组, 共享盘上的路径, 说明)"""
    e = []
    for d in ("grpo300", "grpo300b", "grpo300c"):
        e.append(("data/eval", C + f"outputs/rt/minicpm/{d}/answers.jsonl", f"验收集 {d}：回答（原文-种子固定）"))
        e.append(("data/eval", C + f"outputs/rt/minicpm/{d}/final.jsonl", f"验收集 {d}：Claude 金标逐句标注"))
        for c in CRITICS:
            e.append(("data/eval", C + f"outputs/rt/minicpm/{d}/critic_{c}.json", f"critic {c} 在 {d} 上的逐句打分"))
    e += [("data/eval", C + "outputs/rt/rm/rm1/calib.jsonl", "RAGTruth++ 校准集"), ("data/eval", C + "outputs/rt/rm/rm1/test.jsonl", "RAGTruth++ 测试集")]
    for r in EVAL_RAG:
        e.append(("data/eval", C + f"outputs/rt/core/{r}/eval", f"RAG {r}：三套验收原文上的生成 + 9B 金标打分"))
    e.append(("data/eval", C + "outputs/rt/minicpm/rwexp/eval", "奖励粒度 R0–R3：生成 + 9B 打分"))
    for f in ("core.jsonl", "pairs.jsonl", "label_raw.jsonl", "meta.json", "seed_v2_train.jsonl", "seed_v2_holdout.jsonl", "seed_v2_desc_mix.jsonl",
              "genpool_train.jsonl", "genpool_eval.jsonl", "rag_sft_train.jsonl", "rag_sft_holdout.jsonl", "grpo_train.jsonl", "sevpool_labels.jsonl"):
        e.append(("data/core", C + f"outputs/rt/core/{f}", "核心集 / 派生"))
    for f in ("outputs/rt/minicpm/s1/fix_raw.jsonl", "outputs/rt/minicpm/s1/fix_single_raw.jsonl", "outputs/rt/minicpm/grpo/train_ref.jsonl",
              "outputs/rt/ansgrpo/train_all.jsonl", "outputs/rt/minicpm/grpo/dev.jsonl"):
        e.append(("data/core", C + f, "第一、二步：原文池、2B 原回答、Claude 改对"))
    for d in ("gen_v2_label", "gen_v2_label_r2", "gen_v2_label_t1", "gen_v2_label_t2", "hard_check", "blind_check", "sevaug_r1", "scope_label", "gen_audit"):
        e.append(("data/synth", C + f"outputs/rt/core/{d}", "错误合成：Claude 标注 / 核实"))
    for f in ("labitems_desc9b.jsonl", "labitems_t1_desc9b.jsonl", "labitems_t2_desc9b.jsonl", "expl_labels.jsonl"):
        e.append(("data/synth", C + f"outputs/rt/core/{f}", "错误合成：改动描述 / 解释标注"))
    for d in ("blind_fill", "blind_fill_r2", "rag_pool", "rag_pool_r2", "rag_pool_sftcore"):
        e.append(("data/synth", C + f"outputs/rt/gen_v2_grpo/{d}", "两轮盲区补样本：提示池 / 采样 / 选样 / Claude 核实"))
    for f in ("prompts.jsonl", "prompts_w.jsonl", "cap3_src.json") + tuple(f"probe_{m}.jsonl" for m in ("ge", "v2h", "v2h2", "g1s50", "v2h3")) \
            + ("harvest_hard.jsonl", "harvest_hard_v2.jsonl", "harvest_hard_v3.jsonl", "sft_hard.jsonl", "sft_hard_v2.jsonl", "sft_hard_v3.jsonl"):
        e.append(("data/synth", C + f"outputs/rt/gen_v2_grpo/{f}", "生成器 GRPO：提示、试采、难样本"))
    for c in CRITICS + RMS:
        e.append(("data/rm", C + f"outputs/rt/rm/{c}", f"{c}：train / calib / test + eval/report.json（含校准温度）"))
    for c in CRITICS + RMS:
        e.append(("models/critic" if c.startswith("rm1") else "models/rm", f"adapters/{c}-mc/adapter", f"LoRA：{c}"))
    e.append(("models/critic", "adapters/rm1__rm1_gold_9b-9b/adapter", "LoRA：9B 金标 critic（只用于评测）"))
    # 复现脚本（scripts/repro）需要：9B 金标 critic 的校准温度；闭环 GRPO 的参考回答（起点重采，不发的话复现时会重新采，结果多一层随机性）
    e.append(("data/rm", C + "outputs/rt/rm/rm1__rm1_gold_9b/eval/report.json", "9B 金标 critic 的校准温度"))
    for st in ("rag-sft-core", "step200", "step100"):
        e.append(("data/core", C + f"outputs/rt/core/loop2_prompts_{st}", f"闭环 GRPO 原文池 + 起点参考回答（起点 {st}）"))
    # 每个模型要给出具体的训练数据和配置（docs/MODELS.md）。补：9B 金标 critic 的训练集、GRPO 逐步日志
    e.append(("data/rm", C + "outputs/rt/rm/rm1__rm1_gold_9b", "rm1__rm1_gold_9b：train / calib + eval/report.json"))
    for r in EVAL_RAG:
        for f in ("log.jsonl", "quick_eval.jsonl"):
            e.append(("data/eval", C + f"outputs/rt/core/{r}/run/{f}", f"RAG {r}：GRPO 逐步训练日志"))
    for f in ("log.jsonl", "quick_eval.jsonl"):
        e.append(("data/eval", C + f"outputs/rt/gen_v2_grpo/g1/{f}", "生成器 GRPO g1：逐步训练日志"))
    for k, v in RAG.items():
        e.append(("models/rag", v, f"LoRA：{k}"))
    for k, v in GEN.items():
        e.append(("models/gen", v, f"LoRA：错误生成器 {k}"))
    return e


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="/root/autodl-fs")
    ap.add_argument("--out", default="/root/autodl-tmp/release_inventory.json")
    a = ap.parse_args()
    out, tot = [], {}
    for g, rel, desc in entries():
        p = Path(a.fs) / rel
        ok = p.exists()
        sz = size(p) if ok else 0
        out.append({"group": g, "path": rel, "desc": desc, "exists": ok, "bytes": sz})
        t = tot.setdefault(g, [0, 0, 0])
        t[0] += 1; t[1] += ok; t[2] += sz
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, (n, k, sz) in sorted(tot.items()):
        print(f"{g:14s} {k}/{n} 个存在  {sz / 2**20:9.1f} MB")
    print("MISSING:")
    for x in out:
        if not x["exists"]:
            print("  ", x["path"])


if __name__ == "__main__":
    main()
