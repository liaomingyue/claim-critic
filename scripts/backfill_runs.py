"""补录 manifest 组件上线（2026-10-27 傍晚）之前的实验：按当时的配置、数据、训练日志和评测结果，
写成与 common.manifest 同格式的记录，合并进 outputs/runs/registry_all.jsonl（带 "backfilled": true）。

能补的：配置（当前 yaml；改动过的注明）、数据指纹（本地数据文件；合成组的训练集按固定种子从合成池重建，
已与服务器上的原文件逐行核对）、训练步数 / 耗时 / 末尾 loss（训练日志）、评测指标和测试集指纹、训练机器和环境版本。
补不了的：代码指纹（当时没有记录，标为 unknown）、精确的起止时间（用日志或文件时间近似）。

用法：python -m scripts.backfill_runs  →  再跑 python -m scripts.collect_runs 生成 docs/runs.md
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from common.config import load_config
from common.manifest import data_fingerprint

TEST = "outputs/external/ragtruth/test.jsonl"
ENV_5090 = {"gpu": "NVIDIA GeForce RTX 5090", "python": "3.12.3", "torch": "2.13.0", "transformers": "5.19.0",
            "trl": "1.16.0", "peft": "0.21.2", "vllm": "0.32.0", "source": "AutoDL 镜像 + pip 记录（会话笔记）"}
ENV_LOCAL = {"gpu": "NVIDIA GeForce RTX 4070 SUPER", "python": "3.10.11", "torch": "2.11.0+cu128",
             "transformers": "5.20.2", "trl": "1.14.2", "peft": "0.21.2", "vllm": "WSL Docker 容器（版本未记录）",
             "source": "本机 venv（会话笔记）"}
R = "outputs/autodl/remote_logs"
REBUILT = "outputs/rt/claims/rebuilt"

# (评测名, 配置, 训练数据, 训练机器, 环境, 训练日志, 备注)
RUNS = [
    ("sft-0.6b", "configs/rt/sft_0.6b.yaml", "outputs/external/ragtruth/train_sub.jsonl", "local", ENV_LOCAL, None,
     "① 回答级格式：一次判整条回答、列出错误句；1,410 条回答。configs/rt/sft_0.6b.yaml 之后改成了逐句 v1，"
     "这次训练时的 task 为回答级，其余超参同该文件（ckpt 的 sft_meta.json：n_samples=1410）"),
    ("sft-0.6b-claim", "configs/rt/sft_0.6b.yaml", "outputs/rt/claims/train_sub.jsonl", "local", ENV_LOCAL, None,
     "逐句 v1：9B-Q4 补写证据和理由，1:2 下采样（sft_meta n_samples=3003）"),
    ("sft-0.6b-v2-notips", "configs/rt/sft_0.6b_v2_notips.yaml", "outputs/rt/claims/train_sub35_r3.jsonl", "local",
     ENV_LOCAL, None, "逐句 v2：35B 补写 + 一致性校验，1:3"),
    ("sft-9b-claim", "configs/rt/sft_9b.yaml", "outputs/rt/claims/train_sub.jsonl", "611b489072（已释放）", ENV_5090,
     f"{R}/sft-9b.log", "9B v1"),
    ("sft-9b-v2-notips", "configs/rt/sft_9b_v2.yaml", "outputs/rt/claims/train_sub35_r3.jsonl", "611b489072（已释放）",
     ENV_5090, f"{R}/sft-9b-v2.log", "9B v2"),
    ("sft-0.6b-e2", "configs/rt/exp/sft_0.6b_e2.yaml", "outputs/rt/claims/train_sub35_nat.jsonl",
     "65ee479064（已释放）", ENV_5090, f"{R}/sft-0.6b-e2.log", "② 逐句只判定，自然比例"),
    ("sft-0.6b-e3", "configs/rt/exp/sft_0.6b_e3.yaml", "outputs/rt/claims/train_sub35_nat.jsonl",
     "65ee479064（已释放）", ENV_5090, f"{R}/sft-0.6b-e3.log", "③ 逐句证据 + 理由，自然比例"),
    ("sft-0.6b-e5", "configs/rt/exp/sft_0.6b_e5.yaml", "outputs/rt/resp/train_full.jsonl", "c5k0nax3ry（主实例）",
     ENV_5090, f"{R}/sft-0.6b-e5.log", "⑤ 回答级每句证据（774 条回答）"),
    ("sft-0.6b-e5b", "configs/rt/exp/sft_0.6b_e5b.yaml", "outputs/rt/resp/train_full.jsonl", "c5k0nax3ry（主实例）",
     ENV_5090, f"{R}/sft-0.6b-e5b.log", "⑤b 回答级只列错误句（774 条回答）"),
    ("sft-0.6b-e4", "configs/rt/exp/sft_0.6b_e4.yaml", "outputs/rt/claims/train_sub35_nat.jsonl",
     "gzw4qpf8tl（机器 2）", ENV_5090, f"{R}_diag/sft-0.6b-e4.log", "④ 逐句先判定后证据和理由"),
    ("sft-9b-verdict", "configs/rt/exp/sft_9b_verdict.yaml", "outputs/rt/claims/train_sub35_r3.jsonl",
     "gzw4qpf8tl（机器 2）", ENV_5090, f"{R}_diag/sft-9b-verdict.log", "诊断 B：9B 只判定，1:3"),
    *[(f"sft-0.6b-syn-{a}", f"configs/rt/exp/sft_0.6b_syn_{a}.yaml", f"{REBUILT}/syn_r1_{a}.jsonl",
       "c5k0nax3ry（主实例）", ENV_5090, f"{R}/sft-0.6b-syn-{a}.log",
       f"0.6B 合成实验 {a} 组；训练集按 ragtruth.mine mix --arm {a} --seed 0 从 r1 合成池重建")
      for a in ("direct", "critic", "random")],
    ("sft-9b-syn-direct", "configs/rt/exp/sft_9b_syn_direct.yaml", f"{REBUILT}/syn9b_r1_direct.jsonl",
     "gzw4qpf8tl（机器 2）", ENV_5090, f"{R}_diag/sft-9b-syn-direct.log", "9B 合成实验 direct 组（重建训练集）"),
    *[(f"sft-9b-syn-{a}", f"configs/rt/exp/sft_9b_syn_{a}.yaml", f"{REBUILT}/syn9b_r1_{a}.jsonl",
       "pwr6zxz5jh（机器 3）", ENV_5090, f"{R}_syn9b/sft-9b-syn-{a}.log", f"9B 合成实验 {a} 组（重建训练集）")
      for a in ("critic", "random")],
]


def train_summary(log: str | None) -> dict:
    """从训练日志里取 HF Trainer 打印的字典：最后一个 loss、train_runtime、总步数。"""
    if not log or not Path(log).exists():
        return {}
    text = Path(log).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    out = {}
    for m in re.finditer(r"\{'[a-z_]+':.*?\}", text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "loss" in d:
            out["final_loss"] = d["loss"]
        if "train_runtime" in d:
            out["runtime_s"] = float(d["train_runtime"])
    # 只认训练进度条（行首直接是百分比）；跳过「Map: …」数据预处理和「Writing model shards 1/1」
    totals = [int(t) for t in re.findall(r"(?m)^\s*\d+%\|[^|\n]*\|\s*\d+/(\d+) \[", text) if int(t) > 1]
    if totals:
        out["steps"] = max(totals)
    return out


def main() -> None:
    reg = Path("outputs/runs/registry_all.jsonl")
    reg.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if reg.exists():
        have = {(json.loads(x)["name"], json.loads(x)["kind"]) for x in reg.read_text(encoding="utf-8").splitlines()}
    rows = []
    for name, cfgp, data, machine, env, log, note in RUNS:
        cfg = load_config(cfgp)
        t = train_summary(log)
        ev = Path(f"outputs/rt/eval/{name}/metrics.json")
        base = {"name": name, "machine": machine, "env": env, "code": {"note": "unknown（manifest 组件上线前的实验，未记录）"},
                "backfilled": True, "note": note}
        rows.append({**base, "kind": "run", "out_dir": cfg.get("coldstart", {}).get("output_dir"),
                     "finished": None, "duration_min": round(t["runtime_s"] / 60, 1) if "runtime_s" in t else None,
                     "config": {"config_file": cfgp, **cfg}, "data": [data_fingerprint(data), data_fingerprint(cfgp)],
                     "base_model": cfg.get("base_model"), "train": t})
        if ev.exists():
            m = json.loads(ev.read_text(encoding="utf-8"))
            rows.append({**base, "kind": "eval", "out_dir": str(ev.parent), "finished": None, "duration_min": None,
                         "config": {"test": TEST, "style": cfg.get("style", "full"),
                                    "resp_style": {"ragtruth_resp": "evidence/plain"}.get(cfg.get("task"))},
                         "data": [data_fingerprint(TEST)],
                         "metrics": {k: m[k] for k in ("sentence", "response", "f1_ci95") if k in m}})
    zs = Path("outputs/rt/eval/zs-qwen3-0.6b/metrics.json")
    if zs.exists():
        m = json.loads(zs.read_text(encoding="utf-8"))
        rows.append({"name": "zs-qwen3-0.6b", "kind": "eval", "machine": "local", "env": ENV_LOCAL, "backfilled": True,
                     "code": {"note": "unknown"}, "note": "零样本基线，不训练", "out_dir": str(zs.parent),
                     "finished": None, "duration_min": None, "config": {"test": TEST}, "data": [data_fingerprint(TEST)],
                     "metrics": {k: m[k] for k in ("sentence", "response") if k in m}})
    new = [r for r in rows if (r["name"], r["kind"]) not in have]
    with open(reg, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"backfilled {len(new)} records ({len(rows) - len(new)} already present)")


if __name__ == "__main__":
    main()
