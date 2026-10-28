"""GRPO 训练（verl）。两步：

1. prepare：训练集 JSONL → verl 需要的 parquet。按 batch 预排好顺序（每个 batch 正负 1:1、各类错误轮转），
   训练时关闭 shuffle，保证类别平衡在 batch 粒度成立。
2. launch：拼 hydra 参数，启动 verl.trainer.main_ppo。reward 用 train/reward.py:compute_score。

每轮都从 critic-0（SFT 冷启动）重新起训，只换训练集版本。

用法：
  python -m train.grpo prepare --config configs/grpo.yaml --version 1
  python -m train.grpo launch  --config configs/grpo.yaml --version 1 [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from common.config import load_config
from common.prompts import critic_messages
from common.schema import ERROR_TYPES, cited_ids, read_jsonl

REWARD_PATH = str(Path(__file__).with_name("reward.py"))


def balanced_order(samples: list[dict], batch_size: int, seed: int = 0) -> list[dict]:
    """把样本排成若干 batch，每个 batch 内正负各半、负样本在现有类型间轮转。凑不满一个 batch 的尾部丢弃。"""
    rng = random.Random(seed)
    pos = [s for s in samples if s["label"] == "faithful"]
    neg = defaultdict(list)
    for s in samples:
        if s["label"] == "hallucination":
            neg[s["type"]].append(s)
    rng.shuffle(pos)
    types = [t for t in ERROR_TYPES if neg[t]]
    if not types:
        return []
    for t in types:
        rng.shuffle(neg[t])

    half = batch_size // 2
    ordered = []
    while len(pos) >= half:
        batch_neg = []
        for i in range(half):
            t = types[i % len(types)]
            if not neg[t]:
                return ordered
            batch_neg.append(neg[t].pop())
        batch = [pos.pop() for _ in range(half)] + batch_neg
        rng.shuffle(batch)
        ordered += batch
    return ordered


def to_verl_row(s: dict) -> dict:
    gt = {"label": s["label"], "type": s["type"], "refs": s["refs"], "cited": cited_ids(s),
          "evidence": s.get("evidence", "")}
    return {
        "data_source": "claim_critic",
        "prompt": critic_messages(s),
        "ability": "faithfulness",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "extra_info": {"id": s["id"], "type": s["type"], "round": s.get("round", 0)},
    }


def prepare(cfg: dict, version: int) -> None:
    import pandas as pd

    data_dir = Path(cfg["data_dir"].format(version=version))
    train = balanced_order(list(read_jsonl(data_dir / "train.jsonl")), cfg["train_batch_size"], seed=version)
    val = list(read_jsonl(data_dir / "val.jsonl"))
    pd.DataFrame([to_verl_row(s) for s in train]).to_parquet(data_dir / "train.parquet")
    pd.DataFrame([to_verl_row(s) for s in val]).to_parquet(data_dir / "val.parquet")
    print(f"train {len(train)} rows ({len(train) // cfg['train_batch_size']} batches), val {len(val)}")


def verl_overrides(cfg: dict, version: int) -> list[str]:
    data_dir = cfg["data_dir"].format(version=version)
    o = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": f"{data_dir}/train.parquet",
        "data.val_files": f"{data_dir}/val.parquet",
        "data.train_batch_size": cfg["train_batch_size"],
        "data.max_prompt_length": cfg["max_prompt_length"],
        "data.max_response_length": cfg["max_response_length"],
        "data.shuffle": False,  # 顺序已在 prepare 里按类别平衡排好
        "data.filter_overlong_prompts": True,
        "actor_rollout_ref.model.path": cfg["init_model"],
        "actor_rollout_ref.actor.optim.lr": cfg["lr"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": cfg["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": cfg["kl_coef"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": cfg["group_size"],
        "actor_rollout_ref.rollout.temperature": cfg["temperature"],
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.gpu_memory_utilization": cfg["rollout_gpu_mem"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": cfg["rollout_tp"],
        "reward.custom_reward_function.path": REWARD_PATH,  # verl 0.9 起的位置（旧的顶层 custom_reward_function 仅兼容）
        "reward.custom_reward_function.name": "compute_score",
        "trainer.total_training_steps": cfg["total_steps"],
        "trainer.test_freq": cfg["test_freq"],
        "trainer.save_freq": cfg["save_freq"],
        "trainer.n_gpus_per_node": cfg["n_gpus"],
        "trainer.nnodes": 1,
        "trainer.project_name": cfg["project_name"],
        "trainer.experiment_name": f"{cfg['experiment_prefix']}-v{version}",
        "trainer.default_local_dir": cfg["output_dir"].format(version=version),
        "trainer.logger": "['console','wandb']",
    }
    if cfg.get("lora_rank"):
        o["actor_rollout_ref.model.lora_rank"] = cfg["lora_rank"]
        o["actor_rollout_ref.model.lora_alpha"] = cfg["lora_alpha"]
        o["actor_rollout_ref.model.target_modules"] = "all-linear"
    o.update(cfg.get("extra_overrides") or {})
    return [f"{k}={v}" for k, v in o.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "launch"])
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "prepare":
        prepare(cfg, args.version)
        return
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *verl_overrides(cfg, args.version)]
    print(" \\\n  ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
