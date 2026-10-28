"""GRPO 备选实现（TRL GRPOTrainer + LoRA），用于在 2 张 24 GB 卡上先跑通流程（里程碑 4）。

用法：accelerate launch -m train.grpo_trl --config configs/grpo_trl.yaml --version 1
"""
from __future__ import annotations

import argparse
import json
import random

from common.config import load_config
from common.prompts import critic_messages
from common.schema import cited_ids, read_jsonl
from train.grpo import balanced_order
from train.reward import trl_reward


def drop_overlong(samples: list[dict], tokenizer, max_prompt_length: int, to_messages=critic_messages) -> list[dict]:
    """TRL 1.x 不再截断 prompt，超长的直接丢掉（同 verl 的 filter_overlong_prompts）。"""
    kept_ = [s for s in samples
             # transformers 5 默认返回 BatchEncoding（len 恒为 2），要 return_dict=False 才是 token 列表
             if len(tokenizer.apply_chat_template(to_messages(s), add_generation_prompt=True, tokenize=True,
                                                  return_dict=False)) <= max_prompt_length]
    if len(kept_) < len(samples):
        print(f"[warn] dropped {len(samples) - len(kept_)} prompts longer than {max_prompt_length} tokens")
    return kept_


def build_dataset(path: str, batch_size: int | None, seed: int = 0, tokenizer=None, max_prompt_length: int | None = None,
                  task: str = "claim", tips: bool = False):
    """batch_size 为 None 时不重排（验证集）。训练集需配合 GRPOConfig.shuffle_dataset=False 使用。
    task: claim | ragtruth（见 ragtruth.task，reward 只需要 gold 一列）。"""
    from datasets import Dataset

    samples = list(read_jsonl(path))
    if task == "ragtruth":
        from ragtruth import task as rt
        if tokenizer is not None and max_prompt_length:
            samples = drop_overlong(samples, tokenizer, max_prompt_length, lambda s: rt.messages(s, tips))
        if batch_size:  # 没有错误类型，balanced_order 不适用；只打乱
            random.Random(seed).shuffle(samples)
        return Dataset.from_list([{"prompt": rt.messages(s, tips), "gold": rt.gold_json(s)} for s in samples])
    if tokenizer is not None and max_prompt_length:
        samples = drop_overlong(samples, tokenizer, max_prompt_length)
    if batch_size:
        samples = balanced_order(samples, batch_size, seed)
    rows = [{"prompt": critic_messages(s), "label": s["label"], "type": s["type"],
             "refs": json.dumps(s["refs"], ensure_ascii=False),
             "cited": json.dumps(cited_ids(s), ensure_ascii=False),
             "evidence": s.get("evidence", "")} for s in samples]
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_trl.yaml")
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = cfg["data_dir"].format(version=args.version)

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(cfg["init_model"])
    max_len = cfg.get("max_prompt_length")
    task = cfg.get("task", "claim")
    if task == "ragtruth":
        from ragtruth.task import trl_reward as reward_fn
    else:
        reward_fn = trl_reward
    targs = GRPOConfig(output_dir=cfg["output_dir"].format(version=args.version), **cfg["trainer"])
    trainer = GRPOTrainer(
        model=cfg["init_model"],
        reward_funcs=reward_fn,
        args=targs,
        train_dataset=build_dataset(f"{data_dir}/train.jsonl", cfg["prompts_per_step"], args.version, tok, max_len,
                                    task, cfg.get("prompt_tips", False)),
        eval_dataset=build_dataset(f"{data_dir}/val.jsonl", None, tokenizer=tok, max_prompt_length=max_len, task=task,
                                    tips=cfg.get("prompt_tips", False)),
        peft_config=LoraConfig(**cfg["lora"]),
    )
    trainer.train()
    trainer.save_model(targs.output_dir)


if __name__ == "__main__":
    main()
