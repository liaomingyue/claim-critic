"""训练框架基准：TRL 原生（同 train.sft）和 Unsloth 跑相同数据、相同 LoRA 设置若干步，记录速度、显存和 loss。

用户提出（2026-10-28）：不同框架自带加速。Unsloth 官方说 Qwen3.5 训练比 FA2 写法快 1.5 倍、省一半显存。
在独立 venv 里跑（Unsloth 会带自己要求的 torch / transformers 版本，不能动 vLLM 用的共享环境）。

用法：python -u -m train.bench_train --framework trl --data outputs/rt/claims/closed_g0.jsonl --out outputs/rt/bench/trl.json
      <venv>/bin/python -u -m train.bench_train --framework unsloth --data ... --out outputs/rt/bench/unsloth.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
          "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]   # 9B 文本部分的线性层（含线性注意力）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["trl", "unsloth"], required=True)
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max_length", type=int, default=3072)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    if args.framework == "trl":
        return bench_trl(args)
    from unsloth import FastLanguageModel, FastModel   # 必须先于 trl / transformers 导入

    t0 = time.time()
    try:
        model, tok = FastLanguageModel.from_pretrained(args.base, max_seq_length=args.max_length, load_in_4bit=False,
                                                       dtype=torch.bfloat16)
        loader = "FastLanguageModel"
    except Exception as e:  # noqa: BLE001  Qwen3.5 是多模态结构，纯文本加载器不认时换通用加载器
        print(f"FastLanguageModel failed ({e}); trying FastModel", flush=True)
        model, tok = FastModel.from_pretrained(args.base, max_seq_length=args.max_length, load_in_4bit=False,
                                               dtype=torch.bfloat16)
        loader = "FastModel"
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, target_modules=LINEAR,
                                             use_gradient_checkpointing="unsloth", random_state=0)
    load_s = time.time() - t0

    from common.schema import read_jsonl
    from train.sft import build_dataset
    from trl import SFTTrainer

    ds = build_dataset(list(read_jsonl(args.data)), None, "ragtruth", False, "full")
    proc = getattr(tok, "tokenizer", tok)
    trainer = SFTTrainer(model=model, args=sft_args(args, "/root/autodl-tmp/ckpt/bench-unsloth"), train_dataset=ds,
                         processing_class=proc)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    trainer.train()
    train_s = time.time() - t1
    log = trainer.state.log_history
    finish(args, {"framework": "unsloth", "loader": loader, "load_s": round(load_s, 1)}, trainer, train_s)


def sft_args(args, out: str, **extra):
    from trl import SFTConfig
    return SFTConfig(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                     max_steps=args.steps, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=0.03,
                     bf16=True, logging_steps=2, max_length=args.max_length, save_strategy="no", report_to="none",
                     seed=0, **extra)


def bench_trl(args) -> None:
    """同 train.sft：TRL 按路径加载、LoRA r16 all-linear、梯度检查点。"""
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    from common.schema import read_jsonl
    from train.sft import build_dataset

    t0 = time.time()
    ds = build_dataset(list(read_jsonl(args.data)), None, "ragtruth", False, "full")
    trainer = SFTTrainer(model=args.base, processing_class=AutoTokenizer.from_pretrained(args.base), train_dataset=ds,
                         args=sft_args(args, "/root/autodl-tmp/ckpt/bench-trl", gradient_checkpointing=True,
                                       model_init_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"}),
                         peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    trainer.train()
    finish(args, {"framework": "trl", "load_s": round(load_s, 1)}, trainer, time.time() - t1)


def finish(args, head: dict, trainer, train_s: float) -> None:
    import torch
    log = trainer.state.log_history
    tokens = next((x["num_tokens"] for x in reversed(log) if "num_tokens" in x), None)
    res = {**head, "steps": args.steps, "train_s": round(train_s, 1), "s_per_step": round(train_s / args.steps, 2),
           "tokens_per_s": round(float(tokens) / train_s) if tokens else None,
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
           "loss": [round(float(x["loss"]), 4) for x in log if "loss" in x]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
