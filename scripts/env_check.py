"""环境检查：Python 包版本、GPU、配置字段与所装 TRL 版本是否对得上、chat template 渲染与 token 长度。

用法：
  python -m scripts.env_check                                   # 包、GPU、配置字段
  python -m scripts.env_check --model Qwen/Qwen3-0.6B-Base --data outputs/smoke/coldstart.jsonl --max_length 3072
退出码非 0 表示有必须修的问题；[warn] 只是提示。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import platform
import sys

from common.config import load_config

PACKAGES = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm", "verl"]
REQUIRED = {"torch", "transformers", "trl", "peft", "datasets", "accelerate"}
# (配置文件, 取 trainer 字段的路径, 对应的 TRL 配置类)
TRAINER_CONFIGS = [
    ("configs/sft.yaml", ["trainer"], "SFTConfig"),
    ("configs/sft.yaml", ["full", "trainer"], "SFTConfig"),
    ("configs/grpo_trl.yaml", ["trainer"], "GRPOConfig"),
    ("configs/smoke/sft.yaml", ["trainer"], "SFTConfig"),
    ("configs/smoke/grpo_trl.yaml", ["trainer"], "GRPOConfig"),
]

problems: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)
    print(f"[FAIL] {msg}")


def check_packages() -> None:
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"[ok]   {name} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001  导入失败的原因各种各样，原样打印
            msg = f"{name} not importable ({type(e).__name__}: {e})"
            if name in REQUIRED:
                fail(msg)
            else:
                note = "；Windows 上请在 WSL2 里装" if platform.system() == "Windows" else ""
                print(f"[warn] {msg}{note}")


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        fail(f"torch {torch.__version__} 看不到 CUDA（装成了 CPU 版？torch.version.cuda={torch.version.cuda}）")
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[ok]   cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB, cc {p.major}.{p.minor}")
    if not torch.cuda.is_bf16_supported():
        fail("GPU 不支持 bf16，配置里的 bf16: true 要改成 fp16")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = float((x @ x).float().abs().mean())
    print(f"[ok]   bf16 matmul (mean {y:.1f}), free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")


def check_trainer_fields() -> None:
    try:
        trl = importlib.import_module("trl")
    except ImportError:
        return
    for path, keys, cls_name in TRAINER_CONFIGS:
        try:
            cfg = load_config(path)
        except FileNotFoundError:
            continue
        for k in keys:
            cfg = (cfg or {}).get(k)
        if not cfg:
            continue
        fields = {f.name for f in dataclasses.fields(getattr(trl, cls_name))}
        unknown = sorted(set(cfg) - fields)
        where = f"{path}:{'.'.join(keys)}"
        if unknown:
            fail(f"{where} 有 {cls_name} 不认识的字段 {unknown}（trl {trl.__version__}）")
        else:
            print(f"[ok]   {where} 的 {len(cfg)} 个字段 {cls_name} 都认识")


def check_template(model: str, data: str, max_length: int) -> None:
    from transformers import AutoTokenizer

    from common.prompts import critic_messages
    from common.schema import read_jsonl
    from train.sft import PLACEHOLDER_THINK, target_text

    tok = AutoTokenizer.from_pretrained(model)
    samples = list({s["id"]: s for s in read_jsonl(data)}.values())
    lengths = []
    for s in samples:
        msgs = critic_messages(s) + [{"role": "assistant", "content": target_text(s, PLACEHOLDER_THINK)}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        # Qwen3 模板会改写 assistant 里的 think，回答部分必须恰好保留一段（系统提示的格式说明里也有 <think>，不算）
        n_think = text[len(prompt):].count("<think>") if text.startswith(prompt) else -1
        if n_think != 1:
            fail(f"{s['id']}: 渲染后的训练文本有 {n_think} 个 <think>")
            print(text[-600:])
            break
        lengths.append(len(tok(text)["input_ids"]))
    gen = tok.apply_chat_template(critic_messages(samples[0]), tokenize=False, add_generation_prompt=True)
    print(f"[info] 生成提示结尾: {gen[-60:]!r}")
    if lengths:
        over = sum(n > max_length for n in lengths)
        print(f"[{'ok' if not over else 'FAIL'}]   {len(lengths)} 条训练文本 token 数 "
              f"min {min(lengths)} / max {max(lengths)}，超过 {max_length} 的 {over} 条")
        if over:
            problems.append(f"{over} 条样本超过 max_length {max_length}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="检查这个模型的 chat template")
    ap.add_argument("--data", default="outputs/smoke/coldstart.jsonl")
    ap.add_argument("--max_length", type=int, default=3072)
    args = ap.parse_args()
    check_packages()
    check_gpu()
    check_trainer_fields()
    if args.model:
        check_template(args.model, args.data, args.max_length)
    print(f"\n{len(problems)} 个问题" if problems else "\n环境检查通过")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
