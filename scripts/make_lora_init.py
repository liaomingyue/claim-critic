"""给一个没有现成 LoRA 的基座造「等于基座」的起点 LoRA（B 全零），结构和 train.sft 训出来的一样：
按纯文本 CausalLM 加载（Qwen3.5 不带视觉塔），all-linear，r / alpha 可调。基座评测和 GRPO 从基座起步时用。
  python scripts/make_lora_init.py <基座目录> <输出目录> [--r 32] [--alpha 64]
"""
import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("out")
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--alpha", type=int, default=64)
a = ap.parse_args()
torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
m = get_peft_model(m, LoraConfig(r=a.r, lora_alpha=a.alpha, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"))
for n, p in m.named_parameters():   # PEFT 默认 B 就是 0；显式清一次，保证输出和基座完全相同
    if "lora_B" in n:
        p.data.zero_()
m.save_pretrained(a.out)
print("LORA_INIT", a.base, "->", a.out, "r", a.r, "alpha", a.alpha, flush=True)
