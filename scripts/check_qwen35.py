"""上机检查：Qwen3.5-9B 能否按纯文本 CausalLM 加载、LoRA 该挂哪些模块、一条样本前向 + 反向是否正常、显存占用。

用法（在 5090 实例上）：python -m scripts.check_qwen35 --model /root/autodl-tmp/models/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import collections
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="outputs/rt/claims/train_sub.jsonl")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from ragtruth.task import messages, target_text

    cfg = AutoConfig.from_pretrained(args.model)
    print("config:", type(cfg).__name__, getattr(cfg, "architectures", None))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s; "
          f"params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B; "
          f"gpu {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    kinds = collections.Counter(n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                                if isinstance(m, torch.nn.Linear))
    print("linear module names:", dict(kinds))
    print("has vision modules:", any("visual" in n or "vision" in n for n, _ in model.named_modules()))

    tok = AutoTokenizer.from_pretrained(args.model)
    c = json.loads(open(args.data, encoding="utf-8").readline())
    msgs = messages(c) + [{"role": "assistant", "content": target_text(c)}]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
    print("tokens:", len(ids))
    print("rendered tail:", repr(tok.apply_chat_template(msgs, tokenize=False)[-300:]))

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    peft.print_trainable_parameters()
    peft.train()  # 梯度检查点只在 train 模式生效；from_pretrained 返回的是 eval 模式
    x = torch.tensor([ids], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = peft(input_ids=x, labels=x)
    out.loss.backward()
    torch.cuda.synchronize()
    print(f"fwd+bwd ok: loss {out.loss.item():.3f}, {time.time() - t0:.1f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
