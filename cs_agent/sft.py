"""客服 Agent 的多轮 SFT（9B + LoRA）：样本是 cs_agent.sft_data 转好的对话，token 化和 rollout 完全一样（qwen_fmt.Conv），
只对客服的输出（文字回复、工具调用、结尾的 <|im_end|>）算交叉熵。
  python -m cs_agent.sft --base <Qwen3.5-9B> --data outputs/rt/cs/sft_data/sft_B.jsonl --out <adapter 目录> [--epochs 2] [--lr 1e-4]
显存：一条对话 8–12k token；输出层只在算 loss 的位置上算（全序列的 logits 是 1.2 万 × 25 万词表，单这一项就要 6 GB）。
LoRA 结构同 RAG 的 9B SFT（纯文本 CausalLM 加载、all-linear），GRPO 那边按图文模型加载同一个 adapter（train.gen_grpo.load_adapters_strict）。
每个 epoch 末存一次（<out>/epoch<k>），最后一个 epoch 的同时存到 <out>。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cs_agent  # noqa: F401
from cs_agent.rollout import agent_system_prompt
from cs_agent.sft_data import encode


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tau2.domains.retail.environment import get_environment
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=10240)   # 9B 单卡：更长的对话激活放不下；保留 98.6% 的对话
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    env = get_environment()
    system, tools = agent_system_prompt(env), [t.openai_schema for t in env.get_tools()]
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = rows[: a.limit] if a.limit else rows
    data = []
    for r in rows:
        e = encode(tok, r["messages"], system, tools)
        ids = e["prompt_ids"] + e["ids"]
        lm = [0] * len(e["prompt_ids"]) + e["lmask"]
        if len(ids) <= a.max_len:
            data.append((ids, lm))
    print(f"SFT {len(data)} dialogs, {sum(len(d[0]) for d in data)} tokens, {sum(sum(d[1]) for d in data)} loss tokens", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(data) / a.accum) * a.epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    inner = model.base_model.model   # Qwen3_5ForCausalLM：.model 出隐状态，.lm_head 只在要算 loss 的位置上用
    from torch.utils.checkpoint import checkpoint

    def ce_sum(hc, tc):
        return torch.nn.functional.cross_entropy(inner.lm_head(hc).float(), tc, reduction="sum")
    step, t0 = 0, time.time()
    out = Path(a.out)
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for b in range(0, len(order), a.accum):
            chunk = [data[i] for i in order[b:b + a.accum]]
            n_loss = sum(sum(lm) for _, lm in chunk) or 1
            tot = 0.0
            for ids, lm in chunk:
                x = torch.tensor([ids], device="cuda")
                h = inner.model(input_ids=x).last_hidden_state[0]
                pos = torch.tensor([k - 1 for k in range(1, len(ids)) if lm[k]], device="cuda")   # 位置 k-1 预测 token k
                tgt = torch.tensor([ids[k] for k in range(1, len(ids)) if lm[k]], device="cuda")
                # 分块算输出层和交叉熵，反向时每块重算（checkpoint）：一条对话要学的 token 多到 2,600 个时，一次性的 float logits 就要 2.5 GB（实测在这里 OOM）
                loss = sum(checkpoint(ce_sum, h[pos[c:c + 512]], tgt[c:c + 512], use_reentrant=False) for c in range(0, len(pos), 512)) / n_loss
                loss.backward()
                tot += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"step {step}/{total} ep {ep} loss {tot:.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        model.save_pretrained(str(out / f"epoch{ep + 1}"))
        print(f"EPOCH {ep + 1} saved -> {out / f'epoch{ep + 1}'}", flush=True)
    model.save_pretrained(str(out))
    (out / "sft_manifest.json").write_text(json.dumps({**vars(a), "dialogs": len(data), "steps": step}, indent=1), encoding="utf-8")
    print("SFT_DONE", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
