"""真实性判别器的分类头版本：Qwen3.5-9B + LoRA，取最后一个 token 的隐藏状态接线性层（4096 → 1），输出 logit。

与生成式版本（ragtruth.discrim + train.sft，生成 <origin>natural|synthetic</origin>）同数据、同 LoRA 设置，做对比。
损失：GAN 判别器的基本形式——二元交叉熵；加 GAN 训练常用的单边标签平滑（自然错误标签 0.9），防止判别器过于自信、
把所有合成样本都压到 0 附近而失去排序能力。不用 WGAN / hinge：它们是为了给生成器提供梯度，这里生成器靠 DPO 偏好对更新。
打分用 logit（无上下界，排序分辨率高）；评测报告准确率、AUC、校准误差（ECE，用 sigmoid 概率）。

用法：
  python -m train.discrim_head train --out /root/autodl-tmp/ckpt/discrim-head-9b
  python -m train.discrim_head score --ckpt /root/autodl-tmp/ckpt/discrim-head-9b --rows outputs/rt/discrim/heldout.jsonl \
      --out outputs/rt/discrim/heldout_head_scores.jsonl --eval
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from common import manifest
from common.schema import read_jsonl, write_jsonl
from ragtruth import discrim

BASE = "/root/autodl-tmp/models/Qwen3.5-9B"


def load(base: str, ckpt: str | None = None, train: bool = False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"   # 取最后一个位置的隐藏状态：左侧补齐时最后一个位置就是真实的最后一个 token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if ckpt:
        lm = PeftModel.from_pretrained(lm, f"{ckpt}/adapter", is_trainable=train)
    else:
        lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    head = torch.nn.Linear(lm.get_base_model().config.get_text_config().hidden_size, 1).to("cuda", torch.float32)
    if ckpt:
        head.load_state_dict(torch.load(f"{ckpt}/head.pt"))
    return tok, lm, head


def encode(tok, rows: list[dict], max_len: int):
    texts = [tok.apply_chat_template(discrim.messages(r), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for r in rows]
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")


def logits(lm, head, batch):
    """解码器最后一层、最后一个位置的隐藏状态 → 线性层。直接调解码器主干，不算整个词表的 lm_head。"""
    h = lm.get_base_model().model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    return head(h[:, -1].float()).squeeze(-1)


def cmd_train(args) -> None:
    import torch

    t0 = time.time()
    tok, lm, head = load(args.base)
    lm.gradient_checkpointing_enable()
    lm.enable_input_require_grads()
    lm.train()
    rows = list(read_jsonl(args.data))
    opt = torch.optim.AdamW([{"params": [p for p in lm.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": head.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    steps = math.ceil(len(rows) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = random.Random(0)
    step, losses = 0, []
    for ep in range(args.epochs):
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            y = torch.tensor([args.smooth if r["origin_label"] == "natural" else 0.0], device="cuda")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(lm, head, encode(tok, [r], args.max_len)), y)
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {ep} step {step}/{steps} loss {sum(losses[-160:]) / len(losses[-160:]):.4f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(out / "adapter")
    torch.save(head.state_dict(), out / "head.pt")
    manifest.write("run", out.name, out, vars(args), [args.data], started=t0,
                   extra={"base_model": args.base, "adapter_dir": str(out / "adapter"),
                          "train": {"steps": step, "final_loss": sum(losses[-160:]) / max(1, len(losses[-160:]))},
                          "arch": "LoRA r16 all-linear + Linear(4096→1) on last token; BCE, one-sided smoothing "
                                  f"{args.smooth}"})


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    total, err = len(probs), 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(probs) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if idx:
            err += len(idx) / total * abs(sum(probs[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return err


def summarize(rows: list[dict], key: str = "score", prob_key: str | None = "p_natural") -> dict:
    lab = [int(r["label"] == "natural") for r in rows]
    sc = [r[key] for r in rows]
    out = {"n": len(rows), "auc": round(auc(sc, lab), 4)}
    if prob_key:
        pr = [r[prob_key] for r in rows]
        out.update({"accuracy": round(sum((p >= 0.5) == bool(y) for p, y in zip(pr, lab)) / len(rows), 4),
                    "ece": round(ece(pr, lab), 4),
                    "mean_p_natural": {"natural": round(sum(p for p, y in zip(pr, lab) if y) / max(1, sum(lab)), 3),
                                       "synthetic": round(sum(p for p, y in zip(pr, lab) if not y) /
                                                          max(1, len(lab) - sum(lab)), 3)}})
    return out


def cmd_score(args) -> None:
    import torch

    tok, lm, head = load(args.base, args.ckpt)
    lm.eval()
    rows = list(read_jsonl(args.rows))
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            lg = logits(lm, head, encode(tok, chunk, args.max_len)).tolist()
            out += [{"id": r["id"], "score": s, "p_natural": 1 / (1 + math.exp(-s)), "label": r.get("origin_label"),
                     "tactic": r.get("tactic"), "type": r.get("type"), "task_type": r.get("task_type")}
                    for r, s in zip(chunk, lg)]
    write_jsonl(args.out, out)
    if args.eval:
        summ = summarize(out)
        summ["by_type"] = {t: summarize([r for r in out if r["type"] == t]) for t in ("conflict", "baseless")}
        Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(
            json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summ, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", default=BASE)
    t.add_argument("--data", default="outputs/rt/discrim/train.jsonl")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--accum", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--head_lr", type=float, default=1e-3)
    t.add_argument("--smooth", type=float, default=0.9, help="自然错误的标签值（单边标签平滑）")
    t.add_argument("--max_len", type=int, default=3072)
    s = sub.add_parser("score")
    s.add_argument("--base", default=BASE)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=1, help="Qwen3.5 有线性注意力层，左侧补齐的批量推理是否正确未验证，默认逐条")
    s.add_argument("--max_len", type=int, default=3072)
    s.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_score(args)


if __name__ == "__main__":
    main()
