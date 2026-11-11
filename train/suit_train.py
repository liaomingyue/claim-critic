"""合适度模型（错误类型先验）：2B + LoRA，选择题形式，多标签。

只看原文和问题（不看任何回答），问「回答这个问题时可能出现哪些错误」，8 个机制列成 A–H 选项（每条样本随机打乱顺序），
在作答位置读 8 个字母的 logit，每个字母单独过 sigmoid、各算一个二分类交叉熵（不在选项之间做 softmax：多类错可以同时出现）。
评测时每条样本按 --eval_perms 个固定的随机顺序各读一次，按选项内容对回机制后取平均 logit；全局一个温度在 holdout 上拟合。

  python -m train.suit_train --base <MiniCPM5-2B> --out <目录> [--epochs 3 --lr 1e-4]
  → <目录>/adapter，<目录>/preds_{holdout,test,policy}.jsonl，<目录>/report.json（和 results/suit_baselines.json 同口径）
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.suit_data import MECHS, OUT, evaluate

LETTERS = "ABCDEFGH"
DEFS = {
    "certainty_or_scope_shift": "certainty or scope shift: states something more (or less) certain, general or absolute than the source does (e.g. 'may' becomes 'will', 'some' becomes 'all')",
    "near_synonym_shift": "near-synonym shift: a word or phrase is replaced by a similar-looking one that changes the meaning",
    "contradicts_source": "contradicts the source: states something the source directly contradicts",
    "unsupported_addition": "unsupported addition: adds a specific fact, detail or inference the source does not contain or support",
    "wrong_attribution": "wrong attribution: assigns a statement, action, quote or number to the wrong person, entity or place",
    "vague_evaluation_or_addon": "vague evaluation or add-on: adds a generic judgement, opinion or filler claim not in the source",
    "number_or_date": "number or date error: a number, quantity, date or time is wrong",
    "merged_across_passages": "merged across passages: combines facts from different passages or parts of the source into one wrong claim",
}
assert set(DEFS) == set(MECHS)

SYSTEM = """You will see the exact input an assistant is about to receive: instructions plus source material (a news article to summarize, or passages to answer a question from).
Before any answer is written, predict which kinds of faithfulness errors an assistant answering this input is likely to make. More than one kind can apply.

Options:
{options}

Answer with the letters of all likely error kinds."""

# 对照组：再给一条忠实于原文的回答，问「写这样一条回答时」可能出现哪些错误
SYSTEM_ANS = SYSTEM.replace(
    "Before any answer is written, predict which kinds of faithfulness errors an assistant answering this input is likely to make.",
    "You also see one faithful answer to this input. Predict which kinds of faithfulness errors an assistant writing an answer like this one is likely to make.")
assert SYSTEM_ANS != SYSTEM


def render(source: str, perm: list[str], answer: str | None = None) -> list[dict]:
    opts = "\n".join(f"{LETTERS[i]}. {DEFS[m]}" for i, m in enumerate(perm))
    if answer is None:
        return [{"role": "system", "content": SYSTEM.format(options=opts)}, {"role": "user", "content": source}]
    return [{"role": "system", "content": SYSTEM_ANS.format(options=opts)},
            {"role": "user", "content": f"[Input]\n{source}\n\n[A faithful answer]\n{answer}"}]


def perms_for(rid: str, k: int) -> list[list[str]]:
    """评测用的固定顺序：第 0 个是原顺序，其余按样本 id 定种子打乱（可复现）。"""
    out = [list(MECHS)]
    rng = random.Random(f"suit|{rid}")
    while len(out) < k:
        p = list(MECHS)
        rng.shuffle(p)
        out.append(p)
    return out


def letter_ids(tok) -> list[int]:
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, f"letter {L} is not a single token: {t}"
        ids.append(t[0])
    return ids


def fit_temperature(logits: list[list[float]], labels: list[list[int]]) -> float:
    """全局一个温度：在 holdout 上最小化平均 BCE（一维网格 + 细化）。"""
    def nll(T: float) -> float:
        s = 0.0
        for zs, ys in zip(logits, labels):
            for z, y in zip(zs, ys):
                p = 1 / (1 + math.exp(-z / T))
                p = min(max(p, 1e-6), 1 - 1e-6)
                s -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return s
    grid = [0.25 * i for i in range(1, 41)]
    best = min(grid, key=nll)
    fine = [best + 0.01 * i for i in range(-24, 25) if best + 0.01 * i > 0.05]
    return round(min(fine, key=nll), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8, help="每次更新的样本数（逐条前向、梯度累积）")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--eval_perms", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true", help="不训练，加载 <out>/adapter 只做评测")
    ap.add_argument("--with_answer", action="store_true", help="对照组：输入再加一条正确回答（数据用 *_ans.jsonl）")
    a = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    lids = letter_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    if a.eval_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(out / "adapter"))
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    sfx = "_ans" if a.with_answer else ""
    data = {n: list(read_jsonl(OUT / f"{n}{sfx}.jsonl")) for n in ("train", "holdout", "test", "policy")}

    def encode(src: str, perm: list[str], ans: str | None = None) -> list[int]:
        ids = tok.apply_chat_template(render(src, perm, ans if a.with_answer else None), tokenize=True, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)   # transformers 5：返回 BatchEncoding（不是 dict 子类）
        if len(ids) > a.max_len:   # 原文太长：从原文中间截掉（保留指令开头和选项结尾），实际训练集最长约 2000 token，基本不触发
            ids = ids[: a.max_len // 2] + ids[-(a.max_len - a.max_len // 2):]
        return ids

    def letter_logits(ids: list[int]) -> "torch.Tensor":
        x = torch.tensor([ids], device="cuda")
        return model(input_ids=x).logits[0, -1, lids].float()

    tr = data["train"] if not a.eval_only else []
    steps = max(1, a.epochs * math.ceil(len(tr) / a.batch))
    if not a.eval_only:   # 只评测时加载的 adapter 参数不可训练，建优化器会报空参数表
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * steps))) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    bce = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(a.seed)
    log = open(out / ("train_log.jsonl" if not a.eval_only else "eval_only.log"), "w", encoding="utf-8")
    step, t0 = 0, time.time()
    model.train()
    for ep in range(a.epochs if not a.eval_only else 0):
        order = list(range(len(tr)))
        rng.shuffle(order)
        for b in range(0, len(order), a.batch):
            tot = 0.0
            for i in order[b:b + a.batch]:
                r = tr[i]
                perm = list(MECHS)
                rng.shuffle(perm)
                y = torch.tensor([float(r["labels"][m]) for m in perm], device="cuda")
                loss = bce(letter_logits(encode(r["source"], perm, r.get("answer"))), y) / len(order[b:b + a.batch])
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps({"step": step, "epoch": ep, "loss": round(tot, 4), "lr": sched.get_last_lr()[0], "t": round(time.time() - t0, 1)}) + "\n")
            log.flush()
            if step % 10 == 0:
                print(json.dumps({"step": step, "of": steps, "loss": round(tot, 4)}), flush=True)
    if not a.eval_only:
        model.save_pretrained(str(out / "adapter"))

    model.eval()
    raw = {}
    with torch.no_grad():
        for n in ("holdout", "test", "policy"):
            rows = []
            for r in data[n]:
                acc = {m: 0.0 for m in MECHS}
                ps = perms_for(r["id"], a.eval_perms)
                for perm in ps:
                    z = letter_logits(encode(r["source"], perm, r.get("answer"))).tolist()
                    for m, v in zip(perm, z):
                        acc[m] += v / len(ps)
                rows.append({"id": r["id"], "logits": [round(acc[m], 4) for m in MECHS]})
            raw[n] = rows
    T = fit_temperature([x["logits"] for x in raw["holdout"]], [[r["labels"][m] for m in MECHS] for r in data["holdout"]])
    rep = {"temperature": T, "mechs": MECHS, "epochs": a.epochs, "lr": a.lr, "eval_perms": a.eval_perms, "with_answer": a.with_answer}
    for n, rows in raw.items():
        probs = [[1 / (1 + math.exp(-z / T)) for z in x["logits"]] for x in rows]
        with open(out / f"preds_{n}.jsonl", "w", encoding="utf-8") as f:
            for x, p in zip(rows, probs):
                f.write(json.dumps({**x, "probs": [round(v, 4) for v in p]}) + "\n")
        rep[n] = evaluate({m: [p[j] for p in probs] for j, m in enumerate(MECHS)}, data[n])
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("SUIT_DONE", json.dumps({n: (rep[n]["macro_auc"], rep[n]["macro_spearman_soft"], rep[n]["mean_brier"]) for n in raw}, default=float), flush=True)


if __name__ == "__main__":
    main()
