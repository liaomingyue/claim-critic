"""合成适合度模型：2B + LoRA，输入（原文, 回答, 一个错误类型），在作答位置读 Yes / No 两个词的 logit 之差，二分类交叉熵。

每条样本只问一个类型：没试过的（回答, 类型）不进训练，一条回答适合多个类型时各自是独立的一条样本。
评测只用 holdout 里真实尝试过的样本（kind = requested），和「只看类型的平均成功率」比：整体 AUC、每个类型内部的 AUC、
每个类型内按分数取前 1/3 的成功率。

  python -m train.synth_suit_train --base <MiniCPM5-2B> --out <目录> [--data outputs/rt/synth_suit --epochs 2 --lr 1e-4]
  → <目录>/adapter，<目录>/preds_holdout.jsonl，<目录>/report.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

DEFS = {
    "certainty_or_scope_shift": "certainty or scope shift: the sentence states something more (or less) certain, general or absolute than the source does (e.g. \"may\" becomes \"will\", \"some\" becomes \"all\")",
    "near_synonym_shift": "near-synonym shift: a word or phrase is replaced by a similar-looking one that changes the meaning",
    "contradicts_source": "contradicts the source: the sentence states something the source directly contradicts",
    "added_specific_fact": "added specific fact: the sentence adds a concrete fact or detail the source does not contain",
    "unsupported_inference": "unsupported inference: the sentence draws a conclusion, cause or consequence the source does not support",
    "wrong_attribution": "wrong attribution: a statement, action, quote or number is assigned to the wrong person, entity or place",
    "vague_evaluation_or_addon": "vague evaluation or add-on: the sentence adds a generic judgement, opinion or filler claim not in the source",
    "number_or_date": "number or date error: a number, quantity, date or time is wrong",
    "merged_across_passages": "merged across passages: facts from different passages or parts of the source are combined into one wrong claim",
}
SYSTEM = """You will see the exact input an assistant received (instructions plus source material) and the answer it wrote. An editor will be asked to rewrite one sentence of this answer, chosen by the editor, so that it contains one specific kind of faithfulness error while still reading like a natural answer.

Error kind: {definition}

Does this answer, together with this source, offer a good place for exactly this kind of error? Answer Yes or No."""


def render(source: str, answer: str, mech: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM.format(definition=DEFS[mech])}, {"role": "user", "content": f"[Input]\n{source}\n\n[Answer]\n{answer}"}]


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(r for r, y in zip(rank, labels) if y) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(rows: list[dict], scores: list[float], prior: dict[str, float]) -> dict:
    ys = [r["label"] for r in rows]
    rep = {"n": len(rows), "pos_rate": round(sum(ys) / len(ys), 4), "auc": auc(scores, ys), "auc_type_prior": auc([prior[r["mech"]] for r in rows], ys), "by_mech": {}}
    within, top = [], []
    for m in DEFS:
        ix = [i for i, r in enumerate(rows) if r["mech"] == m]
        if len(ix) < 10:
            continue
        s, y = [scores[i] for i in ix], [ys[i] for i in ix]
        k = max(1, len(ix) // 3)
        best = sorted(range(len(ix)), key=lambda i: -s[i])[:k]
        worst = sorted(range(len(ix)), key=lambda i: s[i])[:k]
        rep["by_mech"][m] = {"n": len(ix), "pos_rate": round(sum(y) / len(y), 4), "auc": auc(s, y), "top_third": round(sum(y[i] for i in best) / k, 4), "bottom_third": round(sum(y[i] for i in worst) / k, 4)}
        if rep["by_mech"][m]["auc"] is not None:
            within.append(rep["by_mech"][m]["auc"])
        top.append((rep["by_mech"][m]["top_third"] - rep["by_mech"][m]["pos_rate"], len(ix)))
    rep["macro_auc_within_type"] = sum(within) / len(within) if within else None
    rep["top_third_lift_weighted"] = sum(d * n for d, n in top) / sum(n for _, n in top) if top else None
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="outputs/rt/synth_suit")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8, help="每次更新的样本数（逐条前向、梯度累积）")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条训练 / 评测（冒烟测试）")
    a = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {n: [json.loads(l) for l in open(Path(a.data) / f"{n}.jsonl", encoding="utf-8")] for n in ("train", "holdout")}
    hold = [r for r in data["holdout"] if r["kind"] == "requested"]
    tr = data["train"]
    if a.limit:
        tr, hold = tr[:a.limit], hold[:a.limit]
    req = [r for r in data["train"] if r["kind"] == "requested"]
    prior = {m: sum(r["label"] for r in req if r["mech"] == m) / max(1, sum(r["mech"] == m for r in req)) for m in DEFS}

    tok = AutoTokenizer.from_pretrained(a.base)
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    assert len(yes) == 1 and len(no) == 1 and yes != no, (yes, no)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))

    def encode(r: dict) -> list[int]:
        ids = tok.apply_chat_template(render(r["source"], r["answer"], r["mech"]), tokenize=True, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if len(ids) > a.max_len:   # 太长时从原文中间截（保留开头的类型定义和结尾的回答）
            ids = ids[: a.max_len // 2] + ids[-(a.max_len - a.max_len // 2):]
        return ids

    def logit(r: dict) -> "torch.Tensor":
        z = model(input_ids=torch.tensor([encode(r)], device="cuda")).logits[0, -1].float()
        return z[yes[0]] - z[no[0]]

    steps = max(1, a.epochs * math.ceil(len(tr) / a.batch))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * steps))) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    bce = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(a.seed)
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    step, t0 = 0, time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(tr)))
        rng.shuffle(order)
        for b in range(0, len(order), a.batch):
            tot, chunk = 0.0, order[b:b + a.batch]
            for i in chunk:
                loss = bce(logit(tr[i]), torch.tensor(float(tr[i]["label"]), device="cuda")) / len(chunk)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps({"step": step, "epoch": ep, "loss": round(tot, 4), "t": round(time.time() - t0, 1)}) + "\n")
            log.flush()
            if step % 20 == 0:
                print(json.dumps({"step": step, "of": steps, "loss": round(tot, 4), "t": round(time.time() - t0)}), flush=True)
    model.save_pretrained(str(out / "adapter"))

    model.eval()
    with torch.no_grad():
        scores = [float(logit(r)) for r in hold]
    with open(out / "preds_holdout.jsonl", "w", encoding="utf-8") as f:
        for r, s in zip(hold, scores):
            f.write(json.dumps({"id": r["id"], "source_id": r["source_id"], "resp_id": r["resp_id"], "mech": r["mech"], "label": r["label"], "logit": round(s, 4)}) + "\n")
    rep = {"epochs": a.epochs, "lr": a.lr, "train_rows": len(tr), "type_prior": prior, "holdout": evaluate(hold, scores, prior)}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    h = rep["holdout"]
    print("SYNTH_SUIT_DONE", json.dumps({k: h[k] for k in ("n", "pos_rate", "auc", "auc_type_prior", "macro_auc_within_type", "top_third_lift_weighted")}), flush=True)


if __name__ == "__main__":
    main()
