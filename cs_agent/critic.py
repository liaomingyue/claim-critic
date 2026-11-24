"""客服轨迹的两个奖励模型（Qwen3.5-9B + LoRA + 线性分类头），一次前向读整条轨迹：
  step     过程 critic：在每个客服步骤的结尾 token 读出 5 类（cs_agent.step_cls：OK / ERR / AFTER_ERR / FIX / TRY），
           步骤奖励 = Σ p(类) × 分值（step_cls.REWARD）。
  outcome  全流程 reward：在轨迹最后一个 token 读出 3 类：
           GOOD   数据库对、过程没有问题；
           FLAWED 数据库对，但过程有问题（违规 / 答错漏答 / 靠用户指出才改 / OVERFIX / 侥幸做对）；
           FAIL   数据库对不上。
           轨迹奖励 = Σ p(类) × 分值（OUTCOME_REWARD）。
token 序列直接用采样时存下的（prompt_ids + ids，和策略看到的完全一样）；每步结尾 = 该轮在 ids 里区间的最后一个 token。

  outcome2 全流程 reward 二分类版：只在数据库正确的轨迹上分 GOOD / FLAWED（GRPO 用的就是这一项，数据库对错用真实比对）。

  python -m cs_agent.critic train --task step|outcome|outcome2 --base <9B> --data <steps_cls_v1.jsonl> --eps <base9_train.jsonl> <sftB_train.jsonl> --out <目录>
  python -m cs_agent.critic eval  --task step|outcome --base <9B> --ckpt <目录> --data ... --eps ... --out <分数 jsonl> [--splits test claude]
显存：长轨迹反向放不下时（--max_len），步骤任务只保留前 max_len 个 token（之后的步骤不算），结果任务保留开头（系统提示）+ 结尾。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

from cs_agent.step_cls import CLASSES as STEP_CLASSES
from cs_agent.step_cls import REWARD as STEP_REWARD

OUT_CLASSES = ["GOOD", "FLAWED", "FAIL"]
OUTCOME_REWARD = {"GOOD": 1.0, "FLAWED": 0.3, "FAIL": 0.0}
# outcome2：只在数据库正确的轨迹上分过程好 / 有问题（数据库对不对 GRPO 用真实比对，不用模型猜）。
# 三分类版 FLAWED 的 AUC 只有 0.48：数据库对错这一大块差异占掉了模型的注意力，过程好坏没学出来
OUT2_CLASSES = ["GOOD", "FLAWED"]
OUTCOME2_REWARD = {"GOOD": 1.0, "FLAWED": 0.5}
FLAW = {"FIRST_ERR", "OVERFIX", "VIOLATION", "VIOLATION_S1", "INFO_ERR", "USER_CORRECTED"}


def outcome_label(r: dict) -> str:
    if not r["db_ok"]:
        return "FAIL"
    if r["meta"].get("lucky") or any(s["fine"] in FLAW for s in r["steps"]):
        return "FLAWED"
    return "GOOD"


def load_data(data: str, eps: list[str], splits: set[str], task: str, max_len: int) -> list[dict]:
    rows = {}
    for l in open(data, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["split"] in splits:
                rows[(r["policy"], r["episode_id"])] = r
    out = []
    for f in eps:
        pol = Path(f).name.split("_")[0]
        for l in open(f, encoding="utf-8"):
            try:
                e = json.loads(l)
            except ValueError:
                continue
            r = rows.get((pol, e.get("episode_id")))
            if r is None:
                continue
            P = len(e["prompt_ids"])
            ids = e["prompt_ids"] + e["ids"]
            ex = {"policy": pol, "episode_id": r["episode_id"], "task_id": r["task_id"], "split": r["split"], "n_tok": len(ids)}
            if task == "step":
                end = {t["msg_idx"]: P + t["b"] - 1 for t in e["turns"]}
                pos, lab, meta = [], [], []
                for s in r["steps"]:
                    p = end.get(s["idx"])
                    if p is None or not s["train_mask"] or p >= max_len:
                        continue
                    pos.append(p)
                    lab.append(STEP_CLASSES.index(s["cls"]))
                    meta.append({"step": s["step"], "fine": s["fine"]})
                if not pos:
                    continue
                ex.update(ids=ids[: max(pos) + 1], pos=pos, lab=lab, meta=meta)
            else:
                if task == "outcome2" and not r["db_ok"]:
                    continue
                if len(ids) > max_len:   # 保留系统提示 + 结尾
                    ids = e["prompt_ids"] + e["ids"][-(max_len - P):]
                cl = OUT2_CLASSES if task == "outcome2" else OUT_CLASSES
                ex.update(ids=ids, pos=[len(ids) - 1], lab=[cl.index(outcome_label(r))], meta=[{"db_ok": r["db_ok"]}])
            out.append(ex)
    return out


def build(base: str, n_cls: int, ckpt: str | None, train: bool, r: int = 32, alpha: int = 64):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if ckpt:
        lm = PeftModel.from_pretrained(lm, ckpt, is_trainable=train)
    else:
        lm = get_peft_model(lm, LoraConfig(r=r, lora_alpha=alpha, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"))
    inner = lm.base_model.model.model   # 只要隐状态，不用输出层
    head = torch.nn.Linear(lm.config.get_text_config().hidden_size if hasattr(lm.config, "get_text_config") else lm.config.hidden_size, n_cls).cuda().float()
    if ckpt:
        head.load_state_dict(torch.load(Path(ckpt) / "head.pt", map_location="cuda"))
    return lm, inner, head


def forward(inner, head, ids, pos):
    import torch
    x = torch.tensor([ids], device="cuda")
    h = inner(input_ids=x).last_hidden_state[0]
    return head(h[torch.tensor(pos, device="cuda")].float())


def evaluate(inner, head, data, classes, reward):
    import torch
    rows = []
    with torch.no_grad():
        for ex in data:
            p = torch.softmax(forward(inner, head, ex["ids"], ex["pos"]), -1).cpu().tolist()
            for k in range(len(ex["pos"])):
                rows.append({"policy": ex["policy"], "episode_id": ex["episode_id"], "split": ex["split"], **ex["meta"][k],
                             "gold": classes[ex["lab"][k]], "probs": dict(zip(classes, [round(v, 5) for v in p[k]])),
                             "reward": sum(reward[c] * v for c, v in zip(classes, p[k]))})
    return rows


def auc(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return None
    s = sorted([(v, 1) for v in pos_scores] + [(v, 0) for v in neg_scores])
    rank, i, tot = 0.0, 0, 0.0
    while i < len(s):   # 并列取平均秩
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        tot += avg * sum(1 for k in range(i, j) if s[k][1])
        i = j
    n1, n0 = len(pos_scores), len(neg_scores)
    return round((tot - n1 * (n1 + 1) / 2) / (n1 * n0), 4)


def metrics(rows, classes):
    out = {}
    for sp in sorted({r["split"] for r in rows}):
        R = [r for r in rows if r["split"] == sp]
        pred = [max(r["probs"], key=r["probs"].get) for r in R]
        conf = collections.Counter((r["gold"], p) for r, p in zip(R, pred))
        per = {}
        for c in classes:
            tp = conf[(c, c)]
            ng = sum(v for (g, _), v in conf.items() if g == c)
            npred = sum(v for (_, p), v in conf.items() if p == c)
            per[c] = {"n": ng, "recall": round(tp / ng, 3) if ng else None, "precision": round(tp / npred, 3) if npred else None,
                      "auc": auc([r["probs"][c] for r in R if r["gold"] == c], [r["probs"][c] for r in R if r["gold"] != c])}
        f1 = [2 * v["recall"] * v["precision"] / (v["recall"] + v["precision"]) for v in per.values() if v["recall"] and v["precision"]]
        out[sp] = {"n": len(R), "acc": round(sum(r["gold"] == p for r, p in zip(R, pred)) / max(1, len(R)), 4),
                   "macro_f1": round(sum(f1) / len(classes), 4), "per_class": per,
                   "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items()) if g != p and v >= 3}}
    return out


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--task", choices=["step", "outcome", "outcome2"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--splits", nargs="+", default=["test", "claude"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--weight_pow", type=float, default=0.5, help="类别权重 = (总数 / 该类数)^pow，再归一化；0 = 不加权")
    ap.add_argument("--eval_every", type=int, default=0, help="每多少步在留出集上评一次（画 AUC 曲线）；0 = 只在 epoch 末评")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    classes, reward = {"step": (STEP_CLASSES, STEP_REWARD), "outcome": (OUT_CLASSES, OUTCOME_REWARD), "outcome2": (OUT2_CLASSES, OUTCOME2_REWARD)}[a.task]
    out = Path(a.out)
    if a.mode == "eval":
        data = load_data(a.data, a.eps, set(a.splits), a.task, max_len=10 ** 9)   # 评测不截断（不反向，显存够）
        lm, inner, head = build(a.base, len(classes), a.ckpt, train=False)
        lm.eval()
        rows = evaluate(inner, head, data, classes, reward)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("CRITIC_EVAL", a.task, json.dumps(metrics(rows, classes), ensure_ascii=False), flush=True)
        return
    data = load_data(a.data, a.eps, {"train"}, a.task, a.max_len)
    data = data[: a.limit] if a.limit else data
    dev = load_data(a.data, a.eps, {"test"}, a.task, max_len=10 ** 9)
    cnt = collections.Counter(l for ex in data for l in ex["lab"])
    N = sum(cnt.values())
    w = [((N / cnt[k]) ** a.weight_pow if cnt[k] else 0.0) for k in range(len(classes))]
    w = [v / (sum(w) / len(w)) for v in w]
    print(f"CRITIC {a.task} train {len(data)} trajectories, {N} labels {dict((classes[k], v) for k, v in sorted(cnt.items()))} "
          f"weights {dict(zip(classes, [round(v, 2) for v in w]))}; dev {len(dev)}", flush=True)
    lm, inner, head = build(a.base, len(classes), None, train=True)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.enable_input_require_grads()
    lm.train()
    lora = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.0)
    total = math.ceil(len(data) / a.accum) * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wt = torch.tensor(w, device="cuda")
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for b in range(0, len(order), a.accum):
            chunk = [data[i] for i in order[b:b + a.accum]]
            n_lab = sum(len(ex["lab"]) for ex in chunk)
            tot = 0.0
            for ex in chunk:
                logits = forward(inner, head, ex["ids"], ex["pos"])
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ex["lab"], device="cuda"), weight=wt, reduction="sum") / n_lab
                loss.backward()
                tot += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(lora + list(head.parameters()), 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"step {step}/{total} ep {ep} loss {tot:.4f} gn {gn:.2f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            if a.eval_every and step % a.eval_every == 0:
                lm.eval()
                m = metrics(evaluate(inner, head, dev, classes, reward), classes)
                lm.train()
                print(f"DEV step {step} {json.dumps(m, ensure_ascii=False)}", flush=True)
        d = out / f"epoch{ep + 1}"
        lm.save_pretrained(str(d))
        torch.save(head.state_dict(), d / "head.pt")
        lm.eval()
        m = metrics(evaluate(inner, head, dev, classes, reward), classes)
        lm.train()
        print(f"EPOCH {ep + 1} saved -> {d}; dev {json.dumps(m, ensure_ascii=False)}", flush=True)
    lm.save_pretrained(str(out))
    torch.save(head.state_dict(), out / "head.pt")
    (out / "critic_manifest.json").write_text(json.dumps({**vars(a), "classes": classes, "reward": reward, "train_traj": len(data),
                                                          "labels": {classes[k]: v for k, v in cnt.items()}, "weights": w, "steps": step}, indent=1), encoding="utf-8")
    print("CRITIC_DONE", a.task, out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
