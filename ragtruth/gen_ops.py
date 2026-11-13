"""按错误操作清单（ragtruth/error_ops.py）建错误合成模型的训练数据。

和 gen_v2 的区别只在指令：原来给 9 个机制名之一，现在给 20 种操作之一（名字 + 说明）。输入输出格式不变：
给原文和整段回答，模型自己挑一句来改，输出改第几句、改写后的句子、一句改动描述。

样本来源：种子错误对（outputs/rt/core/pairs.jsonl：真实模型写错的句子 + 改对的句子）+ 每条的操作标签
（outputs/rt/analysis/ops_label/pairs_labels.jsonl，Claude 对着原文标）。只留单一操作的样本：
一条错误同时做了两种操作时，拿它教「只做这一种」信号是混的。有前提规则的操作，改对的句子不满足规则的也去掉，
训练时和使用时派任务的条件保持一致。

  python -m ragtruth.gen_ops build     # → outputs/rt/core/seed_ops_{train,holdout}.jsonl
"""
import argparse
import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.error_ops import OPS, PRE
from ragtruth.gen_v2 import INSTR, describe, edits, norm

D = Path("outputs/rt/core")
LABELS = Path("outputs/rt/analysis/ops_label/pairs_labels.jsonl")


def prompt(source: str, sentences: list[str], op: str) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(sentences, 1))
    return INSTR.format(mech_name=op, mech_desc=OPS[op]["desc"], source=source.strip(), sentences=sents)


def op_of(o: dict) -> str | None:
    """主操作。标注时清单里还没有 add_hedge，这类被标成了 other 并在说明里写了加推测词，按说明归过去。"""
    if o["op"] == "other":
        return "add_hedge" if "hedge" in o["why_other"].lower() and "add" in o["why_other"].lower() else None
    return o["op"] if o["op2"] == "none" else None


def cmd_build(a) -> None:
    lab = {r["id"]: r["o"] for r in read_jsonl(LABELS)}
    desc = {r["id"]: r["desc"] for r in read_jsonl(D / a.desc) if r.get("ok")} if a.desc else {}
    out, why = collections.defaultdict(list), collections.Counter()
    for p in read_jsonl(D / "pairs.jsonl"):
        if p["id"] not in lab:
            why["no_label"] += 1
            continue
        op = op_of(lab[p["id"]])
        if op is None:
            why["two_ops" if lab[p["id"]]["op"] != "other" else "other"] += 1
            continue
        if norm(p["repaired"]) == norm(p["real"]):
            why["no_edit"] += 1
            continue
        if OPS[op]["pre"] and not PRE[OPS[op]["pre"]](p["repaired"], p["source"]):
            why["pre_fail"] += 1
            continue
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        pr = prompt(p["source"], sents, op)
        comp = json.dumps({"idx": int(p["idx"]), "sentence": p["real"], "edit": desc.get(p["id"]) or describe(p["repaired"], p["real"])}, ensure_ascii=False)
        if len(pr) + len(comp) > a.max_chars:
            why["too_long"] += 1
            continue
        out[p["split"]].append({"id": p["id"], "task": "rewrite", "source_id": p["source_id"], "task_type": p["task_type"], "severity": p["severity"], "op": op, "layer": OPS[op]["layer"],
                                "mechanism": p["mechanism"], "idx": int(p["idx"]), "n_edits": len(edits(p["repaired"], p["real"]) or []), "prompt": pr, "completion": comp})
    random.Random("core|seed_ops").shuffle(out["train"])
    for sp in ("train", "holdout"):
        (D / f"seed_ops_{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[sp]), encoding="utf-8")
    print("SEED_OPS", json.dumps({"train": len(out["train"]), "holdout": len(out["holdout"]), "dropped": dict(why)}, ensure_ascii=False))
    for sp in ("train", "holdout"):
        c, cl = collections.Counter(r["op"] for r in out[sp]), collections.Counter(r["op"] for r in out[sp] if r["severity"] == "clear")
        print(sp, " ".join(f"{k}:{c[k]}({cl[k]})" for k in OPS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--desc", default="", help="改动描述文件（outputs/rt/core 下，{id, desc, ok}）；没有的用规则模板")
    ap.add_argument("--max_chars", type=int, default=14000)
    cmd_build(ap.parse_args())


if __name__ == "__main__":
    main()
