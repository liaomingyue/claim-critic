"""给错误合成把关的三个 2B 模型的训练数据（用上全部 Claude 标注，含之前没进训练的和按操作清单核实的）：
  sev__v4   改动是不是真错、有多严重（A 明显错 / B 轻微错 / C 不是错）。不再同时判「是不是要求的那一类」，所有标注都能用。
  rm2__v4   改完的句子自不自然（A 自然 / B 不自然）。
  opm__v1   改动做的是不是指定的那种操作（A 是，并且造成了错 / B 不是）。给操作卡片的做法、适用条件、为什么算错。

标注来源（outputs/rt 下，每个目录 items.jsonl + labels.jsonl）：
  旧配方里已经用的：core/gen_v2_label、gen_v2_label_r2（按原文 70 / 10 / 20，切分和 sev__v3 / rm2__v3 相同，旧测试集不变）、
                    core/gen_v2_label_t1、gen_v2_label_t2、sevaug_r1（只进训练）
  之前没用的：gen_v2_grpo/blind_fill_r2（critic 拿不准区域的第二轮标注）、core/scope_label、gen_v2_label_r3、gen_v2_label_r4、blind_check
  按操作清单核实的：analysis/ops_verify、ops_verify_pplan_mc、ops_verify_pplan_9b（要求的操作、实际的操作、严重度、自然度）
后两组按原文文本哈希 70 / 10 / 20。只进训练的样本里，原文出现在任何校准 / 测试样本里的去掉，防止同一篇原文两边都有。
opm 另加种子错误对（core/pairs.jsonl + analysis/ops_label/pairs_labels.jsonl）：正例 = 标的操作，负例 = 随机另一种操作。

  python -m ragtruth.rm_ops      # → outputs/rt/rm/{sev__v4, rm2__v4, opm__v1}/{train,calib,test}.jsonl
"""
import ast
import collections
import hashlib
import json
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.core import hsplit, write_rm
from ragtruth.error_ops import CARD, OPS
from ragtruth.gen_ops import examples, op_of
from ragtruth.rm2_v3 import prompt as nat_prompt
from ragtruth.sev_gen import changes, prompt_dne

RT = Path("outputs/rt")
OLD_SPLIT = ["core/gen_v2_label", "core/gen_v2_label_r2"]
OLD_TRAIN = ["core/gen_v2_label_t1", "core/gen_v2_label_t2", "core/sevaug_r1"]
NEW = ["gen_v2_grpo/blind_fill_r2", "core/scope_label", "core/gen_v2_label_r3", "core/gen_v2_label_r4", "core/blind_check"]
OPSETS = {"ops_verify": "samples_ops_mc.jsonl", "ops_verify_pplan_mc": "samples_ops_pplan_mc.jsonl", "ops_verify_pplan_9b": "samples_ops_pplan_9b.jsonl"}

OPM_SYSTEM = """You check whether an edit to one sentence of a response performed a REQUESTED edit operation. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the ORIGINAL and the EDITED version of that sentence, the word-level changes, and the requested operation: what it does, where it can be done, and why its result is an error.

Options:
A. yes: the edit performs the requested operation, and the edited sentence is now wrong for the stated reason.
B. no: the edit does something else, or it leaves the sentence faithful to the source.

Answer with a single letter: A or B."""


def opm_prompt(x: dict, op: str) -> list[dict]:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["context"], 1))
    need, wrong = CARD[op]
    u = (f"<source>\n{x['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
         f"Edited sentence: S{x['idx']}\nORIGINAL: {x['original']}\nEDITED: {x['sentence']}\n{changes(x['original'], x['sentence'])}\n\n"
         f"Requested operation: {op}: {OPS[op]['desc']}.\nWhere it can be done: {need}.\nWhy the result is an error: {wrong}.")
    return [{"role": "system", "content": OPM_SYSTEM}, {"role": "user", "content": u}]


def skey(source: str) -> str:
    return hashlib.sha256(" ".join(source.split()).encode()).hexdigest()[:16]


def load() -> list[dict]:
    """全部标注统一成：uid、set、source、context、idx、original、sentence、severity、natural、split、（want、actual：只有按操作清单核实的有）。"""
    rows = []
    for sub in OLD_SPLIT + OLD_TRAIN + NEW:
        it = {x["aid"]: x for x in read_jsonl(RT / sub / "items.jsonl")}
        for r in read_jsonl(RT / sub / "labels.jsonl"):
            c = r["claude"]
            c = ast.literal_eval(c) if isinstance(c, str) else c
            x = it.get(r["aid"])
            if not x or not isinstance(c, dict):
                continue
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            if sub in OLD_SPLIT:   # 和 sev__v3 / rm2__v3 一样的切分键（两者盐不同，各自在建数据时算），这里先记键
                sp = None
            elif sub in OLD_TRAIN:
                sp = "train_only"
            else:
                sp = ("train", "calib", "test")[hsplit(skey(x["source"]), "rm_ops", (70, 80))]
            rows.append({"uid": f"{sub.split('/')[-1]}/{r['aid']}", "set": sub.split("/")[-1], "source": x["source"], "context": ctx, "idx": int(x["idx"]), "original": x["original"],
                         "sentence": x["sentence"], "severity": c.get("severity"), "natural": c.get("natural"), "split": sp, "old_key": x.get("id", r["aid"]).split("#")[0],
                         "task_type": x.get("task_type", "")})
    for sub, f in OPSETS.items():
        S = {x["id"]: x for x in read_jsonl(RT / "core" / f) if x["why"] == "ok"}
        for r in read_jsonl(RT / "analysis" / sub / "labels.jsonl"):
            x = S[r["id"]]
            rows.append({"uid": f"{sub}/{r['id']}", "set": sub, "source": x["source"], "context": x["context"], "idx": int(x["idx"]), "original": x["original"], "sentence": x["sentence"],
                         "severity": r["c"]["severity"], "natural": r["c"]["natural"], "want": r["want"], "actual": r["c"]["op"],
                         "split": ("train", "calib", "test")[hsplit(skey(x["source"]), "rm_ops", (70, 80))], "task_type": x.get("task_type", "")})
    return rows


def assign(rows: list[dict], salt: str) -> list[dict]:
    """定下每条的切分：旧的两组按各自原来的盐；只进训练的那些，原文在校准 / 测试里出现过的去掉。"""
    out = []
    for r in rows:
        sp = r["split"] or ("train", "calib", "test")[hsplit(r["old_key"], salt, (70, 80))]
        out.append({**r, "split": sp})
    held = {skey(r["source"]) for r in out if r["split"] in ("calib", "test")}
    return [r if r["split"] in ("calib", "test") else {**r, "split": "train"} for r in out if r["split"] in ("calib", "test") or skey(r["source"]) not in held]


def row(uid: str, r: dict, prompt: list[dict], lab: str) -> dict:
    return {"id": uid, "prompt": prompt, "completion": lab, "label": lab, "task_type": r["task_type"], "group": "src-" + skey(r["source"]), "origin": r["set"]}


def main() -> None:
    rows = load()
    print("labelled items", len(rows), dict(collections.Counter(r["set"] for r in rows)))
    # sev
    parts = collections.defaultdict(list)
    for r in assign(rows, "sev_v3"):
        lab = {"clear": "A", "minor": "B", "none": "C"}.get(r["severity"])
        if lab:
            parts[r["split"]].append(row("sev4/" + r["uid"], r, prompt_dne(r), lab))
    m = write_rm("sev__v4", parts, 4096)
    print("SEV_V4", json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), {k: m[k]["label"] for k in ("train", "test")}, "too long", m["too_long_dropped"])
    # 自然度
    parts = collections.defaultdict(list)
    for r in assign(rows, "rm2_v3"):
        if r["natural"] in ("yes", "no"):
            parts[r["split"]].append(row("rm2v4/" + r["uid"], r, nat_prompt(r), "A" if r["natural"] == "yes" else "B"))
    m = write_rm("rm2__v4", parts, 3072)
    print("RM2_V4", json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), {k: m[k]["label"] for k in ("train", "test")}, "too long", m["too_long_dropped"])
    # 判操作
    parts = collections.defaultdict(list)
    ops = list(OPS)
    for r in assign([r for r in rows if "want" in r], "x"):
        real = r["severity"] in ("clear", "minor")
        parts[r["split"]].append(row(f"opm/{r['uid']}|{r['want']}", r, opm_prompt(r, r["want"]), "A" if real and r["actual"] == r["want"] else "B"))
        if real and r["actual"] in OPS and r["actual"] != r["want"]:   # 实际做成的那种操作：补一条正例
            parts[r["split"]].append(row(f"opm/{r['uid']}|{r['actual']}", r, opm_prompt(r, r["actual"]), "A"))
    lab = {r["id"]: r["o"] for r in read_jsonl(RT / "analysis/ops_label/pairs_labels.jsonl")}
    used = {e["id"] for xs in examples().values() for e in xs}
    for p in read_jsonl(RT / "core/pairs.jsonl"):
        op = op_of(lab[p["id"]]) if p["id"] in lab else None
        if not op or p["id"] in used:
            continue
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        ctx = list(sents)
        ctx[int(p["idx"]) - 1] = p["real"]
        r = {"source": p["source"], "context": ctx, "idx": int(p["idx"]), "original": p["repaired"], "sentence": p["real"], "task_type": p["task_type"], "set": "seed_pairs"}
        # 种子集留出部分的原文就是采样用的那批留出回答的原文：和上面用同一个哈希切分，同一篇原文不会两边都有
        sp = "train" if p["split"] == "train" else ("train", "calib", "test")[hsplit(skey(p["source"]), "rm_ops", (70, 80))]
        neg = [o for o in ops if o != op and OPS[o]["layer"] != OPS[op]["layer"]]   # 负例取别的层的操作：同层的几种边界本来就近，标签不可靠
        other = neg[hsplit(p["id"], "opm_neg", tuple(range(1, 100))) % len(neg)]
        parts[sp].append(row(f"opm/seed/{p['id']}|{op}", r, opm_prompt(r, op), "A"))
        parts[sp].append(row(f"opm/seed/{p['id']}|{other}", r, opm_prompt(r, other), "B"))
    m = write_rm("opm__v1", parts, 4096)
    print("OPM_V1", json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), {k: m[k]["label"] for k in ("train", "test")}, {k: m[k]["origin"] for k in ("train", "test")}, "too long", m["too_long_dropped"])


if __name__ == "__main__":
    main()
