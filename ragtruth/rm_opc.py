"""事后分类：一处改动造出来的错，实际是清单里的哪一种操作（不看当初要求的是什么）。

opm__v1 回答的是「做的是不是要求的那种操作」，依赖要求，量不了整批样本的分布，也用不到模型自己犯的错上。
这里分两步判，都是 2B 单字母分类器（一次最多读 20 个选项的概率，二十种操作加「不是错」放不进一步）：
  opl__v1  哪一层：A 值 / B 挂接 / C 限定 / D 句间 / E 不是错或清单里没有
  opc__v1  层内是哪一种操作：给定层，列出这一层的操作，选一个字母
标签都是 Claude 独立判的操作（核实时它看不到要求的操作）：
  按操作清单核实的合成样本（analysis/ops_verify*，含 GRPO 训练前后、阶段三随机抽查和盲区抽查）
  种子错误对（模型自己写错的句子和改对后的句子，analysis/ops_label）：只取单一操作的
切分按原文文本哈希 70 / 10 / 20，和 rm_ops 相同。

  python -m ragtruth.rm_opc      # → outputs/rt/rm/{opl__v1, opc__v1}/{train,calib,test}.jsonl
"""
import collections
import gzip
import json
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.core import hsplit, write_rm
from ragtruth.error_ops import OPS
from ragtruth.gen_ops import examples, op_of
from ragtruth.rm_ops import OPSETS, row, skey
from ragtruth.sev_gen import changes

RT = Path("outputs/rt")
LAYER_OPS = {k: [o for o in OPS if OPS[o]["layer"] == k] for k in ("value", "binding", "qualifier", "between")}
LAYER_LETTER = {"value": "A", "binding": "B", "qualifier": "C", "between": "D"}
LAYER_DESC = {"value": "a word, number, detail or outcome is replaced by, or added as, content that the source does not contain",
              "binding": "every piece of content comes from the source, but it is attached to the wrong thing: wrong speaker, owner, referent, direction, number, time, place or event",
              "qualifier": "the degree of certainty, the scope or a condition is changed: a hedge, a report marker, a plan, a quantifier or a restriction is dropped, added or altered",
              "between": "the relation between statements is changed: a causal, contrastive or temporal link, or a conclusion, that the source does not state"}
OPL_SYSTEM = ("One sentence of an assistant's response was edited. You see the source (the exact input the assistant received), the response with the edited sentence in place, "
              "the ORIGINAL and the EDITED version of that sentence, and the word-level changes. Decide what kind of error the edit introduced, judged against the source.\n\nOptions:\n"
              + "\n".join(f"{LAYER_LETTER[k]}. {k}: {LAYER_DESC[k]}." for k in LAYER_LETTER)
              + "\nE. none: the edited sentence is still faithful to the source, or the error fits none of the above.\n\nAnswer with a single letter: A, B, C, D or E.")


def user(x: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["context"], 1))
    return (f"<source>\n{x['source']}\n</source>\n\n<response>\n{sents}\n</response>\n\n"
            f"Edited sentence: S{x['idx']}\nORIGINAL: {x['original']}\nEDITED: {x['sentence']}\n{changes(x['original'], x['sentence'])}")


def opl_prompt(x: dict) -> list[dict]:
    return [{"role": "system", "content": OPL_SYSTEM}, {"role": "user", "content": user(x)}]


def opc_prompt(x: dict, layer: str) -> list[dict]:
    ops = LAYER_OPS[layer]
    system = ("One sentence of an assistant's response was edited so that it is no longer faithful to the source. You see the source, the response with the edited sentence in place, "
              "the ORIGINAL and the EDITED version of that sentence, and the word-level changes. "
              f"The error is of the kind '{layer}': {LAYER_DESC[layer]}. Decide which operation the edit performed.\n\nOptions:\n"
              + "\n".join(f"{chr(65 + k)}. {o}: {OPS[o]['desc']}." for k, o in enumerate(ops))
              + f"\n\nAnswer with a single letter: {', '.join(chr(65 + k) for k in range(len(ops)))}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user(x)}]


def load() -> list[dict]:
    """全部带「实际操作」标注的样本：uid、set、source、context、idx、original、sentence、op（清单里的操作 / none / other）、split。"""
    rows = []

    def sp(src: str) -> str:
        return ("train", "calib", "test")[hsplit(skey(src), "rm_ops", (70, 80))]

    def add(uid: str, st: str, x: dict, sev: str, op: str) -> None:
        rows.append({"uid": uid, "set": st, "source": x["source"], "context": x["context"], "idx": int(x["idx"]), "original": x["original"], "sentence": x["sentence"],
                     "op": op if sev in ("clear", "minor") else "none", "split": sp(x["source"]), "task_type": x.get("task_type", "")})

    sets = {**OPSETS, "ops_verify_ev_st3": "samples_ops_ev_st3.jsonl", "ops_verify_ev_st3_zone": "samples_ops_ev_st3.jsonl"}
    for sub, f in sets.items():
        S = {x["id"]: x for x in read_jsonl(RT / "core" / f) if x["why"] == "ok"}
        for r in read_jsonl(RT / "analysis" / sub / "labels.jsonl"):
            add(f"{sub}/{r['id']}", sub, S[r["id"]], r["c"]["severity"], r["c"]["op"])
    # GRPO 阶段一训练前后的核实：样本在训练过程存下的文件里，编号 = <窗口>-<步>-<行>
    PR = {r["id"]: r for r in read_jsonl(RT / "gen_ops_grpo/prompts.jsonl")}
    L = {r["id"]: r for r in read_jsonl(RT / "analysis/ops_verify_grpo_s1/labels.jsonl")}
    for step in sorted({int(i.split("-")[1]) for i in L}):
        for j, line in enumerate(gzip.open(RT / f"gen_ops_grpo/s1/samples/step{step}.jsonl.gz", "rt", encoding="utf-8")):
            for win in ("start", "end"):
                r = L.get(f"{win}-{step}-{j}")
                if r:
                    x = json.loads(line)
                    p = PR[x["prompt_id"]]
                    ctx = list(p["sentences"])
                    ctx[int(x["idx"]) - 1] = x["sentence"]
                    add(f"grpo_s1/{r['id']}", "ops_verify_grpo_s1", {"source": p["source"], "context": ctx, "idx": x["idx"], "original": x["original"], "sentence": x["sentence"]},
                        r["c"]["severity"], r["c"]["op"])
    # 种子错误对：模型自己写错的句子。给生成器当例句的那些不用
    lab = {r["id"]: r["o"] for r in read_jsonl(RT / "analysis/ops_label/pairs_labels.jsonl")}
    used = {e["id"] for xs in examples().values() for e in xs}
    for p in read_jsonl(RT / "core/pairs.jsonl"):
        op = op_of(lab[p["id"]]) if p["id"] in lab else None
        if not op or p["id"] in used:
            continue
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        ctx = list(sents)
        ctx[int(p["idx"]) - 1] = p["real"]
        rows.append({"uid": f"seed/{p['id']}", "set": "seed_pairs", "source": p["source"], "context": ctx, "idx": int(p["idx"]), "original": p["repaired"], "sentence": p["real"], "op": op,
                     "split": "train" if p["split"] == "train" else sp(p["source"]), "task_type": p["task_type"]})
    return rows


def main() -> None:
    rows = load()
    print("labelled", len(rows), dict(collections.Counter(r["set"] for r in rows)))
    print("ops", dict(collections.Counter(r["op"] for r in rows).most_common()))
    pl, pc = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        layer = OPS[r["op"]]["layer"] if r["op"] in OPS else None
        pl[r["split"]].append(row("opl/" + r["uid"], r, opl_prompt(r), LAYER_LETTER.get(layer, "E")))
        if layer:
            pc[r["split"]].append({**row("opc/" + r["uid"], r, opc_prompt(r, layer), chr(65 + LAYER_OPS[layer].index(r["op"]))), "layer": layer, "op": r["op"]})
    for name, parts in (("opl__v1", pl), ("opc__v1", pc)):
        m = write_rm(name, parts, 4096)
        print(name.upper(), json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), {k: m[k]["label"] for k in ("train", "test")}, "too long", m["too_long_dropped"])


if __name__ == "__main__":
    main()
