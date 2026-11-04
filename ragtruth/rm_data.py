"""三个 RM 的单字符选项数据（2026-10-30，用户：RM 只输出选项概率，概率要和真实概率一致；三个 9B LoRA 并行训）。

  python -m ragtruth.rm_data            # → outputs/rt/rm/<rm1|rm2|rm3>/{train,calib,test}.jsonl + meta.json

每行：id、prompt（system + user 消息）、completion（一个选项字母）、label（选项）、以及评测用的字段。
- RM1 有没有幻觉：A 无幻觉 / B 冲突 / C 无依据，P(有幻觉) = P(B) + P(C)。输入同 critic（原文 + 整条回答 + 要查的句子）。
  训练 = critic-clean-gfree 的数据（clean_gfree.jsonl），按 RM3 标签过滤合成样本：没改出错的改写、原句本身错的配对原句去掉。
  校准 = 从同一份数据按回答留出 10%（不用 val.jsonl：原标注漏标多，按它校准会把幻觉概率压低）。
  测试 = RAGTruth++（2,306 句，++ 标签；带审核批次和 Claude 结论，便于分批算）。
- RM2 自然还是合成：A 自然 / B 合成。数据 = D1（outputs/rt/discrim），训练里按原文留出 10% 作校准，留出集作测试。
- RM3 合成样本是否合格：A 合格 / B 解释不对 / C 没改出错 / D 原句本身错，P(合格) = P(A)。
  训练 = rm3_train.jsonl（高价值样本重复一次 ≈ 权重 2），按回答留出 10% 作校准；测试 = rm3_eval.jsonl（全部 Claude 标签）。
TRL 截断会切掉答案，所以训练集里估计超长（按 3.2 字符 / token）的样本直接丢掉；校准、测试是推理，全部保留。全部遍历排序、固定种子。
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth import audit_synth, discrim
from ragtruth.task import TIPS, expand_claims, render_input

OUT = Path("outputs/rt/rm")
CLOSED = Path("outputs/rt/closed")
MAX_TOKENS = {"rm1": 3072, "rm2": 3072, "rm3": 4096, "rm2v2": 3072, "rm2v2__mc": 3072, "rm3__mc": 4096}

RM1_SYSTEM = """You check whether one sentence of an assistant's response is faithful to the source material the assistant was given.

The source is the exact input the assistant received (instructions plus material). The full response is shown for context, split into numbered sentences; you check only the sentence given under [sentence to check]. Judge only against the source, not against world knowledge.

Options:
A. faithful: the sentence is supported by the source, or has no factual content.
B. conflict: the sentence contradicts the source.
C. baseless: the sentence adds information that the source does not contain, even if it may be true in the real world.

""" + TIPS + """

Answer with a single letter: A, B or C."""

RM2_SYSTEM = """You see a source (the exact input an assistant received) and one sentence that an assistant wrote with this source, with the sentence before and after it for context. The sentence is known to be unfaithful to the source: it contradicts the source or adds information the source does not contain.

Where does the error come from?
A. natural: the assistant made this mistake on its own while writing its answer.
B. synthetic: a faithful sentence was deliberately edited by someone to introduce the error.

Answer with a single letter: A or B."""

# 定向造错用的 RM2（rm2v2__mcgen，2026-11-03）：自然一侧是 2B 回答模型写的任意句子（用户：不是错误的也行，不然容易走捷径），
# 所以不再假定句子有错，只问这句是回答模型自己写的，还是事后被改出来、埋了错误的。
RM2G_SYSTEM = """You see a source (the exact input an assistant received) and one sentence from the assistant's response, with the sentence before and after it for context.

Was this sentence written by the assistant itself while writing its answer, or was it edited afterwards by someone who deliberately planted an error in it?
A. natural: the assistant wrote this sentence itself.
B. synthetic: someone edited this sentence afterwards to plant an error.

Answer with a single letter: A or B."""

RM1_OPT = {"faithful": "A", "conflict": "B", "baseless": "C"}
RM3_OPT = {"correct": "A", "wrong_explanation": "B", "not_an_error": "C", "original_wrong": "D"}


def _rm3_system() -> str:
    """SYSTEM_V3 的定义和严格口径，四个类别改成按字母排好的 A–D 选项，去掉「自然度」和「推理简短」两句。"""
    s = audit_synth.SYSTEM_V3.split("Answer in this format:")[0]
    lines = {k: next(l for l in s.splitlines() if l.startswith(f"- {k}:")) for k in RM3_OPT}
    s = s.replace("\n".join(lines[k] for k in ("correct", "not_an_error", "wrong_explanation", "original_wrong")),
                  "\n".join(f"{RM3_OPT[k]}. {lines[k][2:]}" for k in RM3_OPT))
    for drop in ("Also judge whether the error is natural: would a real assistant writing this response plausibly make this "
                 "mistake (yes), or does it look artificial, absurd or obviously planted (no)?\n\n",
                 "Keep your reasoning short: check the changed part against the source once, decide, and answer.\n\n"):
        s = s.replace(drop, "")
    return s + "Answer with a single letter: A, B, C or D."


RM3_SYSTEM = _rm3_system()


def too_long(prompt: list[dict], limit: int) -> bool:
    return sum(len(m["content"]) for m in prompt) / 3.2 + 64 > limit


def holdout(keys: list[str], frac: float, seed: int) -> set[str]:
    ks = sorted(set(keys))
    random.Random(seed).shuffle(ks)
    return set(ks[:int(len(ks) * frac)])


def rows(p: Path):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def build_rm1() -> dict[str, list[dict]]:
    cat = {r["id"]: r["category"] for r in rows(CLOSED / "gfree" / "rm3_final.jsonl")}

    def item(c: dict, **extra) -> dict:
        lab = "faithful" if c["label"] == "faithful" else (c["type"] if c["type"] in ("conflict", "baseless") else "baseless")
        prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input(c)}]
        return {"id": c["id"], "prompt": prompt, "completion": RM1_OPT[lab], "label": RM1_OPT[lab],
                "task_type": c["task_type"], **extra}

    data, drop = [], collections.Counter()
    for c in sorted(read_jsonl("outputs/rt/claims/clean_gfree.jsonl"), key=lambda r: r["id"]):
        c = {**c, "idx": int(c["idx"])}
        if c.get("pair_id"):
            k = cat.get(c["pair_id"])
            if c["label"] == "hallucination" and k in (None, "not_an_error"):
                drop["synth_halluc_" + str(k)] += 1
                continue
            if c["label"] == "faithful" and k == "original_wrong":
                drop["synth_orig_wrong"] += 1
                continue
        data.append(item(c, group=c["response_id"].split("-free")[0].split("#")[0], origin=c["origin"]))
    cal = holdout([r["group"] for r in data], 0.1, 0)
    batches = {r["id"]: r for r in read_jsonl("outputs/rt/analysis/test_batches.jsonl")}
    audit = {r["id"]: r for r in read_jsonl("outputs/rt/analysis/test_audit400.jsonl")}
    test = []
    for s in sorted(read_jsonl("outputs/external/ragtruth/test_pp.jsonl"), key=lambda r: r["id"]):
        for c in expand_claims(s):
            b = batches.get(c["id"], {})
            claude = (b.get("claude") if b.get("how") == "pp_valid" else
                      audit[c["id"]]["claude"] if c["id"] in audit else None)
            test.append(item(c, batch=b.get("batch"), claude=claude, coarse=c.get("coarse", False)))
    out = {"train": [r for r in data if r["group"] not in cal], "calib": [r for r in data if r["group"] in cal], "test": test}
    return out, dict(drop)


def build_rm2() -> dict[str, list[dict]]:
    def item(r: dict) -> dict:
        prompt = [{"role": "system", "content": RM2_SYSTEM},
                  {"role": "user", "content": discrim.render(r["source"], r["sentences"], int(r["idx"]))}]
        lab = "A" if r["origin_label"] == "natural" else "B"
        return {"id": r["id"], "prompt": prompt, "completion": lab, "label": lab, "task_type": r["task_type"],
                "type": r["type"], "group": r["source_id"]}

    tr = [item(r) for r in sorted(read_jsonl("outputs/rt/discrim/train.jsonl"), key=lambda r: r["id"])]
    cal = holdout([r["group"] for r in tr], 0.1, 0)
    test = [item(r) for r in sorted(read_jsonl("outputs/rt/discrim/heldout.jsonl"), key=lambda r: r["id"])]
    return {"train": [r for r in tr if r["group"] not in cal], "calib": [r for r in tr if r["group"] in cal], "test": test}, {}


def build_rm3() -> dict[str, list[dict]]:
    recs = {}
    for d in ("gtarget", "gfree"):
        for r in rows(CLOSED / d / "valid.jsonl"):
            for c in r["cands"]:
                if c.get("rec"):
                    recs[(d, f"{r['id']}#{c['j']}")] = c["rec"]

    def item(r: dict) -> dict:
        rec = recs[(r["dir"], r["id"])]
        prompt = [{"role": "system", "content": RM3_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}]
        lab = RM3_OPT[r["category"]]
        return {"id": f"{r['dir']}/{r['id']}", "prompt": prompt, "completion": lab, "label": lab, "category": r["category"],
                "task_type": rec["task_type"], "from": r["from"], "tier": r["tier"], "high_value": r.get("high_value", False),
                "group": r["id"].split("#")[0].split("-free")[0]}

    tr = [item(r) for r in sorted(rows(CLOSED / "rm3_train.jsonl"), key=lambda r: (r["dir"], r["id"]))]
    cal = holdout([r["group"] for r in tr], 0.1, 0)
    train = [r for r in tr if r["group"] not in cal]
    train += [{**r, "id": r["id"] + "~dup"} for r in train if r["high_value"]]   # 高价值样本权重 2
    test = [item(r) for r in sorted(rows(CLOSED / "rm3_eval.jsonl"), key=lambda r: (r["dir"], r["id"]))]
    return {"train": train, "calib": [r for r in tr if r["group"] in cal], "test": test}, {}


def build_rm2v2() -> dict[str, list[dict]]:
    """RM2 重训（2026-10-30，用户：看模型能不能区分出来）：自然 = RAGTruth 训练集人工标出的幻觉句；合成 = 种子生成器 G_seed
    在 600 条训练回答上各采 8 次、RM3 判合格（P ≥ 0.5，不看规则）的改写 / 插入句。按（类型）配平两类，按原文 8 : 1 : 1 切
    训练 / 校准 / 测试，原文不跨集合。"""
    from ragtruth.discrim import natural_rows
    pool = ["outputs/external/ragtruth/train_pool.jsonl", "outputs/external/ragtruth/train_sub.jsonl"]
    src_of = {s["id"]: s.get("source_id") for p in pool for s in read_jsonl(p)}
    syn = []
    for r in rows(Path("outputs/rt/realgen/gen_eval/scored_dpo_gseed_norules.jsonl")):
        if not r.get("rec") or (r.get("p_correct") or 0) < 0.5 or r.get("type") not in ("conflict", "baseless"):
            continue
        rec, i = r["rec"], r["idx"]
        sents = (rec["sentences"][:i - 1] + [rec["sentence"]] + (rec["sentences"][i - 1:] if r.get("op") == "insert" else rec["sentences"][i:]))
        syn.append({"id": r["id"] + "-syn", "source": rec["source"], "sentences": sents, "idx": i, "task_type": rec["task_type"],
                    "type": r["type"], "origin_label": "synthetic", "source_id": src_of.get(r["resp_id"])})
    nat = [{**r, "origin_label": "natural"} for r in natural_rows(pool)]
    rng = random.Random(0)
    keep = []
    for t in ("conflict", "baseless"):   # 两类在每个错误类型里一样多（防止靠类型区分）
        a = sorted((r for r in nat if r["type"] == t), key=lambda r: r["id"])
        b = sorted((r for r in syn if r["type"] == t), key=lambda r: r["id"])
        m = min(len(a), len(b))
        keep += rng.sample(a, m) + rng.sample(b, m)
    srcs = sorted({str(r["source_id"]) for r in keep})
    rng.shuffle(srcs)
    n = len(srcs)
    split_of = {s: ("calib" if k < n // 10 else "test" if k < n // 5 else "train") for k, s in enumerate(srcs)}

    def item(r: dict) -> dict:
        prompt = [{"role": "system", "content": RM2_SYSTEM},
                  {"role": "user", "content": discrim.render(r["source"], r["sentences"], int(r["idx"]))}]
        lab = "A" if r["origin_label"] == "natural" else "B"
        return {"id": r["id"], "prompt": prompt, "completion": lab, "label": lab, "task_type": r["task_type"],
                "type": r["type"], "group": str(r["source_id"])}

    out = {"train": [], "calib": [], "test": []}
    for r in sorted(keep, key=lambda r: r["id"]):
        out[split_of[str(r["source_id"])]].append(item(r))
    return out, {"synthetic_qualified": len(syn), "natural_all": len(nat), "kept_per_class": len(keep) // 2}


def build_rm2v2_mc() -> dict[str, list[dict]]:
    """2B 定向造错用的 RM2（2026-11-03，用户：「真实错误」加上 MiniCPM 自己的错误重训）：rm2v2 全部数据 + MiniCPM5-2B 基线
    （outputs/rt/minicpm/sys，Claude 金标 v1 标的有错句）当「自然」。2B GRPO 第 300 步的标注不用：那 100 条留作 critic 按类型的测试集，
    放进来生成器会学着模仿测试集的错误。MiniCPM 的句子按原文 8 : 1 : 1 切，训练里复制 2 份（量少、又是要贴近的分布）。"""
    out, meta = build_rm2v2()
    d = Path("outputs/rt/minicpm/sys")
    ans = {r["id"]: r for r in read_jsonl(d / "answers.jsonl")}
    nat = [r for r in read_jsonl(d / "final.jsonl") if r["is_error"] and r["type"] in ("conflict", "baseless")]
    srcs = sorted({r["source_id"] for r in nat})
    random.Random(1).shuffle(srcs)
    split_of = {s: ("calib" if k < len(srcs) // 10 else "test" if k < len(srcs) // 5 else "train") for k, s in enumerate(srcs)}
    n = collections.Counter()
    for r in nat:
        a = ans[r["resp_id"]]
        item = {"id": r["id"] + "-mcnat", "prompt": [{"role": "system", "content": RM2_SYSTEM},
                                                    {"role": "user", "content": discrim.render(a["source"], a["sentences"], int(r["idx"]))}],
                "completion": "A", "label": "A", "task_type": r["task_type"], "type": r["type"], "group": "mc-" + str(r["source_id"])}
        sp = split_of[r["source_id"]]
        out[sp] += [item] * (2 if sp == "train" else 1)
        n[sp] += 1
    return out, {**meta, "minicpm_natural": dict(n)}


def main() -> None:
    import sys
    builders = {"rm1": build_rm1, "rm2": build_rm2, "rm3": build_rm3, "rm2v2": build_rm2v2, "rm2v2__mc": build_rm2v2_mc}
    names = sys.argv[1:] or ["rm1", "rm2", "rm3"]   # python -m ragtruth.rm_data rm2v2：只建这一个
    for name, fn in ((n, builders[n]) for n in names):
        splits, drop = fn()
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"dropped": drop, "max_tokens": MAX_TOKENS[name]}
        for sp, rs in splits.items():
            keep = [r for r in rs if sp != "train" or not too_long(r["prompt"], MAX_TOKENS[name])]   # 只有训练会截断
            if sp == "train":
                random.Random(0).shuffle(keep)
            with open(d / f"{sp}.jsonl", "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            meta[sp] = {"n": len(keep), "too_long": len(rs) - len(keep),
                        "label": dict(sorted(collections.Counter(r["label"] for r in keep).items()))}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
