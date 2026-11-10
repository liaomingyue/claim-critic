"""核心集与派生数据集（所有数据集都从「Claude 改对的 2B 回答」这一个核心集派生；README 第二、三步）。

  python -m ragtruth.core build                         # 核心集 core.jsonl / pairs.jsonl / meta.json（本地）
  python -m ragtruth.core derive --tokenizer <基座>       # D1–D4、D7–D9 + sev 的核心集部分（RAG SFT 要分词器的聊天模板）
  python -m ragtruth.core sample --url ... --model gen --tokenizer <基座> --pool genpool_train.jsonl --out samples_core.jsonl --n 4   # 改写模型采样（GPU）
  python -m ragtruth.core rm2                           # D5：核心集回答上的改写样本（负）+ 同位置原句 / 核心集真错句（正）
  python -m ragtruth.core sevpool --n 800               # 第四步：抽待 Claude 单独标注的改写样本

输入：outputs/rt/minicpm/grpo/train_ref.jsonl（2B 原回答）、outputs/rt/minicpm/s1/fix*_raw.jsonl（Claude 改对）、
outputs/rt/core/label*_raw.jsonl（Claude 第二轮标注，ragtruth.core_label）、external_data/ragtruth（更多样的回答）。
输出都在 outputs/rt/core/（--dir 可换，换模型时按模型分目录），打分模型数据写 outputs/rt/rm/<名字>/。
切分按原文、固定字符串种子，重跑结果相同。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import zlib
from pathlib import Path

from common.schema import read_jsonl

PROMPTS = "outputs/rt/minicpm/grpo/train_ref.jsonl"
RM = Path("outputs/rt/rm")
RT_DIR = Path("external_data/ragtruth/dataset")
PP_TEST = Path("outputs/rt/rm/rm1/test.jsonl")   # RAGTruth++ 测试
HOLDOUT_N = 81
OPT = {"faithful": "A", "conflict": "B", "baseless": "C"}
EOS = "<|im_end|>"

# 机制 → 改法（标准清单 synth.FOCUS_TACTICS 里有的用清单原文，没有的写一句描述）
def tactic_of(mech: str, kind: str) -> str:
    from ragtruth.synth import FOCUS_TACTICS as F
    c, b = F["conflict"], F["baseless"]
    m = {("wrong_attribution", "conflict"): c[0], ("merged_across_passages", "conflict"): c[1], ("number_or_date", "conflict"): c[2],
         ("certainty_or_scope_shift", "conflict"): c[3], ("contradicts_source", "conflict"): c[5],
         ("merged_across_passages", "baseless"): b[1], ("unsupported_inference", "baseless"): b[1], ("certainty_or_scope_shift", "baseless"): b[0]}
    other = {"near_synonym_shift": "replace a word with a near-synonym that shifts the meaning",
             "added_specific_fact": "add a specific detail that the source does not give",
             "vague_evaluation_or_addon": "add an evaluation, purpose or consequence that the source does not state",
             "unsupported_inference": "state an inference that the source does not support",
             "wrong_attribution": c[0], "number_or_date": c[2], "contradicts_source": "state something that the source contradicts"}
    return m.get((mech, kind)) or other.get(mech) or "other"


def hsplit(key: str, salt: str, cuts: tuple[int, ...] = (50,)) -> int:
    """按 key 哈希分桶：返回落在第几段（cuts 是百分比边界）。"""
    h = int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), 16) % 100
    return sum(h >= c for c in cuts)


# ── 核心集 ──
def cmd_build(a) -> None:
    from ragtruth.convert import split_sentences
    from ragtruth.core_label import load_fixes, load_labels
    from ragtruth.policy_sft import splice
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    fixes, labels = load_fixes(), load_labels(a.labels)
    rows = [r for r in read_jsonl(PROMPTS) if r.get("ref_text") and r["id"] in fixes]
    srcs = sorted({str(r["source_id"]) for r in rows})
    random.Random("core|holdout").shuffle(srcs)
    hold = set(srcs[:HOLDOUT_N])
    core, pairs, why = [], [], collections.Counter()
    for r in sorted(rows, key=lambda r: str(r["id"])):
        f, lab = fixes[r["id"]], labels.get(r["id"])
        t = r["ref_text"].strip()
        sents = [t[x:y] for x, y in split_sentences(t)]
        split = "holdout" if str(r["source_id"]) in hold else "train"
        corrected = f["rewrite"] or (splice(t, f["fixes"]) if f["fixes"] else t)
        core.append({"id": str(r["id"]), "source_id": str(r["source_id"]), "task_type": r["task_type"], "split": split, "source": r["source"],
                     "original": t, "sentences": sents, "fixes": [] if f["rewrite"] else f["fixes"], "rewrite": f["rewrite"], "corrected": corrected,
                     "labels": lab, "unchanged": [], "fixed_sentences": []})
        if f["rewrite"]:
            why["rewrite"] += 1
            continue
        new = {int(x["idx"]): (x.get("new") or "").strip() for x in f["fixes"]}
        err = {e["idx"]: e for e in (lab or {}).get("errors", [])}
        fixed, pos = [], {}
        for k, s in enumerate(sents, 1):
            if k in new and not new[k]:
                continue
            pos[k] = len(fixed) + 1
            fixed.append(new.get(k, s))
        core[-1]["fixed_sentences"] = fixed
        core[-1]["unchanged"] = [pos[k] for k in range(1, len(sents) + 1) if k not in new]
        for k in sorted(new):
            if not new[k]:
                why["deleted"] += 1
            elif len(split_sentences(new[k])) != 1:
                why["multi_sentence_fix"] += 1
            elif " ".join(new[k].split()) == " ".join(sents[k - 1].split()):
                why["identical"] += 1
            elif len(fixed) < 2:
                why["one_sentence"] += 1
            elif k not in err:
                why["no_label"] += 1
            else:
                e = err[k]
                pairs.append({"id": f"core-{r['id']}-S{k}", "resp_id": str(r["id"]), "source_id": str(r["source_id"]), "task_type": r["task_type"],
                              "split": split, "source": r["source"], "sentences": fixed, "idx": pos[k], "real": sents[k - 1], "repaired": new[k],
                              "severity": e["severity"], "type": e["type"], "mechanism": e["mechanism"], "evidence": e["evidence"], "reason": e["reason"],
                              "fixed_evidence": e["fixed_evidence"], "fixed_reason": e["fixed_reason"]})
    for name, xs in (("core.jsonl", core), ("pairs.jsonl", pairs)):
        with open(d / name, "w", encoding="utf-8", newline="\n") as fh:
            for x in xs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    cnt = lambda xs, k: dict(sorted(collections.Counter(x[k] for x in xs).items()))  # noqa: E731
    meta = {"answers": len(core), "labeled": sum(bool(x["labels"]) for x in core), "holdout_sources": len(hold), "dropped": dict(why),
            "answers_by_split": cnt(core, "split"), "pairs": len(pairs), "pairs_by_split": cnt(pairs, "split"),
            "pairs_train_by_task": cnt([p for p in pairs if p["split"] == "train"], "task_type"),
            "pairs_by_type": cnt(pairs, "type"), "pairs_by_severity": cnt(pairs, "severity"),
            "unchanged_by_split": {s: sum(len(x["unchanged"]) for x in core if x["split"] == s) for s in ("train", "holdout")}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CORE_BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def load(d: Path) -> tuple[list[dict], list[dict]]:
    return list(read_jsonl(d / "core.jsonl")), list(read_jsonl(d / "pairs.jsonl"))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_rm(name: str, parts: dict[str, list[dict]], max_tokens: int, extra: dict | None = None) -> dict:
    """打分模型数据：训练集去掉估计超长的（TRL 截断会切掉答案字母），校准 / 测试全留。"""
    from ragtruth.rm_data import too_long
    d = RM / name
    tr = [r for r in parts["train"] if not too_long(r["prompt"], max_tokens)]
    random.Random(f"{name}|shuffle").shuffle(tr)
    meta = {"max_tokens": max_tokens, "too_long_dropped": len(parts["train"]) - len(tr), **(extra or {})}
    for sp, rows in (("train", tr), ("calib", parts["calib"]), ("test", parts["test"])):
        write(d / f"{sp}.jsonl", rows)
        meta[sp] = {"n": len(rows), "label": dict(sorted(collections.Counter(r["label"] for r in rows).items())),
                    "origin": dict(sorted(collections.Counter(r.get("origin", "") for r in rows).items()))}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ── D1 / D2 ──
def d1_rag_sft(core: list[dict], tok, d: Path) -> dict:
    from ragtruth.ans_data import policy_prompt
    out = collections.defaultdict(list)
    for x in core:
        out[x["split"]].append({"id": x["id"], "task_type": x["task_type"], "prompt": policy_prompt(tok, x, True), "completion": x["corrected"].strip() + EOS,
                                "kind": "rewrite" if x["rewrite"] else "edited" if x["fixes"] else "unchanged"})
    write(d / "rag_sft_train.jsonl", out["train"]); write(d / "rag_sft_holdout.jsonl", out["holdout"])
    return {s: len(v) for s, v in out.items()}


def d2_grpo(core: list[dict], d: Path) -> dict:
    hold = {x["source_id"] for x in core if x["split"] == "holdout"}
    rows = [r for r in read_jsonl(PROMPTS) if str(r["source_id"]) not in hold]
    write(d / "grpo_train.jsonl", rows)
    return {"train": len(rows), "no_ref": sum(not r.get("ref_text") for r in rows)}


# ── D3 / D4：错误改写模型 ──
def example_pool(pairs: list[dict]) -> dict:
    pool = collections.defaultdict(list)
    for p in pairs:
        if p["split"] == "train":
            ctx = p["sentences"][:p["idx"] - 1] + [p["real"]] + p["sentences"][p["idx"]:]
            pool[(p["task_type"], p["type"])].append({"sentences": ctx, "idx": p["idx"], "evidence": p["evidence"] or None, "reason": p["reason"],
                                                      "source": p["source"]})
    return pool


def gen_prompt(resp: dict, pool: dict, want: str | None, key: str) -> str:
    from ragtruth.synth import build_prompt_free, free_examples
    ex = free_examples(pool, resp["task_type"], random.Random(key), exclude_source=resp["source"])
    return build_prompt_free({"source": resp["source"], "sentences": resp["sentences"], "task_type": resp["task_type"]}, ex,
                             focus=True, free_size=True, ctx=True, want_type=want)


def d3_seed(pairs: list[dict], pool: dict, d: Path, max_chars: int) -> dict:
    from ragtruth.synth import FOCUS_TACTICS
    out, why = collections.defaultdict(list), collections.Counter()
    for p in pairs:
        kind = p["type"]
        tactic = tactic_of(p["mechanism"], kind)
        comp = json.dumps({"plan": f"Introduce one {kind} error into S{p['idx']}: {tactic}.", "idx": p["idx"], "type": kind, "tactic": tactic,
                           "sentence": p["real"], "evidence": p["evidence"], "reason": p["reason"],
                           "original_evidence": p["fixed_evidence"], "original_reason": p["fixed_reason"]}, ensure_ascii=False)
        want = kind if zlib.crc32(p["id"].encode()) % 2 else None
        prompt = gen_prompt(p, pool, want, p["id"])
        if len(prompt) + len(comp) > max_chars:
            why["too_long"] += 1
            continue
        out[p["split"]].append({"id": p["id"], "source_id": p["source_id"], "task_type": p["task_type"], "type": kind, "severity": p["severity"],
                                "mechanism": p["mechanism"], "tactic": tactic, "focus": tactic in FOCUS_TACTICS[kind], "want_type": want,
                                "idx": p["idx"], "sentences": p["sentences"], "real": p["real"], "repaired": p["repaired"], "prompt": prompt, "completion": comp})
    random.Random("core|seed").shuffle(out["train"])
    write(d / "seed_sft_train.jsonl", out["train"]); write(d / "seed_sft_holdout.jsonl", out["holdout"])
    return {"train": len(out["train"]), "holdout": len(out["holdout"]), "dropped": dict(why),
            "train_by_type": dict(collections.Counter(r["type"] for r in out["train"])),
            "train_focus_share": round(sum(r["focus"] for r in out["train"]) / max(1, len(out["train"])), 3)}


def diverse_answers(train_src: set[str]) -> list[dict]:
    """RAGTruth 6 个模型在同一批原文上的回答：只要训练原文、QA / Summary、质量 good、没有人工幻觉标注、至少两句。"""
    from ragtruth.convert import split_sentences
    src = {str(s["source_id"]): s for s in read_jsonl(RT_DIR / "source_info.jsonl")}
    prompts = {str(r["source_id"]): r for r in read_jsonl(PROMPTS)}
    out = []
    for r in read_jsonl(RT_DIR / "response.jsonl"):
        sid = str(r["source_id"])
        if sid not in train_src or r.get("labels") or r.get("quality", "good") != "good" or sid not in prompts:
            continue
        t = r["response"].strip()
        sents = [t[x:y] for x, y in split_sentences(t)]
        if len(sents) < 2 or "unable to answer" in t.lower():
            continue
        out.append({"resp_id": f"rt-{r['id']}", "source_id": sid, "task_type": prompts[sid]["task_type"], "model": r["model"],
                    "source": prompts[sid]["source"], "sentences": sents})
    return out


def d4_pool(core: list[dict], pool: dict, d: Path) -> dict:
    def row(resp: dict, want: str | None, tag: str, split: str) -> dict:
        pid = f"{resp['resp_id']}#{tag}"
        return {"id": pid, "resp_id": resp["resp_id"], "source_id": resp["source_id"], "task_type": resp["task_type"], "model": resp["model"],
                "source": resp["source"], "sentences": resp["sentences"], "orig_p": [0.0] * len(resp["sentences"]), "want_type": want,
                "split": split, "prompt": gen_prompt(resp, pool, want, pid)}
    tr, ev = [], []
    for x in core:
        if not x["fixed_sentences"] or len(x["fixed_sentences"]) < 2:
            continue
        resp = {"resp_id": f"core-{x['id']}", "source_id": x["source_id"], "task_type": x["task_type"], "model": "core_corrected",
                "source": x["source"], "sentences": x["fixed_sentences"]}
        if x["split"] == "train":
            tr += [row(resp, t, t[0], "train") for t in ("conflict", "baseless")]
        else:
            t = ("conflict", "baseless")[zlib.crc32(x["id"].encode()) % 2]
            ev.append(row(resp, t, t[0], "holdout"))
    # 改写模型的 GRPO、采集池和 RM2 负样本只用核心集回答，不用 RAGTruth 其他模型的回答（diverse_answers 留着不用）
    write(d / "genpool_train.jsonl", tr); write(d / "genpool_eval.jsonl", ev)
    return {"core_train": len(tr), "core_eval": len(ev)}


# ── D7 满意度 ──
def d7_sat(core: list[dict]) -> dict:
    from ragtruth.sat_build import degrade, pair_rows
    parts = collections.defaultdict(list)
    rng = random.Random("core|sat")
    by_task = collections.defaultdict(list)
    for x in core:
        by_task[x["task_type"]].append(x)
    good = lambda s: s and s["overall"] == "good"  # noqa: E731
    for n, x in enumerate(core):
        sp = "train" if x["split"] == "train" else ("calib", "test")[hsplit(x["source_id"], "core|sat-hold")]
        L, corr, orig, src = x["labels"], x["corrected"], x["original"], x["source"]
        if L and L["original"]["overall"] != L["corrected"]["overall"] and corr.strip() != orig.strip():
            parts[sp] += pair_rows(x["id"], src, corr, orig, "x" if L["corrected"]["overall"] == "good" else "y", "claude", {"n": n, "task_type": x["task_type"]})
        if not (L and good(L["corrected"])):
            continue
        for kind in ("truncated", "too_short", "copied"):
            bad = degrade(kind, corr, src, "", rng)
            if bad:
                parts[sp] += pair_rows(x["id"], src, corr, bad, "x", kind, {"n": f"{n}{kind[0]}", "task_type": x["task_type"]})
        if x["task_type"] == "QA" and L["corrected"]["answers_task"] == "yes":
            parts[sp] += pair_rows(x["id"], src, corr, "I am unable to answer the question based on the given passages.", "x", "refusal",
                                   {"n": f"{n}r", "task_type": "QA"})
    return write_rm("sat_pair__core", parts, 4096)


# ── D8 拒答 ──
WORD = re.compile(r"[a-z]{4,}")


def coverage(answer: str, text: str) -> float:
    a = set(WORD.findall(answer.lower()))
    return len(a & set(WORD.findall(text.lower()))) / max(1, len(a))


def drop_support(source: str, answer: str) -> tuple[str, float] | None:
    """删掉和改对回答重合最多的那一段；返回 (新原文, 改对回答被剩下两段覆盖的比例)。"""
    from ragtruth.refuse_data import PASSAGE, QA_RE
    m = QA_RE.match(source)
    if not m:
        return None
    parts = [p for p in PASSAGE.split(m["passages"]) if p.strip()]
    if len(parts) != 3:
        return None
    hit = max(range(3), key=lambda i: coverage(answer, parts[i]))
    rest = [p for i, p in enumerate(parts) if i != hit]
    keep = [re.sub(r"^passage \d+:", f"passage {k + 1}:", p) for k, p in enumerate(rest)]
    mid = m["mid"].replace("following three passages", "following two passages")
    if mid == m["mid"]:
        return None
    return m["head"] + m["q"] + "\n" + mid + "".join(keep).rstrip("\n") + m["tail"], coverage(answer, "".join(rest))


def d8_refuse(core: list[dict], thr: float) -> dict:
    from ragtruth.refuse_data import QA_RE, RM_SYSTEM, overlap, splice
    qa = [x for x in core if x["task_type"] == "QA" and QA_RE.match(x["source"]) and x["corrected"].strip()
          and "unable to answer" not in x["corrected"].lower()]
    parts, stat = collections.defaultdict(list), collections.Counter()
    rng = random.Random("core|refuse")

    def row(rid, text, lab, origin, group, extra=None):
        return {"id": rid, "prompt": [{"role": "system", "content": RM_SYSTEM}, {"role": "user", "content": text.strip()}], "completion": lab, "label": lab,
                "group": group, "task_type": "QA", "origin": origin, **(extra or {})}

    by_split = collections.defaultdict(list)
    for x in qa:
        by_split["train" if x["split"] == "train" else "hold"].append(x)
    for grp, xs in by_split.items():
        for x in xs:
            sp = "train" if grp == "train" else ("calib", "test")[hsplit(x["source_id"], "core|refuse-hold")]
            parts[sp].append(row(f"core-orig-{x['id']}", x["source"], "A", "original", x["source_id"]))
            dr = drop_support(x["source"], x["corrected"])
            if dr:
                lab = "B" if dr[1] < thr else "A"
                stat[f"drop_{lab}"] += 1
                parts[sp].append(row(f"core-drop-{x['id']}", dr[0], lab, "drop_passage", x["source_id"], {"coverage": round(dr[1], 3)}))
            others = [y for y in xs if y["source_id"] != x["source_id"]]
            rng.shuffle(others)
            for y in others[:20]:
                if overlap(x["source"], y["source"]) < 0.2 and (s := splice(x["source"], y["source"])):
                    parts[sp].append(row(f"core-mis-{x['id']}-{y['id']}", s, "B", "mismatch", x["source_id"]))
                    break
    return write_rm("refuse__core", parts, 3072, {"threshold": thr, "drop": dict(stat)})


def cmd_refuse_calib(a) -> None:
    """用旧的 Claude 复判删段样本（refuse_data nearmiss，method=drop）定覆盖率门槛：同原文的改对回答 → 覆盖率 vs Claude 判能不能答。"""
    from common.llm import kept
    from ragtruth.refuse_data import D as RD
    core = {x["source_id"]: x for x in read_jsonl(Path(a.dir) / "core.jsonl") if x["task_type"] == "QA"}
    items = {x["id"]: x for x in read_jsonl(RD / "items_nearmiss.jsonl") if x.get("method", "drop") == "drop"}
    pts = []
    for r in kept(RD / "labels_nearmiss.jsonl"):
        x = items.get(r["id"])
        c = core.get(str(r.get("source_id")))
        if x and c and "answerable" in r:
            pts.append((coverage(c["corrected"], x["source"]), r["answerable"] == "no"))
    print("REFUSE_CALIB points", len(pts), "no", sum(y for _, y in pts))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        acc = sum((cv < t) == y for cv, y in pts) / max(1, len(pts))
        print(f"thr {t}: acc {acc:.3f}")


# ── D9 critic ──
def d9_critic(core: list[dict], pairs: list[dict]) -> dict:
    from ragtruth.rm_data import RM1_SYSTEM
    from ragtruth.task import render_input
    parts = collections.defaultdict(list)

    def row(rid, src, sents, idx, lab, origin, group, task):
        return {"id": rid, "prompt": [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": src, "sentences": sents, "idx": idx})}],
                "completion": OPT[lab], "label": OPT[lab], "task_type": task, "group": group, "origin": origin}
    for p in pairs:
        sp = "train" if p["split"] == "train" else "calib"
        ctx = p["sentences"][:p["idx"] - 1] + [p["real"]] + p["sentences"][p["idx"]:]
        parts[sp].append(row(p["id"] + "-err", p["source"], ctx, p["idx"], p["type"], "core_error", p["resp_id"], p["task_type"]))
        parts[sp].append(row(p["id"] + "-fix", p["source"], p["sentences"], p["idx"], "faithful", "core_fixed", p["resp_id"], p["task_type"]))
    for x in core:
        sp = "train" if x["split"] == "train" else "calib"
        for k in x["unchanged"]:
            s = x["fixed_sentences"][k - 1]
            if "unable to answer" in s.lower():
                continue
            parts[sp].append(row(f"core-{x['id']}-U{k}", x["source"], x["fixed_sentences"], k, "faithful", "core_unchanged", x["id"], x["task_type"]))
    parts["test"] = list(read_jsonl(PP_TEST))   # RAGTruth++ 测试 2306 句（同 RM1_SYSTEM）
    return write_rm("rm1__core", parts, 3072)


# ── sev：核心集派生的 A–D ──
def sev_core(pairs: list[dict], d: Path) -> dict:
    from ragtruth.sev_rm import SEV_SYSTEM
    from ragtruth import audit_synth
    by_src = collections.defaultdict(list)
    for p in pairs:
        by_src[p["source_id"]].append(p)
    rows, c = [], collections.Counter()
    for p in pairs:
        if p["split"] != "train":
            continue
        ctx = p["sentences"][:p["idx"] - 1] + [p["real"]] + p["sentences"][p["idx"]:]
        base = {"source": p["source"], "sentences": ctx, "idx": p["idx"], "original": p["repaired"], "sentence": p["real"]}
        right = "A" if p["severity"] == "clear" else "C"
        wrong = "B" if p["severity"] == "clear" else "D"
        other = [q for q in by_src[p["source_id"]] if q["id"] != p["id"] and q["evidence"] and q["evidence"] != p["evidence"]]
        variants = [(right, {"type": p["type"], "evidence": p["evidence"], "reason": p["reason"]}, "core_right"),
                    (wrong, {"type": "baseless" if p["type"] == "conflict" else "conflict", "evidence": p["evidence"], "reason": p["reason"]}, "core_flip_type")]
        if other and zlib.crc32(p["id"].encode()) % 2:
            q = other[zlib.crc32(p["id"].encode()) % len(other)]
            variants[1] = (wrong, {"type": p["type"], "evidence": q["evidence"], "reason": q["reason"]}, "core_swap_evidence")
        for lab, claim, origin in variants:
            rec = {**base, **claim}
            rows.append({"id": f"sev/{p['id']}/{origin}", "prompt": [{"role": "system", "content": SEV_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}],
                         "completion": lab, "label": lab, "task_type": p["task_type"], "group": "gh-" + p["source_id"], "origin": origin})
            c[lab] += 1
    write(d / "sev_core_train.jsonl", rows)
    return {"rows": len(rows), "label": dict(sorted(c.items()))}


def cmd_derive(a) -> None:
    d = Path(a.dir)
    core, pairs = load(d)
    pool = example_pool(pairs)
    meta = {"D2_grpo": d2_grpo(core, d), "D3_seed": d3_seed(pairs, pool, d, a.max_chars), "D4_genpool": d4_pool(core, pool, d),
            "D7_sat": d7_sat(core), "D8_refuse": d8_refuse(core, a.refuse_thr), "D9_critic": d9_critic(core, pairs), "sev_core": sev_core(pairs, d)}
    if a.tokenizer:
        from transformers import AutoTokenizer
        meta["D1_rag_sft"] = d1_rag_sft(core, AutoTokenizer.from_pretrained(a.tokenizer), d)
    (d / "derive_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CORE_DERIVE", json.dumps(meta, ensure_ascii=False), flush=True)


# ── 改写模型采样（GPU）→ D5 / sev 待标池 ──
def cmd_sample(a) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    from train.gen_reward import context, parse
    d = Path(a.dir)
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = list(read_jsonl(d / a.pool))

    def one(r: dict) -> list[dict]:
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=768, temperature=1.0, top_p=1.0,
                                    n=a.n, seed=zlib.crc32(r["id"].encode()) % 100000, extra_body={"top_k": -1})
        res = []
        for j, c in enumerate(out.choices):
            rec, why = parse(c.text, r)
            x = {"id": f"{r['id']}#s{j}", "prompt_id": r["id"], "resp_id": r["resp_id"], "source_id": r["source_id"], "task_type": r["task_type"],
                 "model": r["model"], "why": why}
            if rec is not None and rec["op"] == "rewrite":
                x.update({"type": rec["type"], "idx": rec["idx"], "sentence": rec["sentence"], "original": rec["original"], "evidence": rec["evidence"],
                          "reason": rec["reason"], "tactic_raw": rec["tactic_raw"], "source": r["source"], "context": context(rec), "sentences_orig": r["sentences"]})
            res.append(x)
        return res

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = [x for xs in ex.map(one, rows) for x in xs]
    write(d / a.out, out)
    print("CORE_SAMPLE", json.dumps({"pool": a.pool, "prompts": len(rows), "samples": len(out), "why": dict(collections.Counter(x["why"] for x in out))}), flush=True)


def cmd_rm2(a) -> None:
    """D5：B = 改写模型 v0 在核心集改对后回答上的改写句（过硬规则；和该回答原错句相似度 ≥ 0.9 的去掉），A = 同一回答同一位置的原句（改对句）；
    A 再补核心集真错句（自然写出来的错）。改写样本按原文 8 : 1 : 1 切；真错句训练 / 留出跟核心集切分走。"""
    from ragtruth import discrim
    from ragtruth.rm_data import RM2G_SYSTEM
    d = Path(a.dir)
    _, pairs = load(d)
    hold = {p["source_id"] for p in pairs if p["split"] == "holdout"}
    samp = [x for x in read_jsonl(d / a.inp) if x["why"] == "ok"]
    seen, parts = set(), collections.defaultdict(list)

    def row(rid, src, sents, idx, lab, origin, group, task):
        return {"id": rid, "prompt": [{"role": "system", "content": RM2G_SYSTEM}, {"role": "user", "content": discrim.render(src, sents, idx)}],
                "completion": lab, "label": lab, "task_type": task, "group": group, "origin": origin}

    def sp_of(sid: str) -> str:
        return ("train", "calib", "test")[hsplit(sid, "core|rm2", (80, 90))]
    from train.gen_reward import edit_sim
    real = collections.defaultdict(list)   # 每条核心集回答的原错句：改写模型在同一批回答上可能直接写回 2B 的原错句，那是「自然」，不能当负样本
    for p in pairs:
        real["core-" + p["resp_id"]].append(p["real"])
    drop = collections.Counter()
    for x in sorted(samp, key=lambda x: x["id"]):
        key = (x["resp_id"], x["idx"])
        if key in seen or x["source_id"] in hold:
            continue
        if any(edit_sim(x["sentence"], r) >= 0.9 for r in real[x["resp_id"]]):
            drop["same_as_real_error"] += 1
            continue
        seen.add(key)
        sp = sp_of(x["source_id"])
        parts[sp].append(row(x["id"] + "-syn", x["source"], x["context"], x["idx"], "B", "gen_v0", x["source_id"], x["task_type"]))
        parts[sp].append(row(x["id"] + "-nat", x["source"], x["sentences_orig"], x["idx"], "A", "original_" + x["model"], x["source_id"], x["task_type"]))
    for p in pairs:
        ctx = p["sentences"][:p["idx"] - 1] + [p["real"]] + p["sentences"][p["idx"]:]
        sp = "test" if p["split"] == "holdout" else "train"
        parts[sp].append(row(p["id"] + "-real", p["source"], ctx, p["idx"], "A", "core_real_error", p["source_id"], p["task_type"]))
    print("CORE_RM2", json.dumps(write_rm("rm2__core", parts, 3072, {"dropped": dict(drop)}), ensure_ascii=False), flush=True)


def cmd_sevpool(a) -> None:
    """第四步：从改写模型 v0 在 D4 训练池上的样本里，按（任务, 类型）分层抽 n 条待 Claude 单独标注，每个原文至多 2 条。"""
    d = Path(a.dir)
    xs = [x for x in read_jsonl(d / a.inp) if x["why"] == "ok"]
    rng = random.Random("core|sevpool")
    rng.shuffle(xs)
    cells = collections.defaultdict(list)
    for x in xs:
        cells[(x["task_type"], x["type"])].append(x)
    per_src, out = collections.Counter(), []
    keys = sorted(cells)
    while len(out) < a.n and any(cells[k] for k in keys):
        for k in keys:
            while cells[k]:
                x = cells[k].pop()
                if per_src[x["source_id"]] < 2:
                    per_src[x["source_id"]] += 1
                    out.append(x)
                    break
            if len(out) >= a.n:
                break
    from ragtruth.sev_label import family
    out = [{**x, "fam": family(x.get("tactic_raw")), "cell": f"core_{x['type']}", "p_correct": None, "p_halluc": None} for x in out]
    write(d / "sevpool.jsonl", out)
    print("CORE_SEVPOOL", len(out), dict(collections.Counter((x["task_type"], x["type"]) for x in out)), flush=True)


def cmd_sev(a) -> None:
    """sev__core：Claude 单独标注的改写样本（sev_label label → sevpool_labels.jsonl）按原文 70 / 10 / 20 切训练 / 校准 / 测试；
    核心集派生的 A–D（sev_core_train.jsonl）只进训练，和校准 / 测试同原文的去掉。测试只用 Claude 单独标注的部分（门槛实际要判的分布）。"""
    from ragtruth.sev_rm import prompt as sev_prompt
    d = Path(a.dir)
    pool = {x["id"]: x for x in read_jsonl(d / "sevpool.jsonl")}
    labs = [r for r in read_jsonl(d / "sevpool_labels.jsonl") if r.get("option")]
    parts, drop = collections.defaultdict(list), collections.Counter()
    held = set()
    for r in sorted(labs, key=lambda r: r["hid"]):
        x = pool[r["hid"]]
        sp = ("train", "calib", "test")[hsplit(x["source_id"], "core|sev", (70, 80))]
        if sp != "train":
            held.add(x["source_id"])
        parts[sp].append({"id": "sev/" + r["hid"], "prompt": sev_prompt(x), "completion": r["option"], "label": r["option"], "task_type": x["task_type"],
                          "group": "gh-" + x["source_id"], "origin": "claude_sev"})
    for r in read_jsonl(d / "sev_core_train.jsonl"):
        if r["group"][3:] in held:
            drop["core_overlap_held"] += 1
            continue
        parts["train"].append(r)
    print("CORE_SEV", json.dumps(write_rm("sev__core", parts, 4096, {"dropped": dict(drop)}), ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "derive", "refuse_calib", "sample", "rm2", "sevpool", "sev"])
    ap.add_argument("--dir", default="outputs/rt/core")
    ap.add_argument("--labels", default="outputs/rt/core", help="build：Claude 第二轮标注所在目录")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--max_chars", type=int, default=14000)
    ap.add_argument("--refuse_thr", type=float, default=0.3, help="删段后改对回答被剩余段落覆盖的比例低于它 = 答不了（refuse_calib：旧 Claude 复判 23 条上 0.3 最准）")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--pool", default="genpool_train.jsonl")
    ap.add_argument("--inp", default="samples_core.jsonl")
    ap.add_argument("--out", default="samples_core.jsonl")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    {"build": cmd_build, "derive": cmd_derive, "refuse_calib": cmd_refuse_calib, "sample": cmd_sample, "rm2": cmd_rm2, "sevpool": cmd_sevpool, "sev": cmd_sev}[a.cmd](a)


if __name__ == "__main__":
    main()
