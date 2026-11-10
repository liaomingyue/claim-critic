"""严重程度 RM（Claude 标显然 / 非显然 / 没有错误 + 解释对不对，合成一道选择题训 RM；可以先抢单卡）。
6 选 1（单字母，校准输出）：A 显然错 + 解释对、B 显然错 + 解释不对、C 非显然错 + 解释对、D 非显然错 + 解释不对、E 没有错、F 原句本身有错。
奖励用 P(显然) = A + B、P(真错) = A + B + C + D、P(解释对 | 真错) = (A + C) / (A + B + C + D)。

数据：
  - Claude 新标的 787 条（outputs/rt/sevlabel/items_p{1,2,2r,3}_labels.jsonl → configs/data/sev_labels.jsonl），按原文切 test 20% / calib 10% / train 70%；
  - h1 旧标的 504 条（configs/data/rm3v2_labels.jsonl：有 severity，「解释对」用 Claude 类型 = 生成器类型近似，原句当作没问题）只进训练，原文和 test / calib 重叠的去掉；
  - 多任务：rm3__mc 的训练集（RM3_SYSTEM，4 选 1 判是不是错）原样加进训练，同一个 adapter 两种题靠系统提示区分——「是不是错」从 4358 条旧数据学，新数据只补严重程度。
--frac 0.5 只用一半新训练原文（学习曲线）。

  python -m ragtruth.sev_rm merge                       # 本地：合并 Claude 标签 → configs/data/sev_labels.jsonl
  python -m ragtruth.sev_rm build --name sev__mc        # 远端：读 harvest_h1 拼 prompt
  python -m ragtruth.sev_rm gates --name sev__mc        # 训完：test 上 P(显然) 各门槛的精度，和旧 RM3 比
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from common.schema import read_jsonl
from ragtruth import audit_synth
from ragtruth.rm_data import OUT, RM3_SYSTEM, too_long

SEV_SYSTEM = """You audit a sentence that was deliberately edited to plant an error. You see the source (the exact input an assistant received), the assistant's response with the edited sentence in place, the original version of that sentence, and the editor's claimed error: its type, the source evidence it points to, and its reason.

Judge the edited sentence against the source (use the other sentences only to resolve references):
- a clear error: a careful reader would call it a factual error: it contradicts what the source states, assigns an action, statement or property to the wrong person, entity, passage or step, gets a number, date or order wrong, or joins details into a claim the source clearly does not make.
- a minor error: technically unsupported but with small effect, where people could disagree: a slight shift of certainty or scope, a vague evaluation or add-on, a near-synonym that shifts the meaning a little.
- not an error: true synonyms, faithful summaries or reorderings, leaving information out, what follows directly from the source.
The explanation is right only if the claimed type is right (conflict = the source contradicts it; baseless = the source does not contain it), the claimed evidence is the part of the source the error is about, and the reason describes the actual error.

Options:
A. clear error, and the explanation is right.
B. clear error, but the explanation is wrong.
C. minor error, and the explanation is right.
D. minor error, but the explanation is wrong.
E. not an error.
F. the original sentence is already unfaithful to the source.

Answer with a single letter: A, B, C, D, E or F."""
LAB_FILES = ("items_p1_labels.jsonl", "items_p2_labels.jsonl", "items_p2r_labels.jsonl", "items_p3_labels.jsonl")


def cmd_merge(a) -> None:
    rows, seen = [], set()
    for f in LAB_FILES:
        for r in read_jsonl(Path(a.label_dir) / f):
            if r["hid"] in seen or r["option"] is None:
                continue
            seen.add(r["hid"])
            c = r["claude"]
            rows.append({"hid": r["hid"], "option": r["option"], "severity": c.get("severity"), "explanation_ok": c.get("explanation_ok"), "original_ok": c.get("original_ok"),
                         "mechanism": c.get("mechanism"), "cell": r["cell"], "fam": r["fam"], "p_correct": r["p_correct"], "p_halluc": r["p_halluc"]})
    Path(a.labels).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("SEV_MERGE", a.labels, len(rows), dict(collections.Counter(r["option"] for r in rows)), flush=True)


def old_option(lab: dict, gen_type: str) -> str:
    if not lab["is_error"]:
        return "E"
    clear = lab.get("severity") == "clear"
    ok = lab.get("claude_type") == gen_type
    return ("A" if ok else "B") if clear else ("C" if ok else "D")


def prompt(x: dict) -> list[dict]:
    rec = {"source": x["source"], "sentences": x["context"], "idx": x["idx"], "original": x["original"], "sentence": x["sentence"],
           "type": x["type"], "evidence": x["evidence"], "reason": x["reason"]}
    return [{"role": "system", "content": SEV_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}]


def cmd_build(a) -> None:
    new = {r["hid"]: r for r in read_jsonl(a.labels)}
    old = {r["hid"]: r for r in read_jsonl(a.old_labels) if r["hid"] not in new}
    hv = {x["id"]: x for x in read_jsonl(a.harvest) if x["id"] in new or x["id"] in old}
    miss = sorted((set(new) | set(old)) - set(hv))
    assert not miss, f"{len(miss)} labels not in harvest"
    base, d = OUT / a.base, OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    ml = json.load(open(base / "meta.json", encoding="utf-8"))["max_tokens"]
    srcs = sorted({str(hv[h]["source_id"]) for h in new})
    random.Random(f"{a.seed}|sevrm").shuffle(srcs)
    nt, nc = int(len(srcs) * 0.2), int(len(srcs) * 0.1)
    test_s, calib_s = set(srcs[:nt]), set(srcs[nt:nt + nc])
    train_s = srcs[nt + nc:]
    use_s = set(train_s[:int(len(train_s) * a.frac)])
    parts, drop = collections.defaultdict(list), collections.Counter()

    def row(h: str, L: str, origin: str) -> dict | None:
        x = hv[h]
        p = prompt(x)
        if too_long(p, ml):
            drop["too_long"] += 1
            return None
        return {"id": "sev/" + h, "prompt": p, "completion": L, "label": L, "task_type": x["task_type"], "group": "gh-" + str(x["source_id"]), "origin": origin,
                "p_correct_old": float(x["p_correct"]), "p_halluc": x.get("p_halluc"), "cell": new[h]["cell"] if h in new else "old504"}

    for h in sorted(new):
        s = str(hv[h]["source_id"])
        sp = "test" if s in test_s else "calib" if s in calib_s else "train" if s in use_s else None
        if sp and (r := row(h, new[h]["option"], "claude_sev")):
            parts[sp].append(r)
    held = test_s | calib_s
    for h in sorted(old):
        if str(hv[h]["source_id"]) in held:
            drop["old_overlap"] += 1
            continue
        if r := row(h, old_option(old[h], hv[h]["type"]), "claude_h1_proxy"):
            parts["train"].append(r)
    rm3 = list(read_jsonl(base / "train.jsonl"))
    train = [r for r in parts["train"] for _ in range(a.dup)] + rm3
    random.Random(a.seed).shuffle(train)
    for sp, rows in (("train", train), ("calib", parts["calib"]), ("test", parts["test"])):
        with open(d / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab = lambda rs: dict(sorted(collections.Counter(r["label"] for r in rs).items()))
    meta = {"max_tokens": ml, "base": a.base, "frac": a.frac, "dup": a.dup, "dropped": dict(drop), "sources": {"test": len(test_s), "calib": len(calib_s), "train_used": len(use_s)},
            "sev_train": lab(parts["train"]), "rm3_train": len(rm3), "calib": lab(parts["calib"]), "test": lab(parts["test"]), "train_n": len(train)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SEV_BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def auc(pos: list[float], neg: list[float]) -> float:
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))


def cmd_gates(a) -> None:
    from ragtruth.rm_eval import probs
    d = OUT / a.name
    T = json.load(open(d / "eval" / "report.json", encoding="utf-8"))["temperature"]
    rs = [r for r in read_jsonl(d / "eval" / "test_scores.jsonl") if r.get("lp")]
    res = {"name": a.name, "T": T, "n": len(rs), "label": dict(collections.Counter(r["label"] for r in rs))}
    for r in rs:
        p = probs(r["lp"], T)
        r["p_clear"], r["p_err"] = p["A"] + p["B"], p["A"] + p["B"] + p["C"] + p["D"]
    clear = lambda r: r["label"] in "AB"
    err = lambda r: r["label"] in "ABCD"
    res["auc_clear_vs_rest"] = round(auc([r["p_clear"] for r in rs if clear(r)], [r["p_clear"] for r in rs if not clear(r)]), 4)
    res["auc_err_vs_E"] = round(auc([r["p_err"] for r in rs if err(r)], [r["p_err"] for r in rs if r["label"] == "E"]), 4)
    res["old_rm3_auc_clear_vs_rest"] = round(auc([r["p_correct_old"] for r in rs if clear(r)], [r["p_correct_old"] for r in rs if not clear(r)]), 4)
    res["old_rm3_auc_err_vs_E"] = round(auc([r["p_correct_old"] for r in rs if err(r)], [r["p_correct_old"] for r in rs if r["label"] == "E"]), 4)
    res["base_rate_clear"] = round(sum(map(clear, rs)) / len(rs), 3)
    res["gates_p_clear"] = {str(t): {"kept": sum(r["p_clear"] >= t for r in rs), "clear": sum(clear(r) and r["p_clear"] >= t for r in rs),
                                     "precision": round(sum(clear(r) and r["p_clear"] >= t for r in rs) / max(1, sum(r["p_clear"] >= t for r in rs)), 3),
                                     "recall": round(sum(clear(r) and r["p_clear"] >= t for r in rs) / max(1, sum(map(clear, rs))), 3)} for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
    blind = [r for r in rs if r.get("p_halluc") is not None and float(r["p_halluc"]) < 0.5]
    res["blind_critic_lt_0.5"] = {"n": len(blind), "clear": sum(map(clear, blind)),
                                  **{f"precision@{t}": round(sum(clear(r) and r["p_clear"] >= t for r in blind) / max(1, sum(r["p_clear"] >= t for r in blind)), 3) for t in (0.5, 0.7)},
                                  **{f"kept@{t}": sum(r["p_clear"] >= t for r in blind) for t in (0.5, 0.7)}}
    (d / "eval" / "gates.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SEV_GATES", json.dumps(res, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "build", "gates"])
    ap.add_argument("--label_dir", default="outputs/rt/sevlabel")
    ap.add_argument("--labels", default="configs/data/sev_labels.jsonl")
    ap.add_argument("--old_labels", default="configs/data/rm3v2_labels.jsonl")
    ap.add_argument("--harvest", default="outputs/rt/gentarget/harvest_h1.jsonl")
    ap.add_argument("--base", default="rm3__mc")
    ap.add_argument("--name", default="sev__mc")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--dup", type=int, default=2, help="严重程度样本在训练里重复几次（相对 4358 条 RM3 旧数据）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"merge": cmd_merge, "build": cmd_build, "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    main()
