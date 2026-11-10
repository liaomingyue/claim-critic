"""RM3 加 Claude 判过的合成错句重训（加进合成数据时判是否真错的 RM3 要更严）。
现状：h1 合成错句里 Claude 判过的 504 条（真错 399、不是错 105，都已过 RM3 门槛 0.7），RM3 P(A+B) 对 Claude 真错的 AUC 只有 0.51。
这里把它们按原文切出 25% 当留出集（test_synth），其余加进 RM3 训练集（不是错 → C，真错且类型和 Claude 一致 → A，否则 → B），
看新 RM3 在留出集上能不能分开。

  python -m ragtruth.rm3_v2 build --harvest outputs/rt/gentarget/harvest_h1.jsonl --labels configs/data/rm3v2_labels.jsonl
  python -m ragtruth.rm3_v2 score --inp outputs/rt/rm/rm3__mc_v2/test_synth.jsonl --url http://127.0.0.1:8000/v1 --model rm3 --opts ABCD --temp 1.0 --out x.jsonl
  python -m ragtruth.rm3_v2 auc --inp outputs/rt/rm/rm3__mc_v2/test_synth.jsonl --scores old=a.jsonl new=b.jsonl
score 也用来给任意带 prompt 的行打选项概率（例如 critic 训练集里标 A 的行）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

from common.schema import read_jsonl
from ragtruth import audit_synth
from ragtruth.rm_data import OUT, RM3_SYSTEM, too_long

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def label_of(x: dict, lab: dict) -> str:
    if not lab["is_error"]:
        return "C"
    return "A" if lab.get("claude_type") == x["type"] else "B"


def cmd_build(a) -> None:
    labs = {r["hid"]: r for r in read_jsonl(a.labels)}
    hv = {x["id"]: x for x in read_jsonl(a.harvest) if x["id"] in labs}
    miss = sorted(set(labs) - set(hv))
    assert not miss, f"{len(miss)} labels not in harvest"
    base, d = OUT / a.base, OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    ml = json.load(open(base / "meta.json", encoding="utf-8"))["max_tokens"]
    srcs = sorted({str(x["source_id"]) for x in hv.values()})
    random.Random(f"{a.seed}|rm3v2").shuffle(srcs)
    hold = set(srcs[:int(len(srcs) * a.hold)])
    new, test, drop = [], [], 0
    for h in sorted(hv):
        x, lab = hv[h], labs[h]
        rec = {"source": x["source"], "sentences": x["context"], "idx": x["idx"], "original": x["original"], "sentence": x["sentence"],
               "type": x["type"], "evidence": x["evidence"], "reason": x["reason"]}
        prompt = [{"role": "system", "content": RM3_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}]
        if too_long(prompt, ml):
            drop += 1
            continue
        L = label_of(x, lab)
        r = {"id": "h1c/" + h, "prompt": prompt, "completion": L, "label": L, "task_type": x["task_type"], "group": "gh-" + str(x["source_id"]),
             "from": "claude_h1", "is_error": lab["is_error"], "severity": lab.get("severity"), "p_correct_old": x["p_correct"], "p_halluc_2b": x.get("p_halluc")}
        (test if str(x["source_id"]) in hold else new).append(r)
    train = list(read_jsonl(base / "train.jsonl")) + [r for r in new for _ in range(a.dup)]
    random.Random(a.seed).shuffle(train)
    with open(d / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(d / "test_synth.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copy(base / sp, d / sp)
    meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "labels": a.labels, "dup": a.dup, "too_long_dropped": drop,
            "added": dict(collections.Counter(r["label"] for r in new)), "test_synth": dict(collections.Counter(r["label"] for r in test)),
            "test_synth_sources": len(hold), "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("RM3V2_BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def cmd_score(a) -> None:
    from openai import OpenAI
    from ragtruth.rm_eval import option_logprobs, probs
    cl = OpenAI(base_url=a.url, api_key="x", timeout=600)
    rows = [r for r in read_jsonl(a.inp) if not a.label or r.get("label") == a.label]

    def run(r: dict) -> dict:
        resp = cl.chat.completions.create(model=a.model, messages=r["prompt"], max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK)
        lp = option_logprobs(resp.choices[0], a.opts)
        return {"id": r["id"], "p": probs(lp, a.temp) if lp else None}

    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(run, rows))
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("SCORED", a.inp, len(out), "failed", sum(o["p"] is None for o in out), flush=True)


def auc(pos: list[float], neg: list[float]) -> float:
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))


def cmd_auc(a) -> None:
    rows = {r["id"]: r for r in read_jsonl(a.inp)}
    res = {"n": len(rows), "errors": sum(r["is_error"] for r in rows.values())}
    old = [r["p_correct_old"] for r in rows.values()]
    res["harvest_p_correct"] = round(auc([p for p, r in zip(old, rows.values()) if r["is_error"]], [p for p, r in zip(old, rows.values()) if not r["is_error"]]), 4)
    for spec in a.scores:
        name, path = spec.split("=", 1)
        s = {o["id"]: o["p"] for o in read_jsonl(path) if o["p"]}
        pc = {i: s[i]["A"] + s[i]["B"] for i in rows if i in s}
        pos = [pc[i] for i in pc if rows[i]["is_error"]]
        neg = [pc[i] for i in pc if not rows[i]["is_error"]]
        keep = {t: (sum(p >= t for p in pos), sum(p >= t for p in neg)) for t in (0.5, 0.7, 0.9)}
        res[name] = {"auc_A+B": round(auc(pos, neg), 4), "auc_notC": round(auc([1 - s[i]["C"] for i in pc if rows[i]["is_error"]], [1 - s[i]["C"] for i in pc if not rows[i]["is_error"]]), 4),
                     "kept_err/nonerr@": {str(t): v for t, v in keep.items()}}
    print("RM3V2_AUC", json.dumps(res, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "auc"])
    ap.add_argument("--harvest", default="outputs/rt/gentarget/harvest_h1.jsonl")
    ap.add_argument("--labels", default="configs/data/rm3v2_labels.jsonl")
    ap.add_argument("--base", default="rm3__mc")
    ap.add_argument("--name", default="rm3__mc_v2")
    ap.add_argument("--hold", type=float, default=0.25)
    ap.add_argument("--dup", type=int, default=2, help="新加样本重复几次（相对 4358 条原训练集，400 条信号太弱）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inp")
    ap.add_argument("--label", default="", help="score 只打这个标签的行")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--opts", default="ABCD")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--scores", nargs="*", default=[])
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score, "auc": cmd_auc}[a.cmd](a)


if __name__ == "__main__":
    main()
