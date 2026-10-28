"""RAGTruth 原始数据 → critic 样本（一行一条回答，格式见 ragtruth.task）。

- 回答按句切分（split_sentences），与人工片段有重叠（≥3 个字符）的句子算幻觉句；
  句子类型：重叠片段里有 Conflict 就记 conflict，否则 baseless。
- 只被覆盖 ≥3 句的大片段标中的幻觉句记入 coarse（见 coarse_sentences），逐句训练和句级指标不用。
- 丢掉 quality 不是 good 的回答（拒答、截断，共 173 条）。
- 官方 train 按 source_id 再切出 val（同一源的 6 条回答在同一边），官方 test 原样保留。
- 再按源从 train 抽 1/10 作为实验用的小训练集 train_sub（实际业务的 critic 只有几千条训练样本），
  剩下的是 train_pool。

用法：python -m ragtruth.convert --src external_data/ragtruth/dataset --out outputs/external/ragtruth
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from common.schema import write_jsonl

ABBR = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "jr", "sr", "inc", "ltd", "co",
        "no", "fig", "approx", "a.m", "p.m", "mt", "ft", "gen", "gov", "sen", "rep", "prof"}
_BOUND = re.compile(r"\n+|[.!?]+[\"'”’)\]]*\s+")
MIN_OVERLAP = 3
COARSE_SENTS = 3


def split_sentences(text: str) -> list[tuple[int, int]]:
    """返回句子在 text 中的 (start, end)，已去掉首尾空白。换行一律断句；句末标点后接大写、数字、引号或括号才断，
    跳过常见缩写、人名首字母和行首的列表序号（「1. 」）。没有字母的碎片（单独的序号）并入下一句。"""
    cuts = [0]
    for m in _BOUND.finditer(text):
        end = m.end()
        if m.group().startswith("\n"):
            cuts.append(end)
            continue
        nxt = text[end:end + 1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"“'([*-"):
            continue
        tok_m = re.search(r"(\S+)$", text[:m.start() + 1])
        tok = tok_m.group(1) if tok_m else ""
        bare = tok.rstrip(".!?").lstrip("(\"“'").lower()
        line = text[text.rfind("\n", 0, m.start()) + 1: m.start() + 1].strip()
        if bare in ABBR or re.fullmatch(r"[A-Z]\.", tok) or re.fullmatch(r"\d{1,2}[.)]", line):
            continue
        cuts.append(end)
    cuts.append(len(text))
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        if seg.strip():
            spans.append((a + len(seg) - len(seg.lstrip()), a + len(seg.rstrip())))
    merged, carry = [], None
    for a, b in spans:
        if carry is not None:
            a, carry = carry, None
        if not re.search(r"[A-Za-z]{2}", text[a:b]):
            carry = a
            continue
        merged.append((a, b))
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], spans[-1][1])
        else:
            merged.append((carry, spans[-1][1]))
    return merged


def label_sentences(sent_spans: list[tuple[int, int]], labels: list[dict]) -> dict[int, str]:
    bad: dict[int, str] = {}
    for i, (a, b) in enumerate(sent_spans, 1):
        kinds = [x["label_type"] for x in labels if min(b, x["end"]) - max(a, x["start"]) >= MIN_OVERLAP]
        if kinds:
            bad[i] = "conflict" if any("Conflict" in k for k in kinds) else "baseless"
    return bad


def coarse_sentences(sent_spans: list[tuple[int, int]], labels: list[dict], bad: dict[int, str]) -> list[int]:
    """只被「大片段」（覆盖 ≥ COARSE_SENTS 句）标中的幻觉句。标注员有时把整段甚至整条回答标成一个片段，
    按句拆开后逐字照抄原文的句子也成了幻觉句；这些句子不进逐句训练和句级指标，只参与回答级判断。"""
    def n_cover(x):
        return sum(1 for a, b in sent_spans if min(b, x["end"]) - max(a, x["start"]) >= MIN_OVERLAP)

    out = []
    for i in bad:
        a, b = sent_spans[i - 1]
        hit = [x for x in labels if min(b, x["end"]) - max(a, x["start"]) >= MIN_OVERLAP]
        if hit and all(n_cover(x) >= COARSE_SENTS for x in hit):
            out.append(i)
    return sorted(out)


def convert(src: Path, val_ratio: float, seed: int) -> tuple[dict[str, list[dict]], dict]:
    sources = {d["source_id"]: d for d in map(json.loads, open(src / "source_info.jsonl", encoding="utf-8"))}
    stats = collections.Counter()
    train_sources = sorted({json.loads(l)["source_id"] for l in open(src / "response.jsonl", encoding="utf-8")
                            if json.loads(l)["split"] == "train"})
    rng = random.Random(seed)
    val_sources = set(rng.sample(train_sources, int(len(train_sources) * val_ratio)))
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for line in open(src / "response.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("quality", "good") != "good":
            stats["dropped_quality"] += 1
            continue
        src_d = sources[r["source_id"]]
        sent_spans = split_sentences(r["response"])
        bad = label_sentences(sent_spans, r["labels"])
        if r["labels"] and not bad:
            stats["spans_unmatched"] += 1
        split = r["split"] if r["split"] == "test" else ("val" if r["source_id"] in val_sources else "train")
        out[split].append({
            "id": f"rt-{r['id']}", "split": split, "source_id": r["source_id"],
            "task_type": src_d["task_type"], "model": r["model"], "source": src_d["prompt"],
            "response": r["response"],
            "sentences": [r["response"][a:b] for a, b in sent_spans],
            "sent_spans": sent_spans,
            "bad": {str(k): v for k, v in bad.items()},
            "coarse": coarse_sentences(sent_spans, r["labels"], bad),
            "label": "hallucination" if bad else "faithful",
            "spans": [[x["start"], x["end"], x["label_type"]] for x in r["labels"]],
            # 标注员备注（如「No mention of this in the passages」），补写 evidence / reason 时作线索
            "span_notes": [[x["start"], x["end"], " ".join((x.get("meta") or "").split())] for x in r["labels"]
                           if (x.get("meta") or "").strip()],
            "origin": "human",
            "meta": {"implicit_true": any(x.get("implicit_true") for x in r["labels"]),
                     "due_to_null": any(x.get("due_to_null") for x in r["labels"])},
        })
    return out, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="external_data/ragtruth/dataset")
    ap.add_argument("--out", default="outputs/external/ragtruth")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub_frac", type=float, default=0.1, help="小训练集占 train 的源比例")
    args = ap.parse_args()
    out, stats = convert(Path(args.src), args.val_ratio, args.seed)
    for split, rows in out.items():
        n_bad = sum(r["label"] == "hallucination" for r in rows)
        n_sent = sum(len(r["sentences"]) for r in rows)
        n_bad_sent = sum(len(r["bad"]) for r in rows)
        print(f"{split}: {write_jsonl(f'{args.out}/{split}.jsonl', rows)} responses, {n_bad} hallucinated; "
              f"{n_sent} sentences, {n_bad_sent} hallucinated")
    print(stats)
    # 实验用的小训练集：按源抽 sub_frac（同一源的 6 条回答一起进），其余作为备用池
    # （「同等数量真人标注」对照组、合成用的正确回答都从池里取，不和小训练集重叠）
    train = out["train"]
    srcs = sorted({r["source_id"] for r in train})
    sub = set(random.Random(args.seed).sample(srcs, round(len(srcs) * args.sub_frac)))
    for name, rows in (("train_sub", [r for r in train if r["source_id"] in sub]),
                       ("train_pool", [r for r in train if r["source_id"] not in sub])):
        n_bad = sum(r["label"] == "hallucination" for r in rows)
        print(f"{name}: {write_jsonl(f'{args.out}/{name}.jsonl', rows)} responses, {n_bad} hallucinated")


if __name__ == "__main__":
    main()
