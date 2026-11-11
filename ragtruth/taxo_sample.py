"""错误体系人工审阅：从已有 Claude 金标里汇总全部错误（带原文证据和原因），按原因文本聚类出候选类别，分层抽样给人看。

数据来源（都是已有标注，不新调 Claude）：
  - core：核心集 Claude 修正时标的错误（RAGTruth 各模型回答，原因字段 reason）；
  - gold：金标 v1 的 seed + test（RAGTruth 各模型回答，原因字段 note）；
  - policy：我们自己的 2B / 9B 策略输出上 Claude 复核标出的错误。
聚类：原因 + 错句的 TF-IDF，KMeans（固定种子），每簇给高频词、机制分布、代表例句——只用来让人看案例时把相似的排在一起，不替人下结论。
抽样：按来源分配名额（core 80 / gold 80 / policy 全部，至多 40），来源内按机制轮流抽，保证每类都有。

  python -m ragtruth.taxo_sample → outputs/rt/taxo/review_cases.json（审阅页数据）、clusters.json
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from common.schema import read_jsonl

OUT = Path("outputs/rt/taxo")
RT = Path("external_data/ragtruth/dataset")
QUOTA = {"core": 80, "gold": 80, "policy": 40}
K = 18

# 现有机制 → 新维度的预填建议（只是起点，审阅时改）
PREFILL = {
    "number_or_date": ("number", "substitute"),
    "wrong_attribution": ("entity", "misattribute"),
    "certainty_or_scope_shift": ("degree", "scope"),
    "near_synonym_shift": ("attribute", "substitute"),
    "contradicts_source": ("event", "substitute"),
    "merged_across_passages": ("relation", "merge"),
    "added_specific_fact": ("event", "add"),
    "unsupported_inference": ("relation", "overinfer"),
    "vague_evaluation_or_addon": ("evaluation", "add"),
    "self_description": ("other", "add"),
}


def pool() -> list[dict]:
    prompts = {str(s["source_id"]): s["prompt"] for s in read_jsonl(RT / "source_info.jsonl")}
    out = []
    for c in read_jsonl(Path("outputs/rt/core/core.jsonl")):
        sents = c["sentences"]
        for e in c["labels"]["errors"]:
            i = e["idx"] - 1
            if not (0 <= i < len(sents)):
                continue
            out.append({"id": f"core-{c['source_id']}-{e['idx']}", "origin": "core", "model": "ragtruth", "task_type": c["task_type"],
                        "source_id": c["source_id"], "sentence": sents[i], "prev": sents[i - 1] if i > 0 else "",
                        "type": e["type"], "mechanism": e["mechanism"], "severity": e["severity"],
                        "evidence": e.get("evidence", ""), "reason": e.get("reason", "")})
    seen = set()
    for p, origin in [(Path("outputs/rt/gold/seed_final.jsonl"), "gold"), (Path("outputs/rt/gold/test_final.jsonl"), "gold")] + \
                     [(p, "policy") for p in sorted(Path("outputs/rt/core").glob("claude_check*/final_*.jsonl"))]:
        rows = list(read_jsonl(p))
        by = collections.defaultdict(dict)
        for r in rows:
            by[r["resp_id"]][r["idx"]] = r["sentence"]
        for r in rows:
            key = (r["resp_id"], r["idx"])
            if not r.get("is_error") or key in seen:
                continue
            seen.add(key)
            out.append({"id": f"{origin}-{r['resp_id']}-{r['idx']}", "origin": origin, "model": r.get("model", ""), "task_type": r["task_type"],
                        "source_id": r["source_id"], "sentence": r["sentence"], "prev": by[r["resp_id"]].get(r["idx"] - 1, ""),
                        "type": r["type"], "mechanism": r["mechanism"], "severity": r["severity"],
                        "evidence": r.get("evidence", ""), "reason": r.get("note", "")})
    for x in out:
        x["source"] = prompts.get(str(x["source_id"]), "")
    return [x for x in out if x["source"]]


def cluster(items: list[dict]) -> list[dict]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [f"{x['reason']} {x['sentence']}" for x in items]
    vec = TfidfVectorizer(max_features=4000, stop_words="english", min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    terms = vec.get_feature_names_out()
    info = []
    for k in range(K):
        idx = [i for i, l in enumerate(km.labels_) if l == k]
        top = [terms[j] for j in km.cluster_centers_[k].argsort()[::-1][:8]]
        mech = collections.Counter(items[i]["mechanism"] for i in idx).most_common(4)
        info.append({"cluster": k, "size": len(idx), "terms": top, "mechanisms": mech})
    for x, l in zip(items, km.labels_):
        x["cluster"] = int(l)
    return info


def main() -> None:
    items = pool()
    info = cluster(items)
    rng = random.Random("taxo|v1")
    pick = []
    for origin, q in QUOTA.items():
        by = collections.defaultdict(list)
        for x in items:
            if x["origin"] == origin:
                by[x["mechanism"]].append(x)
        for v in by.values():
            rng.shuffle(v)
        mechs = sorted(by, key=lambda m: -len(by[m]))
        got = []
        while len(got) < q and any(by.values()):
            for m in mechs:
                if by[m] and len(got) < q:
                    got.append(by[m].pop())
        pick += got
    for x in pick:
        x["prefill"] = dict(zip(("object", "operation"), PREFILL.get(x["mechanism"], ("other", "other"))))
        x["source"] = re.sub(r"\n{3,}", "\n\n", x["source"])
    pick.sort(key=lambda x: (x["mechanism"], x["origin"], x["id"]))   # 词袋聚类大多按话题分（烹饪、价格…），对错误类型帮助不大，按现有机制排
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_cases.json").write_text(json.dumps({"cases": pick, "clusters": info}, ensure_ascii=False), encoding="utf-8")
    (OUT / "clusters.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pool", len(items), dict(collections.Counter(x["origin"] for x in items)), "picked", len(pick), dict(collections.Counter(x["origin"] for x in pick)))
    for c in info:
        print(c["cluster"], c["size"], ", ".join(c["terms"][:6]), c["mechanisms"][:3])


if __name__ == "__main__":
    main()
