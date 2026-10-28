"""按用户意图构造文献组：主题 × 意图 → 生成用户问题和检索词 → 检索候选 → 规则筛选 → 模型判相关性 → 一组 3–7 篇。

每组带 question / intent，gen_review 围绕这个问题、按意图的写法写综述。
- 问题由 generator 模型生成（每个「主题 × 意图」一次调用），结果缓存，重跑不重复调用。
- 检索（规则）：所有检索词都要出现在标题或摘要里，按 Europe PMC 相关度取前 candidate_pool 篇；
  不删检索词放宽（删了会混进只沾一个宽泛词的文章），近 N 年的意图不够时只把年份放宽 2 年。
  再按 data.study_type 只留意图要的文献类型，且标题里至少出现一个检索词。
- 相关性（模型）：把剩下的候选（标题 + 摘要开头，最多 relevance_candidates 篇）交给 generator 模型，
  选出能回答这个问题的文献，按相关度排序；结果缓存在 relevance_out，重跑不重复调用。
- 用药意图另加一份 FDA 药品说明书（公有领域，优先单方）作为 ref-1。
- 按组切 train / test 后，从训练组里剔除测试组用到的文献，避免同一篇文献同时出现在两边。

用法：python -m data.build_groups --config configs/data_trial.yaml
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import re

from common.config import load_config
from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import write_jsonl
from data.fetch_pmc import fda_label, result_sections, search_core
from data.intents import INTENTS, question_prompt
from data.study_type import classify


def parse_questions(text: str) -> list[dict]:
    """从模型输出里取出 JSON 数组；格式不对的条目丢掉。"""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        kws = [k.strip() for k in it.get("keywords", []) if isinstance(k, str) and k.strip()]
        if isinstance(it, dict) and it.get("question") and kws:
            out.append({"question": it["question"].strip(), "keywords": kws, "drug": (it.get("drug") or "").strip()})
    return out


def intent_core(keywords: list[str], spec: dict, year: int) -> str:
    """检索式只管主题和年份；文献类型不在这里过滤（见 data.study_type）。"""
    core = " AND ".join(f'TITLE_ABS:"{k}"' if " " in k else f"TITLE_ABS:{k}" for k in keywords)
    if spec["recent_years"]:
        core = f"({core}) AND PUB_YEAR:[{year - spec['recent_years'] + 1} TO {year}]"
    return core


def filter_kinds(papers: list[dict], kinds: list[str]) -> list[dict]:
    """按规则判断文献类型，只留意图要的类型，保持相关度顺序。"""
    out = []
    for p in papers:
        p["kind"] = classify(p["title"], p["abstract"], p.get("pub_types"))
        if p["kind"] in kinds:
            out.append(p)
    return out


def title_hit(paper: dict, keywords: list[str]) -> bool:
    title = paper["title"].lower()
    return any(k.lower() in title for k in keywords)


def retrieve(q: dict, spec: dict, need: int, pool: int, year: int) -> list[dict]:
    """检索词全用上，不删词；按文献类型和「标题含检索词」筛。近 N 年的意图不够 need 篇时把年份放宽 2 年。"""
    widen = [0, 2] if spec["recent_years"] else [0]
    papers: list[dict] = []
    for extra in widen:
        s = {**spec, "recent_years": spec["recent_years"] + extra if spec["recent_years"] else None}
        papers = [p for p in filter_kinds(search_core(intent_core(q["keywords"], s, year), pool), spec["kinds"])
                  if title_hit(p, q["keywords"])]
        if len(papers) >= need:
            break
    return papers


RELEVANCE_PROMPT = """A user asked a medical assistant: {question}
Intent: {label}.

Below are candidate papers (title and the start of the abstract). Select the papers whose content directly helps answer this question. Exclude papers about a different population, disease, intervention or outcome, even if they share words with the question.

{candidates}

Output only a JSON array of the selected paper numbers, most relevant first, e.g. [3, 1, 7]. Output [] if none fit."""


def relevance_prompt(question: str, intent: str, papers: list[dict]) -> str:
    cands = "\n\n".join(f"[{i + 1}] {p['title']} ({p.get('year')})\n{p['abstract'][:600]}" for i, p in enumerate(papers))
    return RELEVANCE_PROMPT.format(question=question, label=INTENTS[intent]["label"], candidates=cands)


def parse_selection(text: str, n: int) -> list[int]:
    """模型输出的编号（1 起）→ 0 起下标；越界、重复的丢掉。"""
    m = re.search(r"\[[\d,\s]*\]", text)
    if not m:
        return []
    out = []
    for x in json.loads(m.group(0)):
        if 1 <= x <= n and x - 1 not in out:
            out.append(x - 1)
    return out


def split_groups(groups: list[dict], test_ratio: float, rng: random.Random, min_group: int) -> list[dict]:
    """按组切分，再从训练组里删掉测试组用过的文献；删完不够 min_group 的训练组丢掉。"""
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_ratio)) if groups else 0
    test_ids = {p["pmcid"] for g in groups[:n_test] for p in g["papers"]}
    out = []
    for j, g in enumerate(groups):
        if j < n_test:
            out.append({**g, "split": "test"})
            continue
        keep = [i for i, p in enumerate(g["papers"]) if p["pmcid"] not in test_ids]
        if len(keep) < min_group:
            continue
        texts = list(g["refs"].values())
        out.append({**g, "split": "train",
                    "papers": [g["papers"][i] for i in keep],
                    "refs": {f"ref-{k + 1}": texts[i] for k, i in enumerate(keep)}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_trial.yaml")
    ap.add_argument("--no_sections", action="store_true", help="只用摘要，不取全文的结果/结论段")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["build_groups"]
    rng = random.Random(bcfg["seed"])
    year = datetime.date.today().year
    llm = LLM(LLMConfig.from_dict(cfg["generator"]))

    jobs = [{"id": f"{topic}|{intent}", "topic": topic, "intent": intent}
            for topic in bcfg["topics"] for intent in bcfg["intents"]]

    def ask(job: dict) -> dict:
        text = llm.chat_one([{"role": "user", "content": question_prompt(job["topic"], job["intent"],
                                                                          bcfg["questions_per_intent"])}])
        return {"topic": job["topic"], "intent": job["intent"], "questions": parse_questions(text), "raw": text}

    map_resumable(jobs, ask, bcfg["questions_out"], concurrency=llm.cfg.concurrency)

    # 1. 检索候选（规则）
    cands = []
    for row in kept(bcfg["questions_out"]):
        spec = INTENTS[row["intent"]]
        for qi, q in enumerate(row["questions"]):
            qid = f"{row['topic']}|{row['intent']}|{qi}"
            label = fda_label(q["drug"], bcfg["max_label_chars"]) if spec["drug_label"] and q["drug"] else None
            if spec["drug_label"] and label is None:
                print(f"[groups] skip (no FDA label for {q['drug']!r}): {q['question']}")
                continue
            need = bcfg["min_group"] - (1 if label else 0)
            papers = retrieve(q, spec, need, bcfg["candidate_pool"], year)[: bcfg["relevance_candidates"]]
            if len(papers) < need:
                print(f"[groups] skip ({len(papers)} candidates): {q['question']}")
                continue
            cands.append({"id": qid, "row": row, "q": q, "label": label, "papers": papers, "need": need})

    # 2. 相关性（模型），结果缓存
    def judge(c: dict) -> dict:
        text = llm.chat_one([{"role": "user", "content": relevance_prompt(c["q"]["question"], c["row"]["intent"],
                                                                           c["papers"])}])
        return {"selected": [c["papers"][i]["pmcid"] for i in parse_selection(text, len(c["papers"]))],
                "n_candidates": len(c["papers"]), "raw": text}

    map_resumable(cands, judge, bcfg["relevance_out"], concurrency=llm.cfg.concurrency)
    selected = {r["id"]: r["selected"] for r in kept(bcfg["relevance_out"])}

    # 3. 组装文献组
    groups, seen_sections = [], {}
    for c in cands:
        row, q, label = c["row"], c["q"], c["label"]
        by_id = {p["pmcid"]: p for p in c["papers"]}
        n_studies = bcfg["max_group"] - (1 if label else 0)
        papers = [by_id[i] for i in selected.get(c["id"], []) if i in by_id][:n_studies]
        if len(papers) < c["need"]:
            print(f"[groups] skip ({len(papers)} relevant of {len(c['papers'])}): {q['question']}")
            continue
        if label:
            papers = [label] + papers
        if not args.no_sections:
            for p in papers:
                if not p["pmcid"].startswith("FDA:"):
                    if p["pmcid"] not in seen_sections:
                        seen_sections[p["pmcid"]] = result_sections(p["pmcid"])[: bcfg["max_section_chars"]]
                    p["sections"] = seen_sections[p["pmcid"]]
        groups.append({
            "group_id": f"g-{len(groups):05d}", "qid": c["id"],
            "topic": row["topic"], "intent": row["intent"], "question": q["question"],
            "keywords": q["keywords"], "drug": q["drug"],
            "refs": {f"ref-{j + 1}": (p["abstract"] + " " + p.get("sections", "")).strip()
                     for j, p in enumerate(papers)},
            "papers": [{**{k: p[k] for k in ("pmcid", "title", "year", "license", "pub_types")},
                        "kind": p.get("kind", "drug_label")} for p in papers],
            "candidates": [{"pmcid": p["pmcid"], "title": p["title"], "year": p.get("year"), "kind": p.get("kind"),
                            "selected": p["pmcid"] in selected.get(c["id"], [])} for p in c["papers"]],
        })
        print(f"[groups] {row['intent']:10s} {len(papers)} papers ({len(c['papers'])} candidates): {q['question']}")

    groups = split_groups(groups, bcfg["test_group_ratio"], rng, bcfg["min_group"])
    n_test = sum(g["split"] == "test" for g in groups)
    print(f"wrote {write_jsonl(bcfg['out'], groups)} groups ({n_test} test)")


if __name__ == "__main__":
    main()
