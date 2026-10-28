"""从 Europe PMC 拉 PMC 开放获取子集里 CC-BY 许可的论文，按主题分成每组 3–8 篇。

每篇保留摘要；加 --with_sections 时再取全文 XML 里的 Results / Conclusion(s) 段。

用法：python -m data.fetch_pmc --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from common.config import load_config
from common.schema import write_jsonl

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _get(url: str, retries: int = 4) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"[fetch] {e}; retry")
            time.sleep(2 ** i)
    return b""


FDA_API = "https://api.fda.gov/drug/label.json"
OPEN_FILTER = 'OPEN_ACCESS:y AND LICENSE:"cc by" AND HAS_ABSTRACT:y'
LABEL_SECTIONS = ("indications_and_usage", "dosage_and_administration", "contraindications",
                  "warnings_and_cautions", "warnings", "adverse_reactions", "drug_interactions")


def topic_core(topic: str) -> str:
    """主题 → Europe PMC 检索式主体。普通词组按「各词都出现在标题或摘要」处理（整句加引号要求原样连续出现，
    几乎搜不到）；含引号或字段名（如 TITLE_ABS:）的视为写好的检索式，原样使用。"""
    if '"' in topic or ":" in topic:
        return topic
    return " AND ".join(f"TITLE_ABS:{w}" for w in topic.split())


def topic_query(topic: str) -> str:
    return f"({topic_core(topic)}) AND {OPEN_FILTER}"


def search(topic: str, limit: int) -> list[dict]:
    return search_core(topic_core(topic), limit)


def clean_text(s: str) -> str:
    """Europe PMC 的标题和摘要里有 HTML 标签和转义实体（&lt;sup&gt;）：先反转义再去标签。"""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(s or ""))).strip()


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def parse_results(results: list[dict], seen: set[str]) -> list[dict]:
    """一页检索结果 → 论文列表。只要 CC-BY；跳过撤稿；按 pmcid 和标题去重（同一篇常有多个版本）。"""
    papers = []
    for r in results:
        title = clean_text(r.get("title", ""))
        if (r.get("license") or "").lower() != "cc by" or not r.get("pmcid"):
            continue
        if title.upper().startswith(("RETRACTED", "RETRACTION")):
            continue
        keys = {r["pmcid"], _norm_title(title)}
        if keys & seen:
            continue
        seen |= keys
        papers.append({"pmcid": r["pmcid"], "title": title,
                       "abstract": clean_text(r.get("abstractText", "")),
                       "year": r.get("pubYear"), "license": r["license"],
                       "pub_types": (r.get("pubTypeList") or {}).get("pubType", [])})
    return papers


def search_core(core: str, limit: int) -> list[dict]:
    """按检索式主体搜 CC-BY 开放获取论文，结果按 Europe PMC 的相关度排序。"""
    query = f'({core}) AND {OPEN_FILTER} AND NOT PUB_TYPE:"Retracted Publication"'
    papers, cursor, seen = [], "*", set()
    while len(papers) < limit:
        params = urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core",
                                         "pageSize": 100, "cursorMark": cursor})
        data = json.loads(_get(f"{API}/search?{params}"))
        results = data.get("resultList", {}).get("result", [])
        papers += parse_results(results, seen)
        next_cursor = data.get("nextCursorMark")
        # 最后一页的 nextCursorMark 不变：继续翻会反复取到同一页
        if not results or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.2)
    return papers[:limit]


def fda_label(drug: str, max_chars: int) -> dict | None:
    """openFDA 的药品说明书（美国政府作品，公有领域）。优先单方制剂（只含这一种成分），拼接主要章节。

    按药名检索会命中复方（如西格列汀/二甲双胍），只取第一份会拿到复方说明书。
    """
    params = urllib.parse.urlencode({"search": f'openfda.generic_name:"{drug}"', "limit": 25})
    try:
        results = json.loads(_get(f"{FDA_API}?{params}")).get("results", [])
    except Exception:  # noqa: BLE001 查不到时 openFDA 返回 404
        return None

    def n_ingredients(r: dict) -> int:
        return len((r.get("openfda") or {}).get("substance_name") or ["?", "?"])

    for r in sorted(results, key=n_ingredients):  # 稳定排序：单方在前，同类保持 openFDA 原顺序
        if "indications_and_usage" not in r:
            continue
        parts = [f"{k.replace('_', ' ').upper()}: {' '.join(r[k])}" for k in LABEL_SECTIONS if k in r]
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return {"pmcid": f"FDA:{r.get('set_id', drug)}", "title": f"{drug} drug label (FDA)",
                "abstract": text[:max_chars], "year": (r.get("effective_time") or "")[:4],
                "license": "public domain (FDA SPL)", "pub_types": ["Drug Label"]}
    return None


def result_sections(pmcid: str) -> str:
    try:
        root = ET.fromstring(_get(f"{API}/{pmcid}/fullTextXML"))
    except Exception:  # noqa: BLE001 部分文章没有全文 XML
        return ""
    texts = []
    for sec in root.iter("sec"):
        title = (sec.findtext("title") or "").lower()
        if re.search(r"\bresults?\b|\bconclusions?\b", title):
            texts.append(" ".join(" ".join(p.itertext()) for p in sec.iter("p")))
    return re.sub(r"\s+", " ", " ".join(texts)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--with_sections", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)["fetch"]
    rng = random.Random(cfg["seed"])

    groups = []
    for topic in cfg["topics"]:
        papers = search(topic, cfg["papers_per_topic"])
        print(f"[fetch] {topic}: {len(papers)} papers")
        if args.with_sections:
            for p in papers:
                p["sections"] = result_sections(p["pmcid"])[: cfg["max_section_chars"]]
        rng.shuffle(papers)
        i = 0
        while len(papers) - i >= cfg["min_group"]:
            k = rng.randint(cfg["min_group"], cfg["max_group"])
            chunk = papers[i:i + k]
            i += k
            groups.append({
                "group_id": f"g-{len(groups):05d}",
                "topic": topic,
                "refs": {f"ref-{j + 1}": (p["abstract"] + " " + p.get("sections", "")).strip()
                         for j, p in enumerate(chunk)},
                "papers": [{k2: p[k2] for k2 in ("pmcid", "title", "year", "license")} for p in chunk],
            })

    # 按组划分 train / test，测试集与训练集文献组不重叠
    rng.shuffle(groups)
    n_test = int(len(groups) * cfg["test_group_ratio"])
    for j, g in enumerate(groups):
        g["split"] = "test" if j < n_test else "train"
    print(f"wrote {write_jsonl(cfg['out'], groups)} groups ({n_test} test)")


if __name__ == "__main__":
    main()
