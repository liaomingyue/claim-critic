"""Label Studio 导入导出。标注界面配置见 annotation/label_studio.xml。

export：待标注 claim JSONL → Label Studio 任务 JSON（每个标注者各建一个项目，互不可见）。
import：Label Studio 导出的 JSON → 本项目样本格式的 JSONL（一个标注者一个文件），供 kappa.py 使用。

用法：
  python -m annotation.ls_io export --claims outputs/test_claims.jsonl --out tasks.json
  python -m annotation.ls_io import --ls_export project-a.json --out ann_a.jsonl
"""
from __future__ import annotations

import argparse
import json

from common.glossary import default_glossary
from common.schema import cited_ids, read_jsonl, write_jsonl


def to_task(c: dict) -> dict:
    """参考文献标出「被引用 / 对照」，附术语表，和 critic 看到的输入一致（common/prompts.py）。"""
    cited = cited_ids(c)
    refs_text = "\n\n".join(f"({rid}) [{'被引用' if rid in cited else '对照'}] {t}" for rid, t in c["refs"].items())
    glossary = "\n".join(f"{t.en} = {t.zh}" for t in default_glossary().hits(c["refs"], c["claim"]))
    return {"data": {"id": c["id"], "claim": c["claim"], "refs_text": refs_text,
                     "cited_text": ", ".join(cited) or "无（句子没有标注引用）",
                     "glossary_text": glossary or "（无）",
                     "refs_json": json.dumps(c["refs"], ensure_ascii=False),
                     "cited_json": json.dumps(cited),
                     "source_group": c["source_group"]}}


def from_annotation(task: dict) -> dict | None:
    if not task.get("annotations"):
        return None
    d = task["data"]
    result = {r["from_name"]: r["value"] for r in task["annotations"][0]["result"]}
    label = result["verdict"]["choices"][0]
    type_ = result["type"]["choices"][0] if label == "hallucination" and "type" in result else "none"
    ev_ref = result.get("evidence_ref", {}).get("choices", [""])[0]
    ev_text = result.get("evidence_text", {}).get("text", [""])[0]
    if type_ == "citation_error" and not (ev_ref and ev_text):
        evidence = "none"  # 引用错误且哪篇都没有出处：证据留空
    else:
        evidence = f'{ev_ref}: "{ev_text}"' if ev_ref and ev_text else ""
    return {
        "id": d["id"], "split": "test", "source_group": d["source_group"],
        "refs": json.loads(d["refs_json"]), "cited": json.loads(d.get("cited_json", "null")), "claim": d["claim"],
        "label": label, "type": type_,
        "evidence": evidence,
        "reason": result.get("reason", {}).get("text", [""])[0],
        "origin": "human",
        "meta": {"annotator": task["annotations"][0].get("completed_by")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("--claims")
    ap.add_argument("--ls_export")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "export":
        tasks = [to_task(c) for c in read_jsonl(args.claims)]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=1)
        print(f"wrote {len(tasks)} tasks")
    else:
        with open(args.ls_export, encoding="utf-8") as f:
            tasks = json.load(f)
        rows = [r for r in map(from_annotation, tasks) if r]
        print(f"wrote {write_jsonl(args.out, rows)} annotations ({len(tasks) - len(rows)} unannotated)")


if __name__ == "__main__":
    main()
