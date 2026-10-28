"""人工审核工作台：标准库 HTTP 服务 + 单页前端（review/static/index.html），无第三方依赖。

浏览流水线各阶段的样本（正样本、各批合成负样本、各批难样本、冷启动集），看证据高亮、与原 claim 的差异、
同一原 claim 在各批的派生情况和 critic 的采样输出，以及每条样本的完整调用流程（每次模型调用的 prompt 和输出）
和各阶段的 prompt 模板；可以通过 / 拒绝、打问题标记、直接修改字段。
审核记录写到 outputs/review/<dataset>.jsonl，build_trainset 组装训练集时自动回流。

用法：python -m review.server --config configs/data.yaml --reviewer 张三 [--port 8765]
"""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from common.config import load_config
from common.glossary import default_glossary
from common.output_format import normalize
from common.prompts import TYPE_DEFINITIONS
from common.schema import ERROR_TYPES, cited_ids, cited_refs, read_jsonl
from review import pipeline_view, store

STATIC = Path(__file__).with_name("static")
TYPE_LABELS = {t: d.split("：")[0] for t, d in TYPE_DEFINITIONS.items()}  # missing_condition -> 前提缺失


class Data:
    """按文件修改时间缓存读取结果，流水线重跑后自动刷新。"""

    def __init__(self, cfg: dict, review_dir: str, max_batch: int):
        self.cfg, self.review_dir = cfg, review_dir
        self.datasets = store.datasets(cfg, max_batch)
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def rows(self, path: str | None) -> list[dict]:
        if not path or not Path(path).exists():
            return []
        mtime = Path(path).stat().st_mtime
        hit = self._cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        rows = [r for r in read_jsonl(path) if not r.get("_dropped")]
        self._cache[path] = (mtime, rows)
        return rows

    def by_id(self, dataset: str) -> dict[str, dict]:
        return {r["id"]: r for r in self.rows(self.datasets[dataset]["path"])}

    def audit_ids(self, dataset: str) -> set[str] | None:
        audit = self.datasets[dataset]["audit"]
        return {r["id"] for r in self.rows(audit)} if audit and Path(audit).exists() else None

    def parent(self, sample: dict) -> dict | None:
        pid = sample.get("parent_id")
        return self.by_id("positives").get(pid) if pid else None

    def lineage(self, sample: dict) -> list[dict]:
        """同一原 claim 在各批合成 / 难样本里派生出的所有负样本。"""
        pid = sample.get("parent_id") or (sample["id"] if sample.get("label") == "faithful" else None)
        if not pid:
            return []
        out = []
        for name, ds in self.datasets.items():
            if not name.startswith(("synth_batch", "hard_batch")):
                continue
            for r in self.rows(ds["path"]):
                if r.get("parent_id") == pid:
                    out.append({"dataset": name, "id": r["id"], "claim": r["claim"], "type": r["type"],
                                "round": r.get("round", 0), "detect_rate": r.get("meta", {}).get("detect_rate")})
        return out


def status_of(rv: dict | None) -> str:
    if rv is None:
        return "todo"
    if rv["decision"] == "reject":
        return "rejected"
    return "edited" if rv.get("edits") else "accepted"


def list_samples(data: Data, q: dict) -> dict:
    name = q["dataset"]
    rows = data.rows(data.datasets[name]["path"])
    reviews = store.load_reviews(data.review_dir, name)
    audit = data.audit_ids(name)
    view, type_, round_, text = q.get("view", "all"), q.get("type", ""), q.get("round", ""), q.get("q", "").lower()
    items = []
    for r in rows:
        st = status_of(reviews.get(r["id"]))
        if view == "audit" and (audit is None or r["id"] not in audit):
            continue
        if view in ("todo", "rejected", "edited", "accepted") and st != view:
            continue
        if type_ and r.get("type") != type_:
            continue
        if round_ != "" and str(r.get("round", 0)) != round_:
            continue
        if text and text not in r["claim"].lower() and text not in r["id"].lower():
            continue
        items.append({"id": r["id"], "claim": r["claim"][:140], "label": r["label"], "type": r.get("type"),
                      "round": r.get("round", 0), "status": st, "audit": bool(audit and r["id"] in audit),
                      "flags": reviews.get(r["id"], {}).get("flags", [])})
    return {"items": items, "summary": store.summary(rows, reviews, audit, data.datasets[name]["threshold"])}


def sample_detail(data: Data, dataset: str, sample_id: str) -> dict:
    s = data.by_id(dataset).get(sample_id)
    if s is None:
        raise KeyError(sample_id)
    rv = store.load_reviews(data.review_dir, dataset).get(sample_id)
    current = {**s, **(rv or {}).get("edits", {})}
    parent = data.parent(s)
    g = default_glossary()
    return {
        "sample": s,
        "current": current,
        "review": rv,
        "stale": bool(rv and rv.get("base_hash") != store.content_hash(s)),
        "history": store.history(data.review_dir, dataset, sample_id),
        "parent": parent,
        "lineage": data.lineage(s),
        "refs_display": {k: normalize(v) for k, v in s["refs"].items()},
        "cited": cited_ids(current),
        "parent_cited": cited_ids(parent) if parent else None,
        # critic 输入里附带的术语表（与 common/prompts.py 同一规则），以及词典预检命中的误译
        "glossary": [{"en": t.en, "zh": t.zh} for t in g.hits(s["refs"], current["claim"])],
        "term_check": [{"en": t.en, "zh": t.zh, "wrong": w} for t, w in g.mistranslations(cited_refs(current), current["claim"], s["refs"])],
        "validation": store.validate(current, parent),
        "flags": [{"key": f, "label": store.FLAG_LABELS[f]} for f in data.datasets[dataset]["flags"]],
    }


def make_handler(data: Data, reviewer: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 安静一点，只打错误
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/datasets":
                    out = []
                    for name, ds in data.datasets.items():
                        rows = data.rows(ds["path"])
                        reviews = store.load_reviews(data.review_dir, name)
                        out.append({"name": name, "path": ds["path"], "exists": Path(ds["path"]).exists(),
                                    "summary": store.summary(rows, reviews, data.audit_ids(name), ds["threshold"])})
                    self._json({"datasets": out, "reviewer": reviewer, "types": list(ERROR_TYPES),
                                "type_labels": TYPE_LABELS})
                elif url.path == "/api/list":
                    self._json(list_samples(data, q))
                elif url.path == "/api/sample":
                    self._json(sample_detail(data, q["dataset"], q["id"]))
                elif url.path == "/api/chain":  # 这条样本从用户问题到最终样本的每一步、每次模型调用
                    self._json(pipeline_view.call_chain(data.cfg, data.by_id(q["dataset"])[q["id"]]))
                elif url.path == "/api/prompts":  # 各阶段的 prompt 模板
                    self._json({"stages": pipeline_view.prompt_catalog()})
                else:
                    self._json({"error": "not found"}, 404)
            except KeyError as e:
                self._json({"error": f"not found: {e}"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = self._body()
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._json({"error": f"请求体不是合法的 UTF-8 JSON：{e}"}, 400)
                return
            try:
                s = data.by_id(body["dataset"])[body["id"]]
            except KeyError as e:
                self._json({"error": f"not found: {e}"}, 404)
                return
            edits = {k: v for k, v in (body.get("edits") or {}).items() if k in store.EDITABLE}
            current = {**s, **edits}
            if current.get("label") == "faithful":
                current["type"] = "none"
                edits["type"] = "none"
            validation = store.validate(current, data.parent(s))
            if url.path == "/api/validate":
                self._json(validation)
                return
            if url.path != "/api/review":
                self._json({"error": "not found"}, 404)
                return
            if body["decision"] == "accept" and validation["errors"]:
                self._json({"error": "校验未通过，不能保存为通过", "validation": validation},
                           HTTPStatus.UNPROCESSABLE_ENTITY)
                return
            rv = store.make_review(s, body["dataset"], body["decision"], body.get("flags", []), edits,
                                   body.get("note", ""), body.get("reviewer") or reviewer)
            store.save_review(data.review_dir, rv)
            self._json({"ok": True, "review": rv, "status": status_of(rv)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max_batch", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = Data(cfg, cfg.get("review", {}).get("dir", "outputs/review"), args.max_batch)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data, args.reviewer))
    print(f"review workbench: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
