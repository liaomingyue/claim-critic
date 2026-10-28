"""汇总各台机器的实验记录（common.manifest 写的 outputs/runs/registry.jsonl）到本地，生成 docs/runs.md。

逐台连接 outputs/autodl/web_instance.json 和 extra_*.json 对应的实例（只连在运行中的；关机的机器跳过，
它们的记录在开机后再收），把远端 registry 合并进本地 outputs/runs/registry_all.jsonl（按 名称+完成时间 去重），
再按时间生成一张总表：每次训练 / 评测的配置要点、数据指纹、代码指纹、耗时和结果，复现时照着表找 manifest。

用法：python -m scripts.collect_runs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import paramiko

from scripts.autodl_web import Web

LOCAL = Path("outputs/runs/registry_all.jsonl")
REMOTE = "/root/autodl-tmp/critic_model/outputs/runs/registry.jsonl"


def fetch(info: dict) -> list[dict]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(info["proxy_host"], port=info["ssh_port"], username="root", password=info["root_password"], timeout=30)
    try:
        with c.open_sftp().open(REMOTE) as f:
            rows = [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]
    except OSError:
        rows = []
    c.close()
    for r in rows:
        r["machine"] = info["uuid"]
    return rows


def short(d: dict) -> str:
    """数据指纹的一行摘要：文件名 · 行数 · sha 前 8 位。"""
    if d.get("missing"):
        return f"{Path(d['path']).name}（缺失）"
    return f"{Path(d['path']).name} · {d.get('rows', '?')} 行 · {(d.get('content_sha') or d['sha256'])[:8]}"


def table(rows: list[dict]) -> str:
    out = ["# 实验运行登记", "", "由 `python -m scripts.collect_runs` 生成，每行对应一份 manifest（产出目录里的 "
           "run_manifest.json / eval_manifest.json）。复现时按 manifest 里的完整配置、数据 sha256 和代码指纹核对。", "",
           "| 完成时间 | 类型 | 名称 | 数据 | 关键配置 | 耗时 | 结果 | 机器 |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("finished") or ""):
        cfg = r.get("config", {})
        if r["kind"] == "run":
            tr = cfg.get("trainer", {})
            key = (f"{Path(str(r.get('base_model', ''))).name} · style {cfg.get('style', 'full')} · "
                   f"lr {tr.get('learning_rate')} · ep {tr.get('num_train_epochs')} · "
                   f"bs {tr.get('per_device_train_batch_size')}×{tr.get('gradient_accumulation_steps')}")
            t = r.get("train", {})
            res = " · ".join(x for x in (f"loss {t['final_loss']}" if t.get("final_loss") else "",
                                         f"{t['steps']} 步" if t.get("steps") else "") if x) or "—"
        else:
            key = f"style {cfg.get('style')} · resp_style {cfg.get('resp_style')} · max_tokens {cfg.get('max_tokens')}"
            m = r.get("metrics", {})
            s, rr = m.get("sentence", {}), m.get("response", {})
            res = (f"句 P/R/F1 {s.get('precision', 0):.3f}/{s.get('recall', 0):.3f}/{s.get('f1', 0):.3f} · "
                   f"回答 R {rr.get('recall', 0):.3f}") if s else ""
        data = "<br>".join(short(d) for d in r.get("data", []))
        when = r.get("finished") or ("补录" if r.get("backfilled") else "—")
        dur = f"{r['duration_min']} 分钟" if r.get("duration_min") else "—"
        out.append(f"| {when} | {r['kind']} | {r['name']} | {data} | {key} | {dur} | {res} | {str(r.get('machine', ''))[:14]} |")
    return "\n".join(out) + "\n"


def main() -> None:
    web = Web("autodl_web.json")
    status = {i["uuid"]: i["status"] for i in web.instances()}
    seen = {}
    if LOCAL.exists():
        for line in LOCAL.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            seen[(r["name"], r["kind"], r.get("finished"))] = r
    for f in sorted(Path("outputs/autodl").glob("*.json")):
        if not (f.name == "web_instance.json" or f.name.startswith("extra_")) or "failed" in f.name:
            continue
        info = json.loads(f.read_text(encoding="utf-8"))
        if status.get(info.get("uuid")) != "running":
            print(f"skip {f.name}: {status.get(info.get('uuid'), 'gone')}")
            continue
        rows = fetch(info)
        for r in rows:
            seen[(r["name"], r["kind"], r.get("finished"))] = r
        print(f"{f.name}: {len(rows)} records")
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    LOCAL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in seen.values()), encoding="utf-8")
    Path("docs/runs.md").write_text(table(list(seen.values())), encoding="utf-8")
    print(f"{len(seen)} records -> {LOCAL}, docs/runs.md")


if __name__ == "__main__":
    main()
