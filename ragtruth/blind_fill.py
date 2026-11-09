"""多类型补 critic 盲区（2026-11-09 用户定）。

Claude 抽查发现：盲区里（critic 0.1–0.5、自然、sev P(有错) ≥ 0.5），近义偏移 9/9、张冠李戴 8/10 是真错；
范围偏移和跨段拼接只有约 6/10，要 Claude 核实。用户定：
  - 范围偏移、跨段拼接：按 sev 排序各取前 300 条交 Claude，只留真错 + 自然的；
  - 近义偏移、张冠李戴：做对照，各取前 100 条交 Claude，看核实和不核实的差别；
  - 其他类型这轮不补。
两组 critic（基线 rm1__core_r2，只加错句、不配原句）：
  cl = 范围 / 拼接核实留下的 + 近义 / 张冠李戴核实留下的
  nc = 范围 / 拼接核实留下的 + 近义 / 张冠李戴从同一批里随机取同样条数（不看 Claude）
样本：错误生成器在大提示池（RAG 回答 × 干净句，rag_pool / rag_pool_sftcore）上，每个提示温度 0 只采 1 条，只做这 4 种类型。

  python -m ragtruth.blind_fill sample --model v2h3 --tokenizer $MC --tag v2h3
  python -m ragtruth.blind_fill select          # → blind_fill/items.jsonl（交本地 Claude：gen_v2_label label --out_dir ../gen_v2_grpo/blind_fill）
  python -m ragtruth.blind_fill build           # → rm1__core_r2_bf{cl,nc}
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from common.schema import read_jsonl

O = Path("outputs/rt/gen_v2_grpo")
B = O / "blind_fill"
POOLS = {"g": "rag_pool/prompts.jsonl", "s": "rag_pool_sftcore/prompts.jsonl"}   # g = RAG GRPO 版回答，s = RAG SFT 版回答（两池回答编号会重名，加前缀）
VERIFY = {"certainty_or_scope_shift": 300, "merged_across_passages": 300}   # 只留 Claude 核实过的
CONTROL = {"near_synonym_shift": 100, "wrong_attribution": 100}             # 核实 vs 不核实对照


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_pools() -> list[dict]:
    return [{**r, "id": f"{t}:{r['id']}", "resp_id": f"{t}:{r['resp_id']}"} for t, f in POOLS.items() for r in read_jsonl(O / f)]


def source_of(resp_id: str) -> str:   # "g:rag-<source_id>-<k>" → source_id
    return resp_id.split(":", 1)[1].rsplit("-", 1)[0].replace("rag-", "")


def cmd_sample(a) -> None:
    import zlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import chat_prompt
    from train.gen_reward import load_temps
    from train.gen_reward_v2 import V2SevRewarder
    mechs = set(VERIFY) | set(CONTROL)
    rows = [r for r in load_pools() if r["mechanism"] in mechs]
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=3)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rw = V2SevRewarder(cl, load_temps(a.temps), shape="hard")

    def one(r):
        o = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=512, temperature=0.0,
                                  n=1, seed=zlib.crc32(("blind_fill|" + r["id"]).encode()) % 100000, extra_body={"top_k": -1})
        s = rw.raw(o.choices[0].text, r)
        if s.get("why") != "ok":
            return {"why": s.get("why")}
        return {"why": "ok", "id": r["id"], "resp_id": r["resp_id"], "mechanism": r["mechanism"], "task_type": r["task_type"],
                "clean": s["idx"] in set(r.get("clean_idx") or []), **{k: s[k] for k in ("idx", "original", "sentence", "p_req", "p_err", "p_natural", "p_halluc")}}
    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(one, rows))
    ok = [x for x in out if x["why"] == "ok"]
    write(B / f"samples_{a.tag}.jsonl", ok)
    cand = [x for x in ok if x["clean"] and 0.1 <= x["p_halluc"] < 0.5 and x["p_natural"] >= 0.5 and x["p_err"] >= 0.5]
    print("BLIND_FILL_SAMPLE", json.dumps({"model": a.model, "prompts": len(rows), "ok": len(ok), "why": dict(collections.Counter(x["why"] for x in out)),
                                          "clean": sum(x["clean"] for x in ok), "cand_by_mech": dict(collections.Counter(x["mechanism"] for x in cand))}), flush=True)


def candidates() -> list[dict]:
    """全部 samples_*.jsonl 里：改在干净句上、critic 0.1–0.5、自然 ≥ 0.5、sev P(有错) ≥ 0.5；同一提示同一句去重。"""
    seen, out = set(), []
    for f in sorted(B.glob("samples_*.jsonl")):
        tag = f.stem[len("samples_"):]
        for x in read_jsonl(f):
            if not (x["clean"] and 0.1 <= x["p_halluc"] < 0.5 and x["p_natural"] >= 0.5 and x["p_err"] >= 0.5):
                continue
            k = (x["id"], x["sentence"])
            if k not in seen:
                seen.add(k)
                out.append({**x, "gen": tag})
    return out


def cmd_select(a) -> None:
    """每类按 sev P(有错) 从高到低取前 N；同一原文同一类最多 --per_source 条（同一原文 8 个回答，避免扎堆）。"""
    prompts = {r["id"]: r for r in load_pools()}
    by = collections.defaultdict(list)
    for x in candidates():
        by[x["mechanism"]].append(x)
    items, stat = [], {}
    for m, n in {**VERIFY, **CONTROL}.items():
        xs = sorted(by[m], key=lambda x: (-x["p_err"], x["id"], x["sentence"]))
        per, pick = collections.Counter(), []
        for x in xs:
            sid = source_of(x["resp_id"])
            if per[sid] >= a.per_source:
                continue
            per[sid] += 1
            pick.append(x)
            if len(pick) >= n:
                break
        stat[m] = {"cand": len(xs), "picked": len(pick), "sev_min": round(pick[-1]["p_err"], 3) if pick else None}
        for x in pick:
            p = prompts[x["id"]]
            ctx = list(p["sentences"])
            ctx[x["idx"] - 1] = x["sentence"]
            items.append({"aid": f"{x['gen']}/{x['id']}", "group": "verify" if m in VERIFY else "control", "mechanism": m, "source": p["source"],
                          "context": ctx, "idx": x["idx"], "original": x["original"], "sentence": x["sentence"], "task_type": x["task_type"],
                          "resp_id": x["resp_id"], **{k: x[k] for k in ("p_err", "p_req", "p_halluc", "p_natural")}})
    random.Random("blind_fill|items").shuffle(items)
    write(B / "items.jsonl", items)
    print("BLIND_FILL_SELECT", json.dumps({"items": len(items), "by_mech": stat}), flush=True)


def cmd_build(a) -> None:
    from ragtruth.critic_harvest import row
    from ragtruth.rm_data import OUT, RM1_OPT
    it = {x["aid"]: x for x in read_jsonl(B / "items.jsonl")}
    lab = {r["aid"]: r["claude"] for r in read_jsonl(B / "labels.jsonl") if r["aid"] in it}
    good = lambda k: lab[k].get("severity") in ("clear", "minor") and lab[k].get("natural") == "yes"   # 真错 + 自然（类型用 Claude 判的，不要求等于要求的类型）
    ver = sorted(k for k in lab if it[k]["mechanism"] in VERIFY and good(k))
    ctl_pool = sorted(k for k in lab if it[k]["mechanism"] in CONTROL)   # 只从 Claude 有结论的里取，排除「没返回」带来的差别
    ctl_cl = [k for k in ctl_pool if good(k)]
    ctl_nc = random.Random("blind_fill|nc").sample(ctl_pool, len(ctl_cl))
    stat = {"labeled": len(lab), "verify_kept": len(ver), "control_pool": len(ctl_pool), "control_cl": len(ctl_cl),
            "nc_real": sum(lab[k].get("severity") in ("clear", "minor") for k in ctl_nc), "nc_good": len(set(ctl_nc) & set(ctl_cl)),
            "by_mech": {m: {"labeled": sum(it[k]["mechanism"] == m for k in lab), "good": sum(it[k]["mechanism"] == m and good(k) for k in lab),
                            "real": sum(it[k]["mechanism"] == m and lab[k].get("severity") in ("clear", "minor") for k in lab)} for m in {**VERIFY, **CONTROL}}}
    for arm, keys in (("cl", ver + ctl_cl), ("nc", ver + ctl_nc)):
        add = []
        for k in keys:
            x, c = it[k], lab[k]
            t = c.get("type") if c.get("type") in ("conflict", "baseless") and (arm == "cl" or k in ver) else "conflict"
            sid = source_of(x["resp_id"])
            add.append(row(k.replace("/", "_").replace(":", "_") + "-err", x["source"], x["context"], int(x["idx"]), RM1_OPT[t], x["task_type"], "bf-" + sid, f"blind_fill_{arm}_error"))
        base, d = OUT / a.base, OUT / f"{a.base}_bf{arm}"
        d.mkdir(parents=True, exist_ok=True)
        train = list(read_jsonl(base / "train.jsonl")) + add
        random.Random(f"blind_fill|{arm}|shuffle").shuffle(train)
        write(d / "train.jsonl", train)
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copy(base / sp, d / sp)
        meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "arm": arm, "added_errors": len(keys), **stat,
                "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("BLIND_FILL_BUILD", json.dumps(stat), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "select", "build"])
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="v2h3")
    ap.add_argument("--tag", default="v2h3")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--temps", default="sev=sev__v3c0,rm2v3=rm2__v3b,rm1=rm1__core_r2")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--per_source", type=int, default=3)
    ap.add_argument("--base", default="rm1__core_r2")
    a = ap.parse_args()
    {"sample": cmd_sample, "select": cmd_select, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
