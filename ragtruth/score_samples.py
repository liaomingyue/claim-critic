"""给一批改写样本同时打 rm1（critic）、rm2（自然度）、sev（dne / gen）分，之后本地对照 Claude 标注看各模型靠不靠得住。

样本文件：outputs/rt/core/<子目录>/items.jsonl（gen_v2_label 格式：aid、source、context、idx、original、sentence、mechanism 等）。
  rm1  rm1__core   RM1_SYSTEM，选项 ABC，p_halluc = P(B) + P(C)
  rm2  rm2__core   RM2G_SYSTEM + discrim.render，选项 AB，p_natural = P(A)
  dne  sev__gen_dne  sev_gen.prompt_dne（不给解释），选项 ABC，p_err = P(A) + P(B)，p_clear = P(A)
  gen  sev__gen    sev_gen.prompt（v2 采样没有证据 / 理由，给空的，和训练分布不同，只作参考），p_err = P(A–D)
  python -m ragtruth.score_samples --items gen_v2_label/items.jsonl gen_v2_label_r2/items.jsonl --temps rm1=1.0 rm2=1.0 dne=1.4 gen=1.2 --out gen_v2_label/model_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--temps", nargs="+", required=True, help="rm1=T rm2=T dne=T gen=T（vLLM 上的服务名同键名）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--desc", default="", help="v3 用的改动描述文件（outputs/rt/core 下，id = aid）")
    a = ap.parse_args()
    from openai import OpenAI
    from ragtruth import discrim
    from ragtruth.rm_data import RM1_SYSTEM, RM2G_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.sev_gen import prompt as prompt_gen, prompt_dne, prompt_v3
    from ragtruth.task import render_input
    from train.gen_reward import NO_THINK
    T = {k: float(v) for k, v in (t.split("=") for t in a.temps)}
    cl = OpenAI(base_url=a.url, api_key="x", timeout=600, max_retries=3)
    desc = {r["id"]: r["desc"] for r in read_jsonl(D / a.desc) if r.get("ok")} if a.desc else {}
    items = {}
    for f in a.items:
        for x in read_jsonl(D / f):
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            items[x["aid"]] = {**x, "context": ctx, "idx": int(x["idx"]), "evidence": x.get("evidence") or "", "reason": x.get("reason") or "",
                               "type": x.get("type") or "conflict", "desc": desc.get(x["aid"], "")}

    def ask(model: str, msgs: list[dict], opts: str) -> dict | None:
        r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], opts)
        return probs(lp, T[model]) if lp else None

    def one(x: dict) -> dict:
        out = {"aid": x["aid"]}
        if "rm1" in T:
            p = ask("rm1", [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": x["source"], "sentences": x["context"], "idx": x["idx"]})}], "ABC")
            out["rm1_p_halluc"] = (p["B"] + p["C"]) if p else None
        if "rm2" in T:
            p = ask("rm2", [{"role": "system", "content": RM2G_SYSTEM}, {"role": "user", "content": discrim.render(x["source"], x["context"], x["idx"])}], "AB")
            out["rm2_p_natural"] = p["A"] if p else None
        if "dne" in T:
            p = ask("dne", prompt_dne(x), "ABC")
            out["dne_p_err"], out["dne_p_clear"] = ((p["A"] + p["B"]), p["A"]) if p else (None, None)
        if "rm2v2" in T:
            p = ask("rm2v2", [{"role": "system", "content": RM2G_SYSTEM}, {"role": "user", "content": discrim.render(x["source"], x["context"], x["idx"])}], "AB")
            out["rm2v2_p_natural"] = p["A"] if p else None
        if "rm2v3" in T:   # RM2 第三版：学 Claude 的「自然」（ragtruth.rm2_v3）
            from ragtruth.rm2_v3 import prompt as prompt_rm2v3
            p = ask("rm2v3", prompt_rm2v3(x), "AB")
            out["rm2v3_p_natural"] = p["A"] if p else None
        if "v3" in T:
            p = ask("v3", prompt_v3(x), "ABCD")
            out["v3_p_req"], out["v3_p_err"] = ((p["A"] + p["B"]), (p["A"] + p["B"] + p["C"])) if p else (None, None)
        if "gen" in T:
            p = ask("gen", prompt_gen(x), "ABCDE")
            out["gen_p_err"] = sum(p[k] for k in "ABCD") if p else None
        return out

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, sorted(items.values(), key=lambda x: x["aid"])))
    Path(D / a.out).parent.mkdir(parents=True, exist_ok=True)
    (D / a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res), encoding="utf-8")
    print("SCORE_SAMPLES", len(res), {k: sum(r.get(k) is None for r in res) for k in res[0] if k != "aid"}, flush=True)


if __name__ == "__main__":
    main()
