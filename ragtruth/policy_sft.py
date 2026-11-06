"""回答策略的两组 SFT 对照（2026-11-05，用户：先把 S2 拒绝采样 SFT 跑出来；S1 人工（Claude）修正、不用 critic 也要做；两组比较看重）。
两组和回答 GRPO 用同一批数据：outputs/rt/minicpm/grpo/train_ref.jsonl 的 831 个 prompt（同原文、同 QA 系统提示、同聊天模板）。

S2 拒绝采样 SFT（不调 Claude）：
  s2_sample  v0（mc-init）每个 prompt 采 16 条（同 GRPO：温度 1.0、top_p 1.0、max_tokens 1024，发 token id），
             用和 R3 GRPO 完全相同的奖励打分（同一 critic、去底噪计数 K = max(2, 0.5 × 参考句数)、满意度只罚 0.8 ×、拒答 0.4 × P、硬规则）
  s2_build   每个 prompt 取奖励最高的一条（奖励 ≤ 0 的 prompt 丢掉，记数）→ sft_train.jsonl
S1 Claude 修正 SFT（不用 critic）：
  s1_fix     起点 = 每个 prompt 的参考回答（v0 采样里长度居中的那条，满意度也拿它比）。Claude 按 gold v1 的严格口径找错句，
             只返回错句的改写（最小改动、忠实于原文；改不了就删）；该答却拒答的整条重写；正确的句子不返回。多条回答一次调用、按 id 对齐。
  s1_build   程序按句子区间把改写替换回原回答，没错的句子原样保留 → sft_train.jsonl

  python -m ragtruth.policy_sft s2_sample --url http://127.0.0.1:8000/v1 --tokenizer <MiniCPM> --out outputs/rt/minicpm/s2
  python -m ragtruth.policy_sft s2_build --out outputs/rt/minicpm/s2
  python -m ragtruth.policy_sft s1_fix --out outputs/rt/minicpm/s1 --per_call 4 --max_calls 230     # 本地，Claude
  python -m ragtruth.policy_sft s1_build --tokenizer <MiniCPM> --out outputs/rt/minicpm/s1
"""
from __future__ import annotations

import argparse
import collections
import json
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.convert import split_sentences

PROMPTS = "outputs/rt/minicpm/grpo/train_ref.jsonl"
EOS = "<|im_end|>"


def rows_of(path: str = PROMPTS) -> list[dict]:
    return list(read_jsonl(path))


def sft_row(tok, r: dict, answer: str, **meta) -> dict:
    from ragtruth.ans_data import policy_prompt
    return {"id": r["id"], "task_type": r["task_type"], "prompt": policy_prompt(tok, r, True), "completion": answer.strip() + EOS, **meta}


# ── S2 ──
def r3_args(a) -> types.SimpleNamespace:
    """和 run_r3full.sh 同一套奖励参数。"""
    return types.SimpleNamespace(critic="critic", critic_report=f"outputs/rt/rm/{a.critic}/eval/report.json", vllm_url=a.url, rm_concurrency=64,
                                 sat_model="sat", sat_kind="pair", sat_report="outputs/rt/rm/sat_pair__mc/eval/report.json", sat_w=0.8,
                                 refuse_model="refuse", refuse_report="outputs/rt/rm/refuse/eval/report.json", refuse_w=0.4,
                                 reward_shape="count", count_k=3.0, lam_any=0.5, lam_cnt=0.5, sigma_floor=0.1, count_ratio=0.5, count_min=2.0)


def cmd_s2_sample(a) -> None:
    from transformers import AutoTokenizer
    from common.llm import map_resumable
    from ragtruth.ans_data import policy_prompt
    from train.ans_grpo import build_rewarder
    from train.gen_grpo import Server
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    srv, rw = Server(a.url), build_rewarder(r3_args(a))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def one(r: dict) -> dict:
        ids = tok(policy_prompt(tok, r, True))["input_ids"]
        gens = srv.sample(a.model, ids, a.n, 1024, 1.0)
        sc = rw.score_group(r, gens)
        return {"id": r["id"], "samples": [{"text": g["text"], "finish": g["finish"], **{k: s.get(k) for k in ("reward", "why", "ps", "sat", "p_refuse", "refusal", "chars", "n_sent")}}
                                           for g, s in zip(gens, sc)]}

    map_resumable(rows_of(), one, out / "samples.jsonl", concurrency=a.concurrency)
    print("S2_SAMPLE_DONE", sum(1 for _ in read_jsonl(out / "samples.jsonl")), flush=True)


def cmd_s2_build(a) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    rows = {r["id"]: r for r in rows_of()}
    keep, why = [], collections.Counter()
    for x in read_jsonl(out / "samples.jsonl"):
        best = max(x["samples"], key=lambda s: s["reward"] if s["reward"] is not None else -1)
        if not best["reward"] or best["reward"] <= 0:
            why["no_positive"] += 1
            continue
        why["refusal" if best["why"] == "refusal" else "answer"] += 1
        keep.append(sft_row(tok, rows[x["id"]], best["text"], reward=best["reward"], best_why=best["why"],
                            group_mean=sum(s["reward"] or 0 for s in x["samples"]) / len(x["samples"])))
    (out / "sft_train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    m = {"prompts": len(rows), "kept": len(keep), "by": dict(why), "mean_best_reward": round(sum(r["reward"] for r in keep) / max(1, len(keep)), 4),
         "mean_group_reward": round(sum(r["group_mean"] for r in keep) / max(1, len(keep)), 4), "task": dict(collections.Counter(r["task_type"] for r in keep))}
    (out / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print("S2_BUILD", json.dumps(m, ensure_ascii=False), flush=True)


# ── S1 ──
FIX_SYSTEM = """You correct hallucinations in responses written from a source. Compare every sentence of each response with its source yourself, using the strict standard below, and return corrections ONLY for the sentences that are wrong.

A sentence is wrong whenever it says anything the source does not state and that cannot be directly inferred from it; when unsure, treat it as wrong. Wrong: changed or invented numbers, dates, names, quantities; facts tied to the wrong person, entity or item; changed certainty or scope; added concrete details, evaluations, purposes, consequences or recommendations the source does not make, including vague add-ons; near-synonyms that shift the meaning; details from different passages joined into a claim the source does not make. Not wrong: true synonyms, faithful summaries and reorderings, leaving information out, what follows directly from the source, saying truthfully that the source does not give some information.

For each wrong sentence give the smallest rewrite that makes it fully supported by the source, keeping its role, length and style in the response; if nothing supported can be said in its place, give an empty string to delete it. Never touch correct sentences.
If a response refuses ("unable to answer") but the source does contain information that answers the question, give a "rewrite": a complete faithful answer in the same style and length as a normal response. If the refusal is justified, leave it alone.

Each response is in its own <item id="..."> block with its own source and numbered sentences.
Reply with one JSON object only: {"items": [{"id": "...", "fixes": [{"idx": <sentence number>, "new": "<rewritten sentence or empty>"}], "rewrite": null or "<full answer>"}, ...]} with one entry per item (fixes empty when nothing is wrong)."""


def fix_text(x: dict) -> str:
    sents = "\n".join(f"[S{k}] {s}" for k, s in enumerate(x["sentences"], 1))
    return f'<item id="{x["id"]}">\n<source>\n{x["source"].strip()}\n</source>\n<response>\n{sents}\n</response>\n</item>'


def valid_fix(y, n: int) -> dict | None:
    if not isinstance(y, dict) or not isinstance(y.get("fixes"), list):
        return None
    fx = []
    for f in y["fixes"]:
        if not isinstance(f, dict) or not isinstance(f.get("idx"), int) or not 1 <= f["idx"] <= n or not isinstance(f.get("new"), str):
            return None
        fx.append({"idx": f["idx"], "new": f["new"].strip()})
    rw = y.get("rewrite")
    return {"fixes": fx, "rewrite": rw.strip() if isinstance(rw, str) and rw.strip() else None}


def splice(text: str, fixes: list[dict]) -> str:
    """按句子区间把改写换回原回答：没改的句子（连同它们之间的空白、换行）原样保留；改成空串的句子连同它后面的空白一起删。"""
    t = text.strip()
    spans = split_sentences(t)
    new = {f["idx"]: f["new"] for f in fixes}
    out, pos = [], 0
    for k, (a_, b) in enumerate(spans, 1):
        out.append(t[pos:a_])
        if k in new:
            if new[k]:
                out.append(new[k])
            else:   # 删句：吞掉紧跟的空白，免得留下双空格
                nxt = spans[k][0] if k < len(spans) else len(t)
                pos = nxt
                continue
        else:
            out.append(t[a_:b])
        pos = b
    out.append(t[pos:])
    return "".join(out).strip()


def cmd_s1_fix(a) -> None:
    from common.llm import LLM, LLMConfig, kept, map_resumable
    from ragtruth.synth import parse_json_obj
    from train.ans_reward import sentences
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xs = [{"id": r["id"], "source": r["source"], "sentences": sentences(r["ref_text"].strip())} for r in rows_of() if r.get("ref_text")]
    calls = [{"id": f"f{k // a.per_call:04d}", "xs": xs[k:k + a.per_call]} for k in range(0, len(xs), a.per_call)]
    assert len(calls) <= a.max_calls, f"{len(calls)} calls > max_calls {a.max_calls}"
    if a.limit_calls:
        calls = calls[:a.limit_calls]
        xs = [x for c in calls for x in c["xs"]]
    llm = LLM(LLMConfig(model=a.claude, provider="claude_cli", effort=a.effort, concurrency=a.concurrency))

    def run(c: dict) -> dict:
        o = parse_json_obj(llm.chat_one([{"role": "system", "content": FIX_SYSTEM}, {"role": "user", "content": "\n\n".join(fix_text(x) for x in c["xs"])}]))
        got = {str(y.get("id")): y for y in (o.get("items") or []) if isinstance(y, dict)} if isinstance(o, dict) else {}
        rows = [{"id": x["id"], **v} for x in c["xs"] if (v := valid_fix(got.get(x["id"]), len(x["sentences"]))) is not None]
        return {"id": c["id"], "rows": rows, "missing": [x["id"] for x in c["xs"] if x["id"] not in {r["id"] for r in rows}]}

    map_resumable(calls, run, out / "fix_raw.jsonl", concurrency=a.concurrency)
    got = {r["id"] for b in kept(out / "fix_raw.jsonl") for r in b["rows"]}
    miss = [x for x in xs if x["id"] not in got]
    if miss:   # 漏回的单条重标
        assert len(miss) <= a.max_single, f"{len(miss)} missing > max_single {a.max_single}"
        map_resumable([{"id": f"s-{x['id']}", "xs": [x]} for x in miss], run, out / "fix_single_raw.jsonl", concurrency=a.concurrency)
    print("S1_FIX_DONE", json.dumps({"answers": len(xs), "calls": len(calls), "single": len(miss)}), flush=True)


def cmd_s1_build(a) -> None:
    from transformers import AutoTokenizer
    from common.llm import kept
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    fixes = {}
    for f in ("fix_raw.jsonl", "fix_single_raw.jsonl"):
        if (out / f).exists():
            for b in kept(out / f):
                for r in b["rows"]:
                    fixes[r["id"]] = r
    keep, c = [], collections.Counter()
    for r in rows_of():
        if not r.get("ref_text") or r["id"] not in fixes:
            c["no_fix"] += 1
            continue
        f = fixes[r["id"]]
        if f["rewrite"]:
            ans, kind = f["rewrite"], "rewrite"
        elif f["fixes"]:
            ans, kind = splice(r["ref_text"], f["fixes"]), "edited"
        else:
            ans, kind = r["ref_text"].strip(), "unchanged"
        if not ans:
            c["empty"] += 1
            continue
        c[kind] += 1
        c["sent_fixed"] += len(f["fixes"])
        keep.append(sft_row(tok, r, ans, kind=kind, n_fix=len(f["fixes"])))
    (out / "sft_train.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep), encoding="utf-8")
    m = {"prompts": len(rows_of()), "kept": len(keep), **dict(c), "task": dict(collections.Counter(x["task_type"] for x in keep))}
    (out / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print("S1_BUILD", json.dumps(m, ensure_ascii=False), flush=True)


def cmd_s1_targets(a) -> None:
    """SFT 上限（2026-11-05）：把 S1 的训练目标（拼回后的回答）和原参考回答各写成验收回答格式，给 9B 打分 —— 模仿学习最多学到目标这么干净。"""
    from common.llm import kept
    from train.ans_reward import sentences
    out = Path(a.out)
    fixes = {}
    for f in ("fix_raw.jsonl", "fix_single_raw.jsonl"):
        if (out / f).exists():
            for b in kept(out / f):
                for r in b["rows"]:
                    fixes[r["id"]] = r
    rows = {"orig": [], "tgt": []}
    for r in rows_of():
        if not r.get("ref_text") or r["id"] not in fixes:
            continue
        f = fixes[r["id"]]
        tgt = f["rewrite"] or (splice(r["ref_text"], f["fixes"]) if f["fixes"] else r["ref_text"].strip())
        for k, ans in (("orig", r["ref_text"].strip()), ("tgt", tgt)):
            if ans:
                rows[k].append({"id": r["id"], "source_id": r["source_id"], "task_type": r["task_type"], "source": r["source"],
                                "response": ans, "sentences": sentences(ans), "finish": "stop"})
    d = out / "cap"; d.mkdir(parents=True, exist_ok=True)
    for k, xs in rows.items():
        (d / f"{k}_t.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in xs), encoding="utf-8")
    print("S1_TARGETS", {k: len(v) for k, v in rows.items()}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["s2_sample", "s2_build", "s1_fix", "s1_build", "s1_targets"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="policy_v0")
    ap.add_argument("--critic", default="rm1__mcgold_h3ctl")
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/MiniCPM5-2B")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--per_call", type=int, default=10)   # 用户：一次多丢几条给 Claude，减少总调用次数（4 条约 210 次 → 10 条约 84 次）
    ap.add_argument("--max_calls", type=int, default=90)
    ap.add_argument("--limit_calls", type=int, default=0, help="只跑前 N 次调用（试跑看质量）")
    ap.add_argument("--max_single", type=int, default=40)
    ap.add_argument("--claude", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    a = ap.parse_args()
    {"s2_sample": cmd_s2_sample, "s2_build": cmd_s2_build, "s1_fix": cmd_s1_fix, "s1_build": cmd_s1_build, "s1_targets": cmd_s1_targets}[a.cmd](a)


if __name__ == "__main__":
    main()
