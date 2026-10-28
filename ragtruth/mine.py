"""难样本挖掘 + 三组对照训练集。

mine：用当前 critic（vLLM 服务，逐句 full 格式）判每条合成的错误句，采样 n 次，记识破率（判为 hallucination 的比例）；
      成对的原句也判一次（贪心），记 critic 是否误报。
mix ：在基础训练集上成对加入合成样本（改后句 + 原句），三组：
      direct = 全部合成样本；critic = 识破率 < threshold 的（critic 没识破的）；random = 从全部合成样本里随机抽，
      条数与 critic 组相同（区分「挑难的」和「多加了这么多条」）。

用法：
  vllm serve /root/autodl-tmp/ckpt/sft-0.6b-e3 --served-model-name critic
  python -m ragtruth.mine mine --synth outputs/rt/synth/r1.jsonl --out outputs/rt/synth/r1_mined.jsonl
  python -m ragtruth.mine mix --arm critic --synth outputs/rt/synth/r1.jsonl --mined outputs/rt/synth/r1_mined.jsonl \
      --out outputs/rt/claims/syn_r1_critic.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from common.llm import LLM, LLMConfig, kept, map_resumable
from common.schema import read_jsonl, write_jsonl
from ragtruth.synth import to_claims
from ragtruth.task import messages, parse_output

ARMS = ("direct", "critic", "random")


def mine(args) -> None:
    critic = LLM(LLMConfig(model=args.model, base_url=args.base_url, max_tokens=512,
                           concurrency=args.concurrency, temperature=args.temperature))
    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    def run(rec: dict) -> dict:
        bad, good = to_claims(rec)
        outs = critic.chat(messages(bad, style=args.style), n=args.n_samples, **extra)
        hit = [parse_output(o, args.style).verdict == "hallucination" for o in outs]
        orig = critic.chat_one(messages(good, style=args.style), temperature=0.0, **extra)
        return {"id": rec["id"], "detect_rate": sum(hit) / len(hit), "outputs": outs,
                "orig_flagged": parse_output(orig, args.style).verdict == "hallucination", "orig_output": orig}

    map_resumable(list(read_jsonl(args.synth)), run, args.out, concurrency=args.concurrency)
    rows = kept(args.out)
    rates = collections.Counter(round(r["detect_rate"], 2) for r in rows)
    hard = [r for r in rows if r["detect_rate"] < args.threshold]
    print(f"mined {len(rows)}: hard (detect < {args.threshold}) {len(hard)}; detect-rate histogram {sorted(rates.items())}; "
          f"original flagged {sum(r['orig_flagged'] for r in rows)}")


def select(synth: list[dict], mined: dict[str, dict], arm: str, threshold: float, seed: int) -> list[dict]:
    if arm == "direct":
        return synth
    hard = [r for r in synth if r["id"] in mined and mined[r["id"]]["detect_rate"] < threshold]
    if arm == "critic":
        return hard
    return random.Random(seed).sample(synth, len(hard))   # random：与 critic 组同样条数


def mix(args) -> None:
    base = list(read_jsonl(args.base))
    synth = list(read_jsonl(args.synth))
    mined = {r["id"]: r for r in kept(args.mined)} if args.mined else {}
    chosen = select(synth, mined, args.arm, args.threshold, args.seed)
    added = [c for r in chosen for c in to_claims(r)]
    out = base + added
    random.Random(args.seed).shuffle(out)
    write_jsonl(args.out, out)
    meta = {"arm": args.arm, "base": len(base), "synth_pairs": len(chosen), "added_claims": len(added),
            "types": dict(collections.Counter(r["type"] for r in chosen)),
            "tasks": dict(collections.Counter(r["task_type"] for r in chosen)),
            "hallucination_share": sum(c["label"] == "hallucination" for c in out) / len(out)}
    with open(args.out.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--synth", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--model", default="critic")
    m.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    m.add_argument("--style", default="full")
    m.add_argument("--n_samples", type=int, default=4)
    m.add_argument("--temperature", type=float, default=0.7)
    m.add_argument("--threshold", type=float, default=0.5)
    m.add_argument("--concurrency", type=int, default=64)
    x = sub.add_parser("mix")
    x.add_argument("--arm", choices=ARMS, required=True)
    x.add_argument("--base", default="outputs/rt/claims/train_sub35_nat.jsonl")
    x.add_argument("--synth", required=True)
    x.add_argument("--mined", default=None)
    x.add_argument("--threshold", type=float, default=0.5)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", required=True)
    args = ap.parse_args()
    mine(args) if args.cmd == "mine" else mix(args)


if __name__ == "__main__":
    main()
