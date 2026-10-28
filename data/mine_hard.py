"""难样本挖掘：用当前 critic 判一批合成负样本。

- 漏检（critic 判 faithful）：入库，记 round。
- 检出：丢弃，把 critic 的完整输出和上次的错误 claim 送回生成模型，在同一条原始 claim 上重写，
  重写结果重新校验、重新让 critic 判，最多 max_rewrites 次。

critic 用 vLLM 起服务：vllm serve <ckpt> --served-model-name critic-<round>
每轮抽 audit_size 条入库样本写到抽检文件，在审核台（review/server.py）里标「类型标错 / 不是真错误 / 证据错位」。
漏检样本里混着「根本不是错误」的样本，这一步只能靠人工抽检兜底。

用法：python -m data.mine_hard --config configs/data.yaml --batch 1 --round 0
"""
from __future__ import annotations

import argparse
import random

from common.config import load_config
from common.llm import LLM, LLMConfig, kept, map_resumable
from common.output_format import parse_output
from common.prompts import critic_messages
from common.schema import read_jsonl, write_jsonl
from data.synth_negative import load_pool, synthesize_one


def detect_rate(critic: LLM, sample: dict, n: int) -> tuple[float, list[str]]:
    """critic 采样 n 次，返回判为 hallucination 的比例和全部输出。"""
    outs = critic.chat(critic_messages(sample), n=n)
    hits = [o for o in outs if parse_output(o).verdict == "hallucination"]
    return len(hits) / len(outs), outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--round", type=int, required=True, help="当前 critic 的轮次，0 = SFT 冷启动模型")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg, scfg = cfg["mine_hard"], cfg["synth_negative"]

    critic_cfg = dict(cfg["critic_server"])
    critic_cfg["model"] = critic_cfg["model"].format(round=args.round)
    critic = LLM(LLMConfig.from_dict(critic_cfg))
    generator = LLM(LLMConfig.from_dict(cfg["generator"]))
    pool = load_pool(scfg["example_pool"])
    positives = {p["id"]: p for p in read_jsonl(cfg["label_positive"]["out"]) if not p.get("_dropped")}

    candidates = kept(scfg["out"].format(batch=args.batch))

    def mine(neg: dict) -> dict | None:
        rng = random.Random(neg["id"])
        cur = neg
        for attempt in range(mcfg["max_rewrites"] + 1):
            rate, outs = detect_rate(critic, cur, mcfg["n_samples"])
            if rate < mcfg["keep_if_detect_rate_below"]:
                cur = dict(cur, id=neg["id"], round=args.round + 1)
                # 保留 critic 的全部采样输出，审核台里能看到它为什么漏检
                cur["meta"] = dict(cur["meta"], detect_rate=rate, rewrites=attempt, critic_outputs=outs)
                return cur
            if attempt == mcfg["max_rewrites"]:
                return None
            hit = next(o for o in outs if parse_output(o).verdict == "hallucination")
            feedback = {"critic_output": hit, "prev_claim": cur["claim"]}
            nxt = synthesize_one(generator, positives[neg["parent_id"]], neg["type"], pool, rng,
                                 scfg, feedback=feedback, recipe=neg.get("meta", {}).get("recipe"))
            if nxt is None:
                return None
            cur = nxt
        return None

    out = mcfg["out"].format(batch=args.batch)
    map_resumable(candidates, mine, out, concurrency=critic.cfg.concurrency)

    mined = kept(out)
    audit = random.Random(args.batch).sample(mined, min(mcfg["audit_size"], len(mined)))
    write_jsonl(mcfg["audit_out"].format(batch=args.batch), audit)
    print(f"batch {args.batch}: {len(mined)}/{len(candidates)} kept as hard negatives (round {args.round + 1}); "
          f"audit {len(audit)} -> review in `python -m review.server`; "
          f"stop and fix prompt if not_an_error > {mcfg['max_not_error_ratio']:.0%}")


if __name__ == "__main__":
    main()
