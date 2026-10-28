"""正样本 = 强模型零样本判为 faithful，且给出的证据逐字可定位、出自被引用文献的 claim。另抽 5% 供人工抽检。

用法：python -m data.label_positive --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import random

from common.config import load_config
from common.llm import LLM, LLMConfig, kept, map_resumable
from common.output_format import evidence_locatable, parse_output
from common.prompts import judge_messages, judge_to_critic_output
from common.schema import Sample, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg = cfg["label_positive"]
    llm = LLM(LLMConfig.from_dict(cfg["judge"]))

    claims = [c for c in read_jsonl(cfg["split_claims"]["out"]) if c["split"] == "train"]

    def judge(c: dict) -> dict | None:
        out = llm.chat_one(judge_messages(c), temperature=0.0)
        p = parse_output(judge_to_critic_output(out))
        if not p.format_ok or p.verdict != "faithful":
            return None
        if not evidence_locatable(p.evidence_ref, p.evidence_text, c["refs"]):
            return None
        if c["cited"] and p.evidence_ref not in c["cited"]:
            return None  # 支持它的原文不在被引用的文献里，不能当正样本
        return Sample(id=c["id"], split="train", source_group=c["source_group"], refs=c["refs"],
                      claim=c["claim"], cited=c["cited"], label="faithful", type="none",
                      evidence=f'{p.evidence_ref}: "{p.evidence_text}"',
                      origin="positive", meta={**c.get("meta", {}), "judge_think": p.think}).to_dict()

    map_resumable(claims, judge, pcfg["out"], concurrency=llm.cfg.concurrency)

    positives = kept(pcfg["out"])
    rng = random.Random(0)
    audit = rng.sample(positives, max(1, int(len(positives) * pcfg["audit_ratio"])))
    write_jsonl(pcfg["audit_out"], audit)
    print(f"{len(positives)}/{len(claims)} claims kept as positive; "
          f"{len(audit)} written to {pcfg['audit_out']}, review in `python -m review.server` "
          f"(error rate > {pcfg['max_audit_error']:.0%} -> tighten prompt and rerun)")


if __name__ == "__main__":
    main()
