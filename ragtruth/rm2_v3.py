"""RM2 第三版：直接学 Claude 的「自然」判断。

rm2__v2（真实错误 vs 过滤后的合成错误）在 Claude 自然度标签上 AUC 0.723，不如旧 rm2__core（0.749）：两者定义不同
（「是不是真实模型写的」vs「真实助手会不会这样写」）。这一版标签直接用 Claude 在生成器 v2 采样上标的 natural（gen_v2_label 964 + r2 190），
提示词用同一定义；输入 = 参考文献 + 这句 + 前后句（discrim.render），不给原句。按原文 70 / 10 / 20（salt rm2_v3）。
  python -m ragtruth.rm2_v3
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from common.schema import read_jsonl

D = Path("outputs/rt/core")
RM2V3_SYSTEM = """You see a source (the exact input an assistant received) and one sentence from the assistant's response, with the sentence before and after it for context.

Could a real assistant plausibly have written this sentence in this response: fluent, fitting the context and the style of the rest of the response?
A. natural: yes, a real assistant could have written it like this.
B. unnatural: it reads garbled, out of place or obviously artificial.

Answer with a single letter: A or B."""


def prompt(x: dict) -> list[dict]:
    from ragtruth import discrim
    return [{"role": "system", "content": RM2V3_SYSTEM}, {"role": "user", "content": discrim.render(x["source"], x["context"], int(x["idx"]))}]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rm2__v3")
    ap.add_argument("--extra_train", default="", help="只进训练集的扩增标注目录（逗号分隔）")
    a = ap.parse_args()
    from ragtruth.core import hsplit, write_rm
    parts, c = collections.defaultdict(list), collections.Counter()
    extra = a.extra_train.split(",") if a.extra_train else []
    for sub in ["gen_v2_label", "gen_v2_label_r2"] + extra:
        it = {x["aid"]: x for x in read_jsonl(D / sub / "items.jsonl")}
        for r in read_jsonl(D / sub / "labels.jsonl"):
            x, nat = it.get(r["aid"]), r["claude"].get("natural")
            if not x or nat not in ("yes", "no"):
                continue
            ctx = x["context"] if isinstance(x["context"], list) else json.loads(x["context"])
            x = {**x, "context": ctx}
            lab = "A" if nat == "yes" else "B"
            src = x["id"].split("#")[0]
            sp = "train" if sub in extra else ("train", "calib", "test")[hsplit(src, "rm2_v3", (70, 80))]
            parts[sp].append({"id": "rm2v3/" + r["aid"], "prompt": prompt(x), "completion": lab, "label": lab, "task_type": x["task_type"],
                              "group": "gh-" + src, "origin": f"claude_natural_{x['group']}"})
            c[(sp, lab)] += 1
    meta = write_rm(a.name, parts, 3072, {"counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(c.items())}})
    print("RM2_V3", json.dumps({k: meta[k] for k in ("train", "calib", "test")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
