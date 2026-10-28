"""把正样本和各轮难负样本合成训练集 v{k}，并切出验证集；另出 SFT 冷启动的 2000 条子集。

- 负样本按轮次分层采样，最高轮次（最难）占负样本比例不超过 max_last_round_ratio（默认 40%）。
- 正负 1:1，各类错误等量（按可用量取最小值对齐；某类一条都没有时跳过并告警，不拖垮整个训练集）。
- 去引用增强：uncited_ratio 比例的正样本和「无中生有」负样本（citation_error 且证据为 none）把 cited 置空，
  覆盖推理时「句子没标引用」的情况；挂载错误没有引用就无从谈起，其余类型的判定也不依赖引用标注。
- 验证集按 source_group 切，与训练集文献组不重叠。
- 人工审核结果（outputs/review/<dataset>.jsonl）在这里回流：拒绝的剔除，修改过的用修改后的内容。

用法：python -m data.build_trainset --config configs/data.yaml --version 2
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from common.config import load_config
from common.llm import kept
from common.schema import ERROR_TYPES, Sample, is_unsourced, write_jsonl
from review.store import reviewed


def review_dir(cfg: dict) -> str:
    return cfg.get("review", {}).get("dir", "outputs/review")


def stratified_negatives(neg_by_round: dict[int, list[dict]], n_total: int, max_last_ratio: float,
                         rng: random.Random) -> list[dict]:
    """先按 max_last_ratio 给最高轮次配额，剩余配额均分给其他轮次；某轮不够时把余量让给别的轮次。"""
    rounds = sorted(neg_by_round)
    if len(rounds) == 1:
        pool = list(neg_by_round[rounds[0]])
        rng.shuffle(pool)
        return pool[:n_total]
    last = rounds[-1]
    quota = {last: min(len(neg_by_round[last]), int(n_total * max_last_ratio))}
    rest = n_total - quota[last]
    others = rounds[:-1]
    for i, r in enumerate(others):
        share = rest // (len(others) - i)
        quota[r] = min(len(neg_by_round[r]), share)
        rest -= quota[r]
    out = []
    for r in rounds:
        pool = list(neg_by_round[r])
        rng.shuffle(pool)
        out += pool[:quota[r]]
    return out


def balance_types(negs: list[dict], rng: random.Random) -> list[dict]:
    by_type = defaultdict(list)
    for n in negs:
        by_type[n["type"]].append(n)
    present = [t for t in ERROR_TYPES if by_type[t]]
    missing = [t for t in ERROR_TYPES if not by_type[t]]
    if missing:
        print(f"[warn] no negatives of type {missing}; balancing over {present}")
    k = min((len(by_type[t]) for t in present), default=0)
    out = []
    for t in present:
        rng.shuffle(by_type[t])
        out += by_type[t][:k]
    return out


def drop_citations(samples: list[dict], ratio: float, rng: random.Random) -> list[dict]:
    """正样本和无中生有负样本按 ratio 去掉引用标注（cited=[]），refs 不变。"""
    out = []
    for s in samples:
        if (s["label"] == "faithful" or is_unsourced(s)) and rng.random() < ratio:
            s = {**s, "cited": [], "meta": {**s.get("meta", {}), "uncited_aug": True}}
        out.append(s)
    return out


def split_by_group(samples: list[dict], val_ratio: float, rng: random.Random) -> tuple[list, list]:
    groups = sorted({s["source_group"] for s in samples})
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, int(len(groups) * val_ratio))])
    train = [s for s in samples if s["source_group"] not in val_groups]
    val = [s for s in samples if s["source_group"] in val_groups]
    return train, val


def build_coldstart(cfg: dict, positives: list[dict], rng: random.Random) -> None:
    """SFT 冷启动集（version 0）：正样本 + 第 1 批未经 critic 过滤的合成负样本，正负各半、各类错误等量。

    critic-0 要用它训练出来，再去挖第 1 批难样本，所以这里只能用未过滤的合成样本。
    """
    bcfg = cfg["build_trainset"]
    half = bcfg["coldstart_size"] // 2
    negs = balance_types(reviewed(review_dir(cfg), "synth_batch1",
                                  kept(cfg["synth_negative"]["out"].format(batch=1))), rng)
    rng.shuffle(negs)
    rng.shuffle(positives)
    samples = drop_citations(positives[:half] + negs[:half], bcfg.get("uncited_ratio", 0.0), rng)
    print(f"coldstart: wrote {write_jsonl(bcfg['coldstart_out'], samples)} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--version", type=int, required=True,
                    help="0 = SFT 冷启动集；k >= 1 = 使用前 k 批难样本的训练集 v{k}")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["build_trainset"]
    rng = random.Random(args.version)

    positives = reviewed(review_dir(cfg), "positives", kept(cfg["label_positive"]["out"]))
    if args.version == 0:
        build_coldstart(cfg, positives, rng)
        return
    neg_by_round: dict[int, list[dict]] = defaultdict(list)
    for b in range(1, args.version + 1):
        for n in reviewed(review_dir(cfg), f"hard_batch{b}", kept(cfg["mine_hard"]["out"].format(batch=b))):
            neg_by_round[n["round"]].append(n)

    n_neg = min(sum(map(len, neg_by_round.values())), len(positives), bcfg["max_per_class"])
    negs = balance_types(stratified_negatives(neg_by_round, n_neg, bcfg["max_last_round_ratio"], rng), rng)
    rng.shuffle(positives)
    samples = drop_citations(positives[: len(negs)] + negs, bcfg.get("uncited_ratio", 0.0), rng)
    for s in samples:
        Sample.from_dict(s)  # 校验字段

    train, val = split_by_group(samples, bcfg["val_ratio"], rng)
    for s in train:
        s["split"] = "train"
    for s in val:
        s["split"] = "val"
    out = bcfg["out"].format(version=args.version)
    write_jsonl(out + "/train.jsonl", train)
    write_jsonl(out + "/val.jsonl", val)

    rounds = defaultdict(int)
    for s in negs:
        rounds[s["round"]] += 1
    print(f"v{args.version}: train {len(train)}, val {len(val)}, negatives by round {dict(rounds)}")


if __name__ == "__main__":
    main()
