"""判别器版的自然度模型：这句错是模型自己写错的，还是事后改出来的（思路同 GAN 的判别模型）。

rm2__v3 / v4 学的是 Claude 标的「通顺、合乎上下文」。按操作清单核实时发现它管不到一类样本：句子通顺，但这个错没有来由，
助手自己不会犯（核实过的自然真错里占四成）；合成的错 critic 认出九成以上、真实的只认出六成，差的就是这一类。
判别器直接学「真实错句 vs 合成错句」，不用人工标注，生成器每换一版就用它的新样本重训一次。

两类用的是同一批回答、同一批原文：
  A 自然：种子错误对里模型自己写错的句子（core/pairs.jsonl 的 real），放回改对后的回答里
  B 合成：生成器在同一批改对后的回答上造的错句，只取验证模型判为真错的（两类都是错句，学的是错的样子，不是有没有错）
输入只给原文、这一句、前后各一句，不给原句（真实错句没有「原句」）。训练 / 测试按种子集的原文切分。
测试集另放一批 Claude 核实过的合成样本（origin 以 check_ 开头），用来对照 Claude 判的「自然」和「这个错有没有来由」。

  python -m ragtruth.rm_disc --run s1     # → outputs/rt/rm/rm2__d1/{train,calib,test}.jsonl
"""
import argparse
import collections
import gzip
import json
import random
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.core import hsplit, write_rm
from ragtruth.discrim import render

RT = Path("outputs/rt")
SYSTEM = """You see a source (the exact input an assistant received) and one sentence from an assistant's answer written from this source, with the sentence before and after it for context. The sentence is known to be unfaithful to the source: it contradicts the source or says something the source does not support.

Decide where the error comes from:
A. natural: the assistant made this mistake on its own while writing its answer.
B. synthetic: a faithful sentence was edited afterwards to plant the error.

Answer with a single letter: A or B."""


def prompt(source: str, ctx: list[str], idx: int) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(source, ctx, idx)}]


def row(uid: str, source: str, ctx: list[str], idx: int, lab: str, origin: str, group: str) -> dict:
    return {"id": uid, "prompt": prompt(source, ctx, idx), "completion": lab, "label": lab, "task_type": "", "group": group, "origin": origin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s1", help="训练集里合成一类取哪次 GRPO 存下的样本（outputs/rt/gen_ops_grpo/<run>/samples）")
    ap.add_argument("--name", default="rm2__d1")
    ap.add_argument("--neg_per_pos", type=float, default=2.0)
    ap.add_argument("--test_evs", default="", help="测试集里另加的合成一类：留出回答上的采样（core/samples_ops_<名字>.jsonl + _rm4 打分，逗号分隔），取验证模型判为真错的，每个最多 500 条")
    a = ap.parse_args()
    parts, rng = collections.defaultdict(list), random.Random("rm_disc|" + a.name)
    # A：真实错句。种子集训练部分的原文再按哈希留一成做校准
    for p in read_jsonl(RT / "core/pairs.jsonl"):
        sents = list(p["sentences"] if isinstance(p["sentences"], list) else json.loads(p["sentences"]))
        k = int(p["idx"])
        ctx = list(sents)
        ctx[k - 1] = p["real"]
        sp = "test" if p["split"] != "train" else ("train", "calib")[hsplit(str(p["source_id"]), "rm_disc", (90,))]
        parts[sp].append(row("d/real/" + p["id"], p["source"], ctx, k, "A", "real", "src-" + str(p["source_id"])))
    n_pos = len(parts["train"])
    # B（训练 / 校准）：GRPO 训练过程里存下的样本，验证模型判为真错的。Claude 核实抽走的那些不进训练，放到测试集当对照
    PR = {r["id"]: r for r in read_jsonl(RT / "gen_ops_grpo/prompts.jsonl")}
    chk = {r["id"]: r for r in read_jsonl(RT / f"analysis/ops_verify_grpo_{a.run}/labels.jsonl")} if (RT / f"analysis/ops_verify_grpo_{a.run}/labels.jsonl").exists() else {}
    neg = collections.defaultdict(list)
    for f in sorted((RT / f"gen_ops_grpo/{a.run}/samples").glob("step*.jsonl.gz")):
        step = int(f.name[4:].split(".")[0])
        for j, l in enumerate(gzip.open(f, "rt", encoding="utf-8")):
            x = json.loads(l)
            if x.get("why") != "ok":
                continue
            r = PR[x["prompt_id"]]
            ctx = list(r["sentences"])
            ctx[int(x["idx"]) - 1] = x["sentence"]
            sid = r["resp_id"].replace("core-", "")
            win = "start" if step <= 10 else "end"
            cid = f"{win}-{step}-{j}"
            if cid in chk:
                c = chk[cid]["c"]
                if c["severity"] != "none":
                    parts["test"].append(row(f"d/check_{a.run}/{cid}", r["source"], ctx, int(x["idx"]), "B", f"check_{a.run}", "src-" + sid))
            elif x["p_err"] >= 0.5:
                neg[("train", "calib")[hsplit(sid, "rm_disc", (90,))]].append(row(f"d/syn/{a.run}/{step}-{j}", r["source"], ctx, int(x["idx"]), "B", f"syn_{a.run}_{win}", "src-" + sid))
    for sp in ("train", "calib"):
        rng.shuffle(neg[sp])
        parts[sp] += neg[sp][:int(a.neg_per_pos * sum(r["label"] == "A" for r in parts[sp]))]
    # B（测试）：留出原文上的合成错句——提示词版本的 2B 生成器，Claude 判为真错的
    S = {x["id"]: x for x in read_jsonl(RT / "core/samples_ops_pplan_mc.jsonl") if x["why"] == "ok"}
    for r in read_jsonl(RT / "analysis/ops_verify_pplan_mc/labels.jsonl"):
        x = S[r["id"]]
        if r["c"]["severity"] != "none":
            parts["test"].append(row("d/check_pplan/" + r["id"], x["source"], x["context"], int(x["idx"]), "B", "check_pplan", "src-" + str(x["source_id"])))
    for tag in [t for t in a.test_evs.split(",") if t]:
        P = {r["id"]: r for r in read_jsonl(RT / f"core/samples_ops_{tag}_rm4.jsonl")}
        xs = [x for x in read_jsonl(RT / f"core/samples_ops_{tag}.jsonl") if x["why"] == "ok" and P[x["id"]]["p_err"] >= 0.5]
        rng.shuffle(xs)
        for x in xs[:500]:
            parts["test"].append(row(f"d/{tag}/{x['id']}", x["source"], x["context"], int(x["idx"]), "B", tag, "src-" + str(x["source_id"])))
    m = write_rm(a.name, parts, 3072)
    print("RM_DISC", a.name, json.dumps({k: m[k]["n"] for k in ("train", "calib", "test")}), {k: m[k]["label"] for k in ("train", "calib", "test")}, m["test"]["origin"], "train pos", n_pos, "too long", m["too_long_dropped"])


if __name__ == "__main__":
    main()
