"""生成器的样本加进 critic 训练数据，效果到底有没有提升（2026-10-31，用户：这才是最终要回答的问题；先用单字母分类版，训得快）。

基线 = 现有 RM1（人工 + 600 条 G_free 合成）。实验组 = 基线 + N 条新样本，只换样本来源：
  dpo_all / dpo_hard：G_dpo2 的合格样本 / 其中 RM1 < 0.5 的难样本；
  v2_all / v2_hard、v3_all / v3_hard：GRPO 主线 / 换位置分支第 300 步，同样两种抽法；
  human：只留人工样本（对照：现有 600 条合成本身有没有用）。
第二轮（2026-11-01，用户：两个都做。第一轮追加样本没有提升，正例比例也变了）：
  repl_v3：去掉现有合成（498 条错误句 + 414 条配对原句），换成 v3 的错误句和它们自己的原句，每个标签条数都和原来相同，正例比例不变；
  conf_all / conf_hard：追加 v2 + v3 的矛盾类样本，随机 / RM1 分最低的，两组条数相同。
第三轮（2026-11-01，用户：可能有用的地方是开始只有 200 条种子样本的情况）：现有样本测上限，生成器和 RM 都见过全部人工数据，结果偏乐观：
  h200：按回答抽约 200 条人工样本，训练步数和加样本的组相同；h200_2ep：同样的数据，按默认 2 轮训（步数少很多）；
  h200_gfree / h200_dpo / h200_v3：种子 + 合成，每个标签条数都等于现有 G_free（错误句 + 配对原句）。
第四轮（2026-11-01，用户：金标种子训练，跑吧）：金标种子来自 ragtruth.gold（100 条回答，docs/gold_standard.md），在金标评测集上比较：
  gold：只用金标种子；gold_gfree：金标种子 + G_free；rm1_gold：现有 RM1 数据 + 金标种子，同一句话的旧标签由金标覆盖。

  python -m ragtruth.critic_aug sample --model v3 --tag v3 --base_url http://127.0.0.1:8000/v1   # 机器上：采样 + RM 打分
  python -m ragtruth.critic_aug build --tags dpo v2 v3 --n 600                                   # 建 outputs/rt/rm/rm1__<组>/
  python -m ragtruth.critic_aug build2                                                          # 第二轮：建 rm1__repl_v3 / conf_all / conf_hard
  python -m ragtruth.critic_aug build3                                                          # 第三轮：建 rm1__h200*
  python -m ragtruth.critic_aug build4                                                          # 第四轮：建 rm1__gold / gold_gfree / rm1_gold
  python -m ragtruth.critic_aug build5                                                          # 未清洗版：建 rm1__raw
  python -m ragtruth.critic_aug compare                                                         # 各组 vs 基线，配对 bootstrap
  python -m ragtruth.critic_aug compare --base_arm rm1__h200 --arms rm1__h200_gfree rm1__h200_v3  # 换基线，结果写 compare_<基线>.json

抽样口径：每条回答 k 次（只取第一套示例，回答覆盖面优先），过硬规则 + RM3 合格 + 原句过滤；同一回答的同一位置 + 类型去重；
校准集的回答不用（不泄漏到校准）。原句（faithful）人工数据里已经有，所以实验组实际只多了正例：阈值 0.5 的召回会因为先验变化而上涨，
不代表模型变好，判定看 AUC 和「固定精确率下的召回」（提前定好）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.rm_data import RM1_OPT, RM1_SYSTEM, too_long
from ragtruth.rm_eval import auc
from ragtruth.task import render_input

D = Path("outputs/rt/realgen/critic_aug")
RM = Path("outputs/rt/rm")
PROMPTS = "outputs/rt/realgen/grpo/prompts.jsonl"
ARM_KINDS = ("all", "hard")


def cmd_sample(args) -> None:
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_grpo import Server, chat_prompt
    from train.gen_reward import Rewarder, load_temps, parse
    D.mkdir(parents=True, exist_ok=True)
    shard, n_shards = args.shard   # 两张卡各跑一份：回答按顺序轮流分
    # 认领（用户：加开的机器临时帮忙）：谁先 mkdir 成功谁跑这一份，其他机器等它写完；对方 10 分钟没动静（挂了）就接着跑
    claim = D / f"claim_{args.tag}_s{shard}"
    try:
        claim.mkdir()
    except FileExistsError:
        while not (claim / "done").exists():
            if time.time() - claim.stat().st_mtime > 600:
                print("claim stale, taking over", args.tag, shard, flush=True)
                break
            time.sleep(30)
        else:
            print("SAMPLE_DONE by other", args.tag, shard, flush=True)
            return
    out = D / f"samples_{args.tag}.s{shard}.jsonl"
    done = {r["resp_id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = sorted((r for r in read_jsonl(PROMPTS) if r["id"].endswith("#v0")), key=lambda r: r["id"])[: args.n_resps]
    rows = [r for r in rows[shard::n_shards] if r["resp_id"] not in done]
    tok = AutoTokenizer.from_pretrained(args.base)
    srv = Server(args.base_url)
    rw = Rewarder(OpenAI(base_url=args.base_url, api_key="x", timeout=600, max_retries=3), load_temps())
    rm = ThreadPoolExecutor(32)

    def one(r: dict) -> dict:
        gs = srv.sample(args.model, chat_prompt(tok, r["prompt"]), args.k, args.max_tokens, 1.0)
        recs = []
        # 先打一条，把三个 RM 这条回答的前缀写进 vLLM 的前缀缓存；几条同时发时都还没算完，谁也命中不了（用户提醒可以走缓存）
        first = [rw.raw(gs[0]["text"], r)] if gs else []
        for g, s in zip(gs, first + list(rm.map(lambda g: rw.raw(g["text"], r), gs[1:]))):
            rec, why = parse(g["text"], r)
            if rec is None or s.get("why") != "ok" or s.get("p_correct", 0) < 0.5 or s.get("p_orig_halluc", 0.0) > rw.orig_max or "p_halluc" not in s:
                continue
            recs.append({k: rec[k] for k in ("idx", "op", "type", "sentence", "tactic", "original")} | {k: s[k] for k in ("p_correct", "p_halluc", "p_natural")})
        return {"resp_id": r["resp_id"], "n": len(gs), "qualified": recs}

    with ThreadPoolExecutor(args.concurrency) as ex, open(out, "a", encoding="utf-8") as f:
        for k, x in enumerate(ex.map(one, rows)):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            os.utime(claim)   # 心跳
            if k % 200 == 0:
                print(args.tag, k, len(rows), flush=True)
    (claim / "done").touch()
    print("SAMPLE_DONE", args.tag, flush=True)


def claim_row(r: dict, s: dict, tag: str, kind: str, j: int) -> dict:
    i = s["idx"]
    sents = r["sentences"][:i - 1] + [s["sentence"]] + (r["sentences"][i - 1:] if s["op"] == "insert" else r["sentences"][i:])
    prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": r["source"], "sentences": sents, "idx": i})}]
    lab = RM1_OPT[s["type"]]
    return {"id": f"{r['resp_id']}-{tag}{j}-S{i}", "prompt": prompt, "completion": lab, "label": lab, "task_type": r["task_type"],
            "group": r["resp_id"], "origin": f"{tag}_{kind}"}


def orig_row(r: dict, s: dict, tag: str, j: int) -> dict:
    """配对的负例：同一回答同一位置的原句（只有改写才有，插入没有原句），id 和 G_free 的 -orig- 同一写法。"""
    i = s["idx"]
    prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": r["source"], "sentences": r["sentences"], "idx": i})}]
    return {"id": f"{r['resp_id']}-{tag}{j}-orig-S{i}", "prompt": prompt, "completion": "A", "label": "A", "task_type": r["task_type"],
            "group": r["resp_id"], "origin": f"{tag}_orig"}


def pick_replacement(cand: list[tuple], quota: dict[str, int], orig_of) -> list[dict]:
    """cand 已打乱，元素第 0 项是错误句行。先按标签配额取错误句，再取它们的原句；不够时用没选中的样本的原句补，保证每个标签条数都等于配额。
    orig_of 返回原句行，没有可用原句时返回 None。"""
    left, chosen, rest = dict(quota), [], []
    for c in cand:
        lab = c[0]["label"]
        if left.get(lab, 0) > 0:
            left[lab] -= 1
            chosen.append(c)
        else:
            rest.append(c)
    if any(v > 0 for k, v in left.items() if k != "A"):
        raise SystemExit(f"replacement pool too small: short of {left}")
    negs, seen = [], set()
    for c in chosen + rest:
        if len(negs) == quota["A"]:
            break
        o = orig_of(c)
        if o is not None and json.dumps(o["prompt"]) not in seen:
            seen.add(json.dumps(o["prompt"]))
            negs.append(o)
    if len(negs) < quota["A"]:
        raise SystemExit(f"replacement pool too small: {len(negs)} originals, need {quota['A']}")
    return [c[0] for c in chosen] + negs


def cmd_build2(args) -> None:
    base = list(read_jsonl(RM / "rm1" / "train.jsonl"))
    human = [r for r in base if r.get("origin") == "human"]
    syn = [r for r in base if r.get("origin") != "human"]
    calib_groups = {r["group"] for r in read_jsonl(RM / "rm1" / "calib.jsonl")}
    prompts = {r["resp_id"]: r for r in read_jsonl(PROMPTS) if r["id"].endswith("#v0")}
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    human_prompts = {json.dumps(r["prompt"]) for r in human}
    meta = {}

    def write(arm: str, rows: list[dict], info: dict) -> None:
        d = RM / f"rm1__{arm}"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows, key=lambda r: r["id"])
        assert len({r["id"] for r in rows}) == len(rows), arm
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copyfile(RM / "rm1" / sp, d / sp)
        lab = collections.Counter(r["label"] for r in rows)
        (d / "meta.json").write_text(json.dumps({"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}, **info}, indent=1, ensure_ascii=False), encoding="utf-8")
        meta[arm] = {"n": len(rows), "label": dict(lab), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3), **info}

    def pool(tag: str, seen: set) -> list[tuple]:
        out = []
        for x in (x for p in sorted(D.glob(f"samples_{tag}.s*.jsonl")) for x in read_jsonl(p)):
            if x["resp_id"] in calib_groups:
                continue
            for j, s in enumerate(x["qualified"]):
                key = (x["resp_id"], s["op"], s["idx"], s["type"])
                if key not in seen and not too_long(claim_row(prompts[x["resp_id"]], s, tag, "x", j)["prompt"], limit):
                    seen.add(key)
                    out.append((tag, x["resp_id"], j, s))
        return out

    # 替换：标签配额等于现有合成
    quota = dict(collections.Counter(r["label"] for r in syn))
    cand = pool("v3", set())
    random.Random(f"{args.seed}|repl").shuffle(cand)

    def orig_of(c: tuple) -> dict | None:
        _, s, (rid, j) = c
        if s["op"] != "rewrite":
            return None
        o = orig_row(prompts[rid], s, "v3", j)
        return None if json.dumps(o["prompt"]) in human_prompts or too_long(o["prompt"], limit) else o   # 和人工数据完全相同的行只会加权重，不算新负例

    rows = pick_replacement([(claim_row(prompts[rid], s, "v3", "repl", j), s, (rid, j)) for _, rid, j, s in cand], quota, orig_of)
    write("repl_v3", human + rows, {"replaced": len(syn), "quota": quota, "pool": len(cand), "op": dict(collections.Counter(c[3]["op"] for c in cand))})

    # 只加矛盾类：两个模型合并去重，两组条数相同
    seen = set()
    conf = [c for t in ("v2", "v3") for c in pool(t, seen) if c[3]["type"] == "conflict"]
    n = min(args.n, len(conf))
    print("conflict pool", len(conf), "RM1 < 0.5:", sum(c[3]["p_halluc"] < 0.5 for c in conf), "use n =", n, flush=True)
    rand = list(conf)
    random.Random(f"{args.seed}|conf|all").shuffle(rand)
    # RM1 < 0.5 的矛盾类只有约 100 条，太少看不出差别，所以按 RM1 分从低到高取最难的 n 条
    for kind, cs in (("all", rand), ("hard", sorted(conf, key=lambda c: (c[3]["p_halluc"], c[0], c[1], c[2])))):
        added = [claim_row(prompts[rid], s, t, f"conf{kind}", j) for t, rid, j, s in cs[:n]]
        write(f"conf_{kind}", base + added, {"added": n, "pool": len(cs), "by_model": dict(collections.Counter(c[0] for c in cs[:n])),
                                             "mean_p_halluc": round(sum(c[3]["p_halluc"] for c in cs[:n]) / max(1, n), 3),
                                             "mean_p_natural": round(sum(c[3]["p_natural"] for c in cs[:n]) / max(1, n), 3)})
    (D / "build2.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def pick_seed(rows: list[dict], n: int, seed: int) -> list[dict]:
    """按回答抽种子：同一回答的行一起进（和训练、测试的划分口径一致），刚够 n 条就停。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    groups = sorted(by)
    random.Random(f"{seed}|h200").shuffle(groups)
    out = []
    for g in groups:
        if len(out) >= n:
            break
        out += by[g]
    return out


def cmd_build3(args) -> None:
    base = list(read_jsonl(RM / "rm1" / "train.jsonl"))
    human = [r for r in base if r.get("origin") == "human"]
    gfree = [r for r in base if r.get("origin") != "human"]
    calib_groups = {r["group"] for r in read_jsonl(RM / "rm1" / "calib.jsonl")}
    prompts = {r["resp_id"]: r for r in read_jsonl(PROMPTS) if r["id"].endswith("#v0")}
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    seed_rows = pick_seed(human, args.seed_n, args.seed)
    seed_prompts = {json.dumps(r["prompt"]) for r in seed_rows}
    quota = dict(collections.Counter(r["label"] for r in gfree))
    meta = {}

    def write(arm: str, rows: list[dict], info: dict) -> None:
        d = RM / f"rm1__{arm}"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows, key=lambda r: r["id"])
        assert len({r["id"] for r in rows}) == len(rows), arm
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copyfile(RM / "rm1" / sp, d / sp)
        lab = collections.Counter(r["label"] for r in rows)
        (d / "meta.json").write_text(json.dumps({"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}, **info}, indent=1, ensure_ascii=False), encoding="utf-8")
        meta[arm] = {"n": len(rows), "label": dict(lab), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3), **info}

    write("h200_gfree", seed_rows + gfree, {"added": len(gfree)})
    for tag in ("dpo", "v3"):
        seen, cand = set(), []
        for x in (x for p in sorted(D.glob(f"samples_{tag}.s*.jsonl")) for x in read_jsonl(p)):
            if x["resp_id"] in calib_groups:
                continue
            for j, c in enumerate(x["qualified"]):
                key = (x["resp_id"], c["op"], c["idx"], c["type"])
                if key not in seen and not too_long(claim_row(prompts[x["resp_id"]], c, tag, "x", j)["prompt"], limit):
                    seen.add(key)
                    cand.append((x["resp_id"], j, c))
        random.Random(f"{args.seed}|h200|{tag}").shuffle(cand)

        def orig_of(c: tuple) -> dict | None:
            _, sm, (rid, j) = c
            if sm["op"] != "rewrite":
                return None
            o = orig_row(prompts[rid], sm, tag, j)
            return None if json.dumps(o["prompt"]) in seed_prompts or too_long(o["prompt"], limit) else o

        rows = pick_replacement([(claim_row(prompts[rid], c, tag, "h200", j), c, (rid, j)) for rid, j, c in cand], quota, orig_of)
        write(f"h200_{tag}", seed_rows + rows, {"added": len(rows), "pool": len(cand)})
    n_added = len(seed_rows) + sum(quota.values())
    write("h200", seed_rows, {"added": 0, "epochs": round(2 * n_added / len(seed_rows), 4)})   # 和加样本的组步数相同（否则比的不只是数据）
    write("h200_2ep", seed_rows, {"added": 0})
    (D / "build3.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def gold_rows(picks: dict[str, dict], final: list[dict], limit: int, origin: str = "gold") -> tuple[list[dict], int]:
    """金标逐句结论 → RM1 训练行（A 没错 / B 矛盾 / C 无依据）。太长的句子不进训练，和 rm_data 同一口径。"""
    rows, too = [], 0
    for x in final:
        r = picks[x["resp_id"]]
        prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": r["source"], "sentences": r["sentences"], "idx": x["idx"]})}]
        if too_long(prompt, limit):
            too += 1
            continue
        lab = RM1_OPT[x["type"]] if x["is_error"] else "A"
        rows.append({"id": f"{x['resp_id']}-{origin}-S{x['idx']}", "prompt": prompt, "completion": lab, "label": lab, "task_type": r["task_type"], "group": x["resp_id"], "origin": origin})
    return rows, too


def raw_rows(claims: list[dict], calib_groups: set[str], limit: int) -> tuple[list[dict], int]:
    """未清洗的人工数据 → RM1 训练行，留出口径和 RM1 相同（校准集的回答不进训练）。"""
    rows, too = [], 0
    for c in claims:
        if c["response_id"] in calib_groups:
            continue
        prompt = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({**c, "idx": int(c["idx"])})}]
        if too_long(prompt, limit):
            too += 1
            continue
        lab = RM1_OPT[("baseless" if c["type"] not in ("conflict", "baseless") else c["type"]) if c["label"] != "faithful" else "faithful"]
        rows.append({"id": c["id"], "prompt": prompt, "completion": lab, "label": lab, "task_type": c["task_type"], "group": c["response_id"], "origin": "human"})
    return rows, too


def cmd_build5(args) -> None:
    """用户（2026-11-01）：GRPO 的低版本 critic 用没清洗过的数据训练。旧的 sft-9b-v2-notips 没留权重和打分，又是生成式，这里用同一份数据（train_sub35_r3）按 RM1 的分类形式重训。"""
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    calib = {r["group"] for r in read_jsonl(RM / "rm1" / "calib.jsonl")}
    rows, too = raw_rows(sorted(read_jsonl("outputs/rt/claims/train_sub35_r3.jsonl"), key=lambda r: r["id"]), calib, limit)
    d = RM / "rm1__raw"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("calib.jsonl", "test.jsonl"):
        shutil.copyfile(RM / "rm1" / sp, d / sp)
    lab = collections.Counter(r["label"] for r in rows)
    (d / "meta.json").write_text(json.dumps({"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}, "too_long": too}, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"raw": {"n": len(rows), "label": dict(lab), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3), "too_long": too}}
    (D / "build5.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def cmd_build4(args) -> None:
    base = list(read_jsonl(RM / "rm1" / "train.jsonl"))
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    G = Path("outputs/rt/gold")
    picks = {r["id"]: r for r in read_jsonl(G / "seed_ids.jsonl")}
    gold, too = gold_rows(picks, list(read_jsonl(G / "seed_final.jsonl")), limit)
    keys = {json.dumps(r["prompt"]) for r in gold}
    kept_base = [r for r in base if json.dumps(r["prompt"]) not in keys]   # 同一句话，金标说了算
    gfree = [r for r in base if r.get("origin") != "human"]
    meta = {}
    for arm, rows, info in (("gold", gold, {}), ("gold_gfree", gold + gfree, {"added": len(gfree)}), ("rm1_gold", kept_base + gold, {"replaced_old_rows": len(base) - len(kept_base)})):
        d = RM / f"rm1__{arm}"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows, key=lambda r: r["id"])
        assert len({r["id"] for r in rows}) == len(rows), arm
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copyfile(RM / "rm1" / sp, d / sp)
        lab = collections.Counter(r["label"] for r in rows)
        (d / "meta.json").write_text(json.dumps({"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}, **info}, indent=1, ensure_ascii=False), encoding="utf-8")
        meta[arm] = {"n": len(rows), "label": dict(lab), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3), "gold_too_long": too, **info}
    (D / "build4.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def cmd_build6(args) -> None:
    """用户（2026-11-01）：rm1-9b 在 9B 回答上漏报多，先用 9B 自己的回答（训练池原文、Claude 标注）重训 critic，再决定 GRPO。"""
    base = list(read_jsonl(RM / "rm1" / "train.jsonl"))
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    G, P = Path("outputs/rt/gold"), Path("outputs/rt/base9b/pool")
    picks = {r["id"]: r for r in read_jsonl(P / "answers.jsonl") if not r.get("_dropped")}
    b9, too = gold_rows(picks, list(read_jsonl(P / "final.jsonl")), limit, origin="b9")
    gold, _ = gold_rows({r["id"]: r for r in read_jsonl(G / "seed_ids.jsonl")}, list(read_jsonl(G / "seed_final.jsonl")), limit)
    keys = {json.dumps(r["prompt"]) for r in gold}
    meta = {}
    for arm, rows in (("rm1_9b", base + b9), ("rm1_gold_9b", [r for r in base if json.dumps(r["prompt"]) not in keys] + gold + b9)):
        d = RM / f"rm1__{arm}"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows, key=lambda r: r["id"])
        assert len({r["id"] for r in rows}) == len(rows), arm
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copyfile(RM / "rm1" / sp, d / sp)
        lab = collections.Counter(r["label"] for r in rows)
        (d / "meta.json").write_text(json.dumps({"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}}, indent=1, ensure_ascii=False), encoding="utf-8")
        meta[arm] = {"n": len(rows), "b9_rows": len(b9), "b9_too_long": too, "label": dict(lab), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3)}
    (D / "build6.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def cmd_build(args) -> None:
    base = list(read_jsonl(RM / "rm1" / "train.jsonl"))
    calib_groups = {r["group"] for r in read_jsonl(RM / "rm1" / "calib.jsonl")}
    prompts = {r["resp_id"]: r for r in read_jsonl(PROMPTS) if r["id"].endswith("#v0")}
    limit = json.load(open(RM / "rm1" / "meta.json", encoding="utf-8"))["max_tokens"]
    meta = {}

    def write(arm: str, rows: list[dict], info: dict) -> None:
        d = RM / f"rm1__{arm}"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows, key=lambda r: r["id"])
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copyfile(RM / "rm1" / sp, d / sp)
        lab = collections.Counter(r["label"] for r in rows)
        m = {"max_tokens": limit, "train": {"n": len(rows), "label": dict(lab)}, **info}
        (d / "meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
        meta[arm] = {"n": len(rows), "pos_rate": round((lab["B"] + lab["C"]) / len(rows), 3), **info}

    write("human", [r for r in base if r.get("origin") == "human"], {"added": 0})
    pools, stats = {}, {}
    for tag in args.tags:
        seen, pool = set(), []
        n_all = n_q = 0
        for x in (x for p in sorted(D.glob(f"samples_{tag}.s*.jsonl")) for x in read_jsonl(p)):
            n_all += x["n"]
            n_q += len(x["qualified"])
            if x["resp_id"] in calib_groups:
                continue
            for j, s in enumerate(x["qualified"]):
                key = (x["resp_id"], s["op"], s["idx"], s["type"])
                if key not in seen:
                    seen.add(key)
                    row = claim_row(prompts[x["resp_id"]], s, tag, "x", j)
                    if not too_long(row["prompt"], limit):   # 和 rm_data 建 RM1 训练集同一口径
                        pool.append((x["resp_id"], j, s))
        stats[tag] = (n_all, n_q)
        for kind in ARM_KINDS:
            pools[tag, kind] = [p for p in pool if kind == "all" or p[2]["p_halluc"] < 0.5]
    n = min(args.n, *(len(v) for v in pools.values()))   # 每组条数必须相同，否则比的不只是样本来源；不够 N 时全部降到最少那组
    print("arm pools", {f"{t}_{k}": len(v) for (t, k), v in pools.items()}, "use n =", n, flush=True)
    for tag in args.tags:
        n_all, n_q = stats[tag]
        for kind in ARM_KINDS:
            cand = pools[tag, kind]
            random.Random(f"{args.seed}|{tag}|{kind}").shuffle(cand)
            rows = [(claim_row(prompts[rid], s, tag, kind, j), s) for rid, j, s in cand[:n]]
            skipped = 0
            added = [r for r, _ in rows]
            ty = collections.Counter(s["type"] for _, s in rows)
            write(f"{tag}_{kind}", base + added, {
                "added": len(added), "too_long_skipped": skipped, "pool": len(cand), "samples": n_all, "qualified": n_q,
                "conflict_share": round(ty["conflict"] / max(1, len(rows)), 3), "mean_p_halluc": round(sum(s["p_halluc"] for _, s in rows) / max(1, len(rows)), 3),
                "mean_p_natural": round(sum(s["p_natural"] for _, s in rows) / max(1, len(rows)), 3)})
    (D / "build.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def _scores(arm: str) -> dict[str, tuple[int, float, str]]:
    from ragtruth.rm_eval import probs
    d = RM / arm / "eval"
    T = json.load(open(d / "report.json", encoding="utf-8"))["temperature"]
    out = {}
    for r in read_jsonl(d / "test_scores.jsonl"):
        if r.get("lp"):
            p = probs(r["lp"], T)
            out[r["id"]] = (int(r["label"] in "BC"), p["B"] + p["C"], r["id"].rsplit("-S", 1)[0], r["label"])
    return out


def recall_at_precision(ys: list[int], ps: list[float], prec: float) -> float:
    """精确率不低于 prec 时能达到的最高召回。"""
    order = sorted(zip(ps, ys), reverse=True)
    pos, tp, best = sum(ys), 0, 0.0
    for k, (_, y) in enumerate(order, 1):
        tp += y
        if tp / k >= prec:
            best = tp / pos
    return best


def _point(ys: list[int], ps: list[float], prec: float) -> dict:
    tp = sum(y and p >= 0.5 for y, p in zip(ys, ps))
    fp = sum((not y) and p >= 0.5 for y, p in zip(ys, ps))
    r, q = tp / max(1, sum(ys)), tp / max(1, tp + fp)
    return {"auc": auc(ys, ps), "recall@prec": recall_at_precision(ys, ps, prec), "recall": r, "precision": q, "f1": 2 * r * q / max(1e-9, r + q)}


def cmd_compare(args) -> None:
    from ragtruth.pp_valid import load as pp_labels
    lab = pp_labels()
    base = _scores(args.base_arm)
    arms = args.arms or sorted(p.name for p in RM.glob("rm1__*") if (p / "eval" / "report.json").exists() and not p.name.startswith(("rm1__h200", "rm1__gold", "rm1__rm1_gold", "rm1__raw")))   # 第三、四轮有自己的基线
    rep = {}
    valid = lambda i, y: not (y and lab.get(i, "real") != "real")
    subsets = (("valid", valid), ("strict", lambda i, y: True),   # 难样本组几乎只有无依据类，所以按真实类型分开看（负例都留着）
               ("valid_conflict", lambda i, y: valid(i, y) and base[i][3] != "C"), ("valid_baseless", lambda i, y: valid(i, y) and base[i][3] != "B"))
    for sub, keep in subsets:
        ids = sorted(i for i, (y, _, _, _) in base.items() if keep(i, y))
        by = collections.defaultdict(list)
        for i in ids:
            by[base[i][2]].append(i)
        groups = sorted(by)
        ys, bp = [base[i][0] for i in ids], [base[i][1] for i in ids]
        tp = sum(y and p >= 0.5 for y, p in zip(ys, bp))
        prec = tp / max(1, sum(p >= 0.5 for p in bp))   # 固定在基线阈值 0.5 的精确率
        rep[sub] = {"n": len(ids), "pos": sum(ys), "baseline": {k: round(v, 4) for k, v in _point(ys, bp, prec).items()}, "fixed_precision": round(prec, 4)}
        rng = random.Random(args.seed)
        draws = [[rng.choice(groups) for _ in groups] for _ in range(args.boot)]
        for arm in arms:
            s = _scores(arm)
            ap = [s[i][1] for i in ids]
            pt = _point(ys, ap, prec)
            diffs = collections.defaultdict(list)
            for g in draws:
                idx = [i for x in g for i in by[x]]
                yy, bb, aa = [base[i][0] for i in idx], [base[i][1] for i in idx], [s[i][1] for i in idx]
                a, b = _point(yy, aa, prec), _point(yy, bb, prec)
                for k in a:
                    diffs[k].append(a[k] - b[k])
            res = {}
            for k in pt:
                v = sorted(diffs[k])
                res[k] = {"value": round(pt[k], 4), "diff": round(pt[k] - _point(ys, bp, prec)[k], 4), "ci95": [round(v[int(.025 * len(v))], 4), round(v[int(.975 * len(v)) - 1], 4)]}
            rep[sub][arm] = res
    out = D / ("compare.json" if args.base_arm == "rm1" else f"compare_{args.base_arm}.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for sub in rep:
        print("==", sub, "n", rep[sub]["n"], "pos", rep[sub]["pos"], "fixed precision", rep[sub]["fixed_precision"], "baseline", rep[sub]["baseline"])
        for arm in arms:
            r = rep[sub][arm]
            print(f"{arm:18s}", {k: (r[k]["value"], r[k]["diff"], r[k]["ci95"]) for k in ("auc", "recall@prec", "f1", "recall", "precision")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "build", "build2", "build3", "build4", "build5", "build6", "compare"])
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tags", nargs="+", default=["dpo", "v2", "v3"])
    ap.add_argument("--n_resps", type=int, default=3000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], help="第几份 共几份")
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed_n", type=int, default=200)
    ap.add_argument("--base_arm", default="rm1", help="compare 的基线")
    ap.add_argument("--arms", nargs="*", default=None, help="compare 只比这些组；不给就比第一、二轮全部")
    args = ap.parse_args()
    {"sample": cmd_sample, "build": cmd_build, "build2": cmd_build2, "build3": cmd_build3, "build4": cmd_build4, "build5": cmd_build5, "build6": cmd_build6, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
