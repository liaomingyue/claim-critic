"""定向造错生成器 GRPO 的提示池（2026-11-03，用户：合成 2B critic 漏掉的几大类错误，用 GRPO 训出和真实模型一致的错误）。

种子是 2B GRPO 第 300 步策略自己写的回答（在策略：造出来的错要像这个模型会犯的），原文是回答 GRPO 的训练原文
（outputs/rt/ansgrpo/train_all.jsonl，和 critic 按类型的测试集 outputs/rt/minicpm/grpo300 的评测原文不重叠）。
prompt 用自由版 + 定向改法（synth.FOCUS_TACTICS：张冠李戴、跨段拼接、数字、确定性 / 范围、顺序、关系反转、泛化、因果连接），
正好覆盖 critic 按类型统计里 2B critic 的盲区。每句原句预先用 9B 金标 critic 打 P(有错)（训练外调用用 9B，用户），
存在行里的 orig_p：改的原句本来就有错，奖励给 0。

  python -m ragtruth.gen_target_data answers --url http://127.0.0.1:8000/v1 --model g300     # 每个原文采 k 条回答
  python -m ragtruth.critic_score --answers outputs/rt/gentarget/answers.jsonl ... --out outputs/rt/gentarget/critic9b_scores.json
  python -m ragtruth.gen_target_data build                                                     # → prompts.jsonl / eval_prompts.jsonl / meta.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.schema import read_jsonl
from ragtruth.ans_data import SYSTEM_QA
from ragtruth.base_eval import to_answer
from ragtruth.synth import build_prompt_free, example_pool, free_examples

D = Path("outputs/rt/gentarget")
SRC = "outputs/rt/ansgrpo/train_all.jsonl"
REFUSAL = "unable to answer"


def cmd_answers(a) -> None:
    from openai import OpenAI
    cl = OpenAI(base_url=a.url, api_key="x", timeout=1800)
    srcs = sorted(read_jsonl(SRC), key=lambda r: r["id"])
    jobs = [(r, k) for r in srcs for k in range(a.k)]

    def run(job):
        r, k = job
        msgs = ([{"role": "system", "content": SYSTEM_QA}] if r["task_type"] == "QA" else []) + [{"role": "user", "content": r["source"]}]
        out = cl.chat.completions.create(model=a.model, messages=msgs, max_tokens=1024, temperature=1.0, top_p=1.0, seed=k,
                                         extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": False}})
        c = out.choices[0]
        return {**to_answer(r, k, (c.message.content or "").strip(), c.finish_reason), "id": f"gt-{r['source_id']}-{k}", "model": "minicpm5-2b-grpo300"}

    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(run, jobs))
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "answers.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("ANSWERS", len(rows), collections.Counter(x["finish"] for x in rows), flush=True)


def cmd_hardpool(a) -> None:
    """骗过 critic 的示例池（2026-11-03，用户：目前骗不过 critic，要想办法）：两轮 Claude 确认的 2B 真错，用 2B critic（rm1）在原回答里打分，
    P(判出) < 0.5 的就是 critic 真实漏掉的 2B 错误，存成 synth.example_pool 的格式，给 build --example_pool 当 prompt 里的示例
    （原来的示例是 RAGTruth 里别的模型的错句）。"""
    from ragtruth.rm_data import RM1_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.task import render_input
    from train.gen_reward import NO_THINK
    cls = _clients(a.url)
    ans = {r["id"]: r for r in read_jsonl(D / "answers.jsonl")}
    items = [x for f in ("labels.jsonl", "labels2.jsonl") for x in read_jsonl(D / "seed2b" / f) if x["label"]["is_error"] and x["label"]["type"] in ("conflict", "baseless")]

    def run(x: dict) -> dict:
        r, l = ans[x["resp_id"]], x["label"]
        resp = _pick(cls, x["id"]).chat.completions.create(model=a.model, messages=[{"role": "system", "content": RM1_SYSTEM},
                                                           {"role": "user", "content": render_input({"source": r["source"], "sentences": r["sentences"], "idx": x["idx"]})}],
                                                           max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(resp.choices[0], "ABC")
        p = probs(lp, a.rm1_temp) if lp else None
        return {"id": x["id"], "source_id": str(r["source_id"]), "source": r["source"], "task_type": r["task_type"], "type": l["type"], "mechanism": l["mechanism"],
                "sentences": r["sentences"], "idx": int(x["idx"]), "evidence": l.get("evidence"), "reason": l.get("reason") or "",
                "p_halluc": (p["B"] + p["C"]) if p else None}

    with ThreadPoolExecutor(32) as ex:
        out = [x for x in ex.map(run, items) if x["p_halluc"] is not None]
    hard = [x for x in out if x["p_halluc"] < 0.5 and x["reason"]]
    with open(D / "hard_pool.jsonl", "w", encoding="utf-8") as f:
        for x in hard:
            f.write(json.dumps(x, ensure_ascii=False) + chr(10))
    print("HARDPOOL", json.dumps({"scored": len(out), "missed_by_critic": len(hard), "by": dict(collections.Counter(f"{x['task_type']}|{x['type']}" for x in hard)),
                                  "mechanism": dict(collections.Counter(x["mechanism"] for x in hard).most_common())}, ensure_ascii=False), flush=True)


def cmd_build(a) -> None:
    sc = {x["id"]: x["p"] for x in json.load(open(D / "critic9b_scores.json", encoding="utf-8"))}
    if a.example_pool:   # 示例换成 critic 漏掉的 2B 真错（hardpool）；同一原文的不用（free_examples 按原文排除）
        pool = collections.defaultdict(list)
        for x in read_jsonl(D / a.example_pool):
            pool[(x["task_type"], x["type"])].append(x)
    else:
        pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
    resps, why = [], collections.Counter()
    for r in read_jsonl(D / "answers.jsonl"):
        n = len(r["sentences"])
        if r["finish"] != "stop" or REFUSAL in r["response"].lower():
            why["unfinished_or_refusal"] += 1
            continue
        if n < 2:
            why["one_sentence"] += 1
            continue
        p = [sc.get(f"{r['id']}-S{i}") for i in range(1, n + 1)]
        if any(x is None for x in p):
            why["unscored"] += 1
            continue
        if sum(x <= 0.5 for x in p) < 1:
            why["no_clean_sentence"] += 1
            continue
        resps.append({**r, "orig_p": [round(x, 4) for x in p]})
    srcs = sorted({r["source_id"] for r in resps})
    random.Random(a.seed).shuffle(srcs)
    ev_src = set(srcs[:a.eval_sources])
    rows, ev = [], []
    for r in sorted(resps, key=lambda r: r["id"]):
        for v in range(1 if r["source_id"] in ev_src else a.n_variants):
            ex = free_examples(pool, r["task_type"], random.Random(f"{r['id']}#{v}"), exclude_source=r["source"])
            # --type_split：一半 prompt 指定 baseless、一半指定 conflict（训练集按变体号，快评按回答 id 的哈希）
            want = (("baseless", "conflict")[(v if r["source_id"] not in ev_src else zlib.crc32(r["id"].encode())) % 2] if a.type_split else None)
            p = build_prompt_free(r, ex, focus=True, free_size=bool(a.free_size), ctx=bool(a.ctx), want_type=want)
            if len(p) > a.max_chars:
                why["too_long"] += 1
                break
            row = {"id": f"{r['id']}#v{v}", "resp_id": r["id"], "source_id": r["source_id"], "task_type": r["task_type"], "model": r["model"],
                   "source": r["source"], "sentences": r["sentences"], "orig_p": r["orig_p"], "prompt": p, **({"want_type": want} if want else {})}
            (ev if r["source_id"] in ev_src else rows).append(row)
    random.Random(a.seed).shuffle(rows)
    for name, xs in (("prompts", rows), ("eval_prompts", ev)):
        with open(D / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    meta = {"answers_kept": len(resps), "dropped": dict(why), "prompts": len(rows), "eval_prompts": len(ev),
            "train_sources": len(srcs) - len(ev_src), "eval_sources": len(ev_src),
            "by_task": dict(collections.Counter(x["task_type"] for x in rows)),
            "orig_flagged_sentence_share": round(sum(p > 0.5 for r in resps for p in r["orig_p"]) / max(1, sum(len(r["orig_p"]) for r in resps)), 4),
            "mean_prompt_chars": round(sum(len(x["prompt"]) for x in rows) / max(1, len(rows)))}
    (D / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("BUILD", json.dumps(meta, ensure_ascii=False), flush=True)


def _clients(url: str) -> list:
    """--url 可用逗号分隔多个服务（双卡各起一个 vLLM，同样的模型）；按样本 id 的哈希分配，同一条样本的生成和打分走同一个服务。"""
    from openai import OpenAI
    return [OpenAI(base_url=u, api_key="x", timeout=1800) for u in url.split(",")]


def _pick(cls: list, key: str):
    return cls[zlib.crc32(key.encode()) % len(cls)]


def cmd_synth(a) -> None:
    """RM2 的「合成」一侧（用户，2026-11-03：RM2 完全重造；SFT 完的 2B 造错生成器采出来的就是合成，回答 GRPO 策略自己写出的错就是自然）：
    gen-seed-mc 在训练 prompt 上各采 1 条，过硬规则，2B RM3 P(合格) ≥ gate、原句 orig_p ≤ 0.5 的留下。"""
    from transformers import AutoTokenizer
    from ragtruth import audit_synth
    from ragtruth.rm_data import RM3_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from train.gen_grpo import chat_prompt
    from train.gen_reward import NO_THINK, context, parse
    cls = _clients(a.url)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = list(read_jsonl(D / "prompts.jsonl"))

    def run(job: tuple) -> dict:
        r, k = job
        cl = _pick(cls, f"{r['id']}#{k}")
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=768, temperature=1.0, top_p=1.0,
                                    seed=k, extra_body={"top_k": -1})
        rec, why = parse(out.choices[0].text, r)
        x = {"id": r["id"] + (f"#s{k}" if k else ""), "resp_id": r["resp_id"], "source_id": r["source_id"], "task_type": r["task_type"], "why": why}
        if rec is None:
            return x
        resp = cl.chat.completions.create(model=a.rm3, messages=[{"role": "system", "content": RM3_SYSTEM}, {"role": "user", "content": audit_synth.render(rec)}],
                                          max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(resp.choices[0], "ABCD")
        pc = probs(lp, a.rm3_temp)["A"] if lp else 0.0
        orig = r["orig_p"][rec["idx"] - 1] if rec["op"] == "rewrite" else 0.0
        return {**x, "type": rec["type"], "op": rec["op"], "idx": rec["idx"], "tactic_raw": rec["tactic_raw"], "sentence": rec["sentence"],
                "original": rec["original"], "p_correct": pc, "orig_p": orig, "context": context(rec), "source": r["source"],
                "keep": pc >= a.gate and orig <= 0.5}

    with ThreadPoolExecutor(64) as ex:   # --samples：每个 prompt 采几条（RM2 合成一侧凑数用）
        out = list(ex.map(run, [(r, k) for r in rows for k in range(a.samples)]))
    with open(D / a.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    kept = [x for x in out if x.get("keep")]
    print("SYNTH", json.dumps({"samples": len(out), "why": dict(collections.Counter(x["why"] for x in out)), "kept": len(kept),
                               "kept_types": dict(collections.Counter(x["type"] for x in kept))}, ensure_ascii=False), flush=True)


def cmd_harvest(a) -> None:
    """批量造 critic 训练数据（2026-11-03，用户：数据生成和 critic 重训的脚本先准备）：选定的生成器检查点在训练 prompt 上每个采 --samples 条，
    和训练时一样打分：RM3 P(真错) = A + B、原句 orig_p（9B 预先算好）、2B critic P(判出)、RM2 P(自然)。全部存下（之后按边界区间筛），
    输出 outputs/rt/gentarget/<--out>。双卡：--url 逗号分隔，每条样本的生成和打分走同一个服务。"""
    from transformers import AutoTokenizer
    from ragtruth import audit_synth, discrim
    from ragtruth.rm_data import RM1_SYSTEM, RM2G_SYSTEM, RM3_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from ragtruth.task import render_input
    from train.gen_grpo import chat_prompt
    from train.gen_reward import NO_THINK, context, parse
    from ragtruth.sev_rm import SEV_SYSTEM
    cls = _clients(a.url)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = list(read_jsonl(D / a.inp))
    t1, t2, t3 = a.rm1_temp, a.rm2_temp, a.rm3_temp

    def opt(cl, model, system, user, opts, t):
        r = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(r.choices[0], opts)
        return probs(lp, t) if lp else None

    def run(job: tuple) -> list[dict]:
        r, k0 = job
        cl = _pick(cls, f"{r['id']}#{k0}")
        out = cl.completions.create(model=a.model, prompt=tok(chat_prompt(tok, r["prompt"]))["input_ids"], max_tokens=768, temperature=1.0, top_p=1.0,
                                    n=a.samples, seed=k0, extra_body={"top_k": -1})
        res = []
        for j, c in enumerate(out.choices):
            rec, why = parse(c.text, r)
            x = {"id": f"{r['id']}#h{j}", "prompt_id": r["id"], "resp_id": r["resp_id"], "source_id": r["source_id"], "task_type": r["task_type"], "why": why}
            if rec is None or rec["op"] != "rewrite":
                res.append(x)
                continue
            if a.sev:   # 新循环（2026-11-05）：门槛用严重程度 RM，输入同 sev_rm.prompt（回答里已换成改写句）；原句有错用 sev 的 P(F)，不用 9B 的 orig_p
                p3 = opt(cl, a.sev, SEV_SYSTEM, audit_synth.render({**rec, "sentences": context(rec)}), "ABCDEF", a.sev_temp)
            else:
                p3 = opt(cl, a.rm3, RM3_SYSTEM, audit_synth.render(rec), "ABCD", t3)
            pc = (p3["A"] + p3["B"]) if p3 else 0.0
            x.update({"type": rec["type"], "idx": rec["idx"], "sentence": rec["sentence"], "original": rec["original"], "evidence": rec["evidence"],
                      "reason": rec["reason"], "tactic_raw": rec["tactic_raw"], "p_correct": pc, "p_a": p3["A"] if p3 else 0.0,
                      "orig_p": (p3["F"] if p3 else 1.0) if a.sev else r["orig_p"][rec["idx"] - 1], "p_sev": p3 if a.sev else None, "source": r["source"], "context": context(rec), "sentences_orig": r["sentences"]})
            if pc >= a.gate and x["orig_p"] <= 0.5:
                p1 = opt(cl, a.rm1, RM1_SYSTEM, render_input({"source": rec["source"], "sentences": context(rec), "idx": rec["idx"]}), "ABC", t1)
                p2 = opt(cl, a.rm2, RM2G_SYSTEM, discrim.render(rec["source"], context(rec), rec["idx"]), "AB", t2)
                x.update({"p_halluc": (p1["B"] + p1["C"]) if p1 else None, "p_natural": p2["A"] if p2 else None})
            res.append(x)
        return res

    jobs = [(r, k) for r in rows for k in range(a.rounds)]
    with ThreadPoolExecutor(64) as ex:
        out = [x for xs in ex.map(run, jobs) for x in xs]
    with open(D / a.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + chr(10))
    q = [x for x in out if x.get("p_halluc") is not None]
    bins = ((0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01))
    print("HARVEST", json.dumps({"samples": len(out), "qualified": len(q), "why": dict(collections.Counter(x["why"] for x in out).most_common(6)),
                                 "bins": {f"{lo}-{hi}": sum(lo <= x["p_halluc"] < hi for x in q) for lo, hi in bins},
                                 "natural_in_0.1-0.7": round(sum(x["p_natural"] > 0.5 for x in q if 0.1 <= x["p_halluc"] < 0.7) / max(1, sum(0.1 <= x["p_halluc"] < 0.7 for x in q)), 3)},
                                ensure_ascii=False), flush=True)


def cmd_rm2score(a) -> None:
    """新旧种子生成器对比（2026-11-03，种子换成 2B 自己的错句）：synth 文件里合格的样本用 RM2 打 P(自然)；
    只统计原文不在 RM2 训练集里的（旧 synth 的合格样本有一部分就是 RM2 的训练负例）。"""
    from ragtruth import discrim
    from ragtruth.rm_data import OUT, RM2G_SYSTEM
    from ragtruth.rm_eval import option_logprobs, probs
    from train.gen_reward import NO_THINK
    cls = _clients(a.url)
    train_src = {r["group"] for r in read_jsonl(OUT / a.rm2_name / "train.jsonl")}
    rows = [x for x in read_jsonl(D / a.inp) if x.get("keep") and str(x["source_id"]) not in train_src]

    def run(x: dict) -> dict:
        cl = _pick(cls, x["id"])
        resp = cl.chat.completions.create(model=a.model, messages=[{"role": "system", "content": RM2G_SYSTEM},
                                                                   {"role": "user", "content": discrim.render(x["source"], x["context"], int(x["idx"]))}],
                                          max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = option_logprobs(resp.choices[0], "AB")
        return {"id": x["id"], "type": x["type"], "task_type": x["task_type"], "p_natural": probs(lp, a.rm2_temp)["A"] if lp else 0.0}

    with ThreadPoolExecutor(64) as ex:
        out = list(ex.map(run, rows))
    with open(D / (a.inp.replace(".jsonl", "") + f"_{a.rm2_name}.jsonl"), "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ps = sorted(x["p_natural"] for x in out)
    q = lambda f: round(ps[int(f * (len(ps) - 1))], 4) if ps else None  # noqa: E731
    print("RM2SCORE", json.dumps({"inp": a.inp, "rm2": a.rm2_name, "n": len(out), "mean": round(sum(ps) / max(1, len(ps)), 4), "median": q(0.5), "p90": q(0.9),
                                  "share_gt_0.5": round(sum(p > 0.5 for p in ps) / max(1, len(ps)), 4)}), flush=True)


def cmd_rm2data(a) -> None:
    """重造 RM2（rm2v2__mcgen，A 自然 / B 合成；系统提示 RM2G_SYSTEM，不假定句子有错）：
      自然 = 2B GRPO 第 300 步策略在训练原文上写的回答里的句子，不看对错（用户：不是错误的也行，不然容易走捷径——
             只拿 9B 判错的句子当自然，合成又只改 9B 判对的句子，RM2 就能靠「像不像 9B 判错的句子」来区分）；
             另加 MiniCPM 2B 基线里 Claude 标出的有错句（outputs/rt/minicpm/sys，156 句，全留）；
      合成 = synth：同一批回答上 gen-seed-mc 造出、RM3 判合格的改写句。
    两边在每个任务类型里一样多（自然从全部句子里随机抽）；按原文 8 : 1 : 1 切训练 / 校准 / 测试，原文不跨集合。
    第 300 步验收那 100 条（critic 按类型测试集）不用。"""
    from ragtruth import discrim
    from ragtruth.rm_data import OUT, RM2G_SYSTEM
    sc = {x["id"]: x["p"] for x in json.load(open(D / "critic9b_scores.json", encoding="utf-8"))}
    nat = []
    for r in read_jsonl(D / "answers.jsonl"):
        if r["finish"] != "stop" or REFUSAL in r["response"].lower():
            continue
        for i in range(1, len(r["sentences"]) + 1):
            nat.append({"id": f"{r['id']}-S{i}-nat", "source_id": str(r["source_id"]), "source": r["source"], "sentences": r["sentences"], "idx": i,
                        "task_type": r["task_type"], "type": "-", "from": "grpo300", "p9b": sc.get(f"{r['id']}-S{i}")})
    base = Path("outputs/rt/minicpm/sys")
    ans = {r["id"]: r for r in read_jsonl(base / "answers.jsonl")}
    claude = []
    for r in read_jsonl(base / "final.jsonl"):
        if r["is_error"] and r["type"] in ("conflict", "baseless"):
            x = ans[r["resp_id"]]
            claude.append({"id": r["id"] + "-mcnat", "source_id": "mc-" + str(r["source_id"]), "source": x["source"], "sentences": x["sentences"],
                           "idx": int(r["idx"]), "task_type": r["task_type"], "type": r["type"], "from": "base_claude"})
    if a.seed_errors:   # 自然一侧补上 Claude 确认过的 2B 真错（gen_seed2b 两轮标注）：合成一侧过了 RM3 门槛几乎全是明显的错，
        # 不补的话「错得明显」本身就成了区分合成的信号（rm2v2__mcgen2 上 2B 真错 clear 的 P(自然) 0.67、minor 0.86、合成 0.09）
        ans = {r["id"]: r for r in read_jsonl(D / "answers.jsonl")}
        for f in ("labels.jsonl", "labels2.jsonl"):
            for x in read_jsonl(D / "seed2b" / f):
                if x["label"]["is_error"] and x["label"]["type"] in ("conflict", "baseless"):
                    r = ans[x["resp_id"]]
                    claude.append({"id": x["id"] + "-seednat", "source_id": str(r["source_id"]), "source": r["source"], "sentences": r["sentences"],
                                   "idx": int(x["idx"]), "task_type": r["task_type"], "type": x["label"]["type"], "from": "seed_claude"})
        dup = {c["id"].rsplit("-seednat", 1)[0] for c in claude}
        nat = [r for r in nat if r["id"].rsplit("-nat", 1)[0] not in dup]   # 同一句不进两次
    syn = [{"id": x["id"] + "-syn", "source_id": str(x["source_id"]), "source": x["source"], "sentences": x["context"], "idx": x["idx"],
            "task_type": x["task_type"], "type": x["type"], "from": "gen_seed_mc", "p9b": None} for x in read_jsonl(D / a.inp) if x.get("keep")]
    rng = random.Random(0)
    keep = []
    for t in sorted({r["task_type"] for r in syn}):
        c = sorted((r for r in claude if r["task_type"] == t), key=lambda r: r["id"])
        auto = sorted((r for r in nat if r["task_type"] == t), key=lambda r: r["id"])
        b = sorted((r for r in syn if r["task_type"] == t), key=lambda r: r["id"])
        m = min(len(c) + len(auto), len(b))
        if a.seed_errors:   # Claude 确认的真错最多占自然一侧一半（用户：自然一侧不能只有错句，不然容易走捷径），随机抽
            c = rng.sample(c, min(len(c), m // 2))
        else:
            c = c[:m]
        keep += c + rng.sample(auto, max(0, min(len(auto), m - len(c)))) + rng.sample(b, m)
    srcs = sorted({r["source_id"] for r in keep})
    rng.shuffle(srcs)
    n = len(srcs)
    split_of = {s: ("calib" if k < n // 10 else "test" if k < n // 5 else "train") for k, s in enumerate(srcs)}
    out = {"train": [], "calib": [], "test": []}
    for r in sorted(keep, key=lambda r: r["id"]):
        lab = "B" if r["from"] == "gen_seed_mc" else "A"
        out[split_of[r["source_id"]]].append({"id": r["id"], "prompt": [{"role": "system", "content": RM2G_SYSTEM},
                                                                        {"role": "user", "content": discrim.render(r["source"], r["sentences"], int(r["idx"]))}],
                                               "completion": lab, "label": lab, "task_type": r["task_type"], "type": r["type"], "from": r["from"], "group": r["source_id"]})
    d = OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    nk = [r for r in keep if r["from"] == "grpo300" and r.get("p9b") is not None]
    meta = {"natural_pool": len(nat), "claude_natural": len(claude), "synthetic_pool": len(syn), "max_tokens": 3072,
            "natural_kept_9b_flagged_share": round(sum(r["p9b"] > 0.5 for r in nk) / max(1, len(nk)), 4)}
    for sp, rs in out.items():
        if sp == "train":
            rng.shuffle(rs)
        with open(d / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[sp] = {"n": len(rs), "label": dict(sorted(collections.Counter(r["label"] for r in rs).items())),
                    "from": dict(collections.Counter(r["from"] for r in rs)), "task": dict(collections.Counter(r["task_type"] for r in rs))}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("RM2DATA", json.dumps(meta, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answers", "build", "hardpool", "synth", "harvest", "rm2score", "rm2data"])
    ap.add_argument("--rm1", default="rm1")
    ap.add_argument("--rm2", default="rm2")
    ap.add_argument("--rounds", type=int, default=1, help="harvest：每个 prompt 发几次请求（每次 n = --samples）")
    ap.add_argument("--example_pool", default="", help="build：示例池文件（hardpool 的输出），默认 RAGTruth 真实错句")
    ap.add_argument("--rm1_temp", type=float, default=1.0)
    ap.add_argument("--sev", default="", help="harvest：vLLM 上严重程度 RM 的名字；给了就用它代替 RM3 做门槛（P(显然) = A + B），原句有错用它的 P(F)")
    ap.add_argument("--sev_temp", type=float, default=1.0)
    ap.add_argument("--out", default="synth.jsonl", help="synth：输出文件名（在 outputs/rt/gentarget 下）")
    ap.add_argument("--inp", default="synth.jsonl", help="rm2score / rm2data：synth 文件")
    ap.add_argument("--rm2_temp", type=float, default=1.0)
    ap.add_argument("--rm2_name", default="rm2v2__mcgen", help="rm2score：打分用的 RM2（只统计它训练集外的原文）")
    ap.add_argument("--samples", type=int, default=1, help="synth：每个 prompt 采几条")
    ap.add_argument("--name", default="rm2v2__mcgen", help="rm2data：输出的 RM 名")
    ap.add_argument("--free_size", type=int, default=0, help="build：生成器 prompt 用改动大小不限的一版")
    ap.add_argument("--ctx", type=int, default=0, help="build：生成器 prompt 加「结合整条回答和上下句」")
    ap.add_argument("--type_split", type=int, default=0, help="build：一半 prompt 指定 baseless、一半指定 conflict")
    ap.add_argument("--seed_errors", type=int, default=0, help="rm2data：自然一侧补上 Claude 确认的 2B 真错")
    ap.add_argument("--base", default="/root/autodl-fs/models/MiniCPM5-2B")
    ap.add_argument("--rm3", default="rm3")
    ap.add_argument("--rm3_temp", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=0.7, help="synth：RM3 P(合格) 门槛，和 GRPO 奖励的门槛一致")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="g300")
    ap.add_argument("--k", type=int, default=4, help="每个原文采几条回答")
    ap.add_argument("--n_variants", type=int, default=2, help="每条回答出几个 prompt（示例不同）")
    ap.add_argument("--eval_sources", type=int, default=40, help="留作快评的原文数（每条回答 1 个 prompt）")
    ap.add_argument("--max_chars", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"answers": cmd_answers, "build": cmd_build, "synth": cmd_synth, "rm2score": cmd_rm2score, "hardpool": cmd_hardpool, "harvest": cmd_harvest, "rm2data": cmd_rm2data}[a.cmd](a)


if __name__ == "__main__":
    main()
