"""专项补 critic 的一类盲区：张冠李戴（把原文里 A 的事安到原文里的 B 头上）。

和按操作合成的补法（ops_fill）的区别：不用错误生成器，按规则构造，保证对口、花样铺开：
  - 宿主句只取摘要任务的句子，三种来源：核心集改正后的参考回答、RAG 模型训练时自己写的回答、RAGTruth 训练集里各模型的回答（没被标错的句子）；
    原文都来自 RAGTruth 训练集（验收原文全在测试集），核心集留出的原文也不用；
  - 每条只换一个主体（人 / 机构 / 地点 / 群体），换进来的必须是同一篇原文里出现过的另一个同类主体，其余一字不动；
  - 原主体要能在原文里找到支持这句话的句子（证据句，和宿主句有足够多的实词重合），换进来的主体不能出现在任何支持这句话的原文句子里；
  - 每条错句配它的原句（标为忠实），同一句只差一个主体、标签相反。
两组训练集（都在基线训练集上加同一批句子）：
  l = 只给标签（单字母，和基线同格式）
  t = 先写核对过程再给结论：句子里的主体、原文证据句、原文里的主体，最后一行 Answer: <字母>；基线的句子不写核对过程，只有最后一行

  python -m ragtruth.attr_fill rows  --own_arms cont9dyn cont8dyn satdyn          # 本地：构造 → outputs/rt/attr_fill/<round>/{items,add}.jsonl
  python -m ragtruth.attr_fill build --base rm1__core_r2_bfcl                     # 远程：→ outputs/rt/rm/<name>{l,t}
  python -m ragtruth.attr_fill score --answers x.jsonl --url ... --model m --out y.json   # t 组的打分（先生成核对过程，取 Answer: 后那个字母的概率）
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re
import shutil
from pathlib import Path

from common.schema import read_jsonl

CORE = Path("outputs/rt/core")
RT = Path("external_data/ragtruth/dataset")
O = Path("outputs/rt/attr_fill")
KIND = {"PERSON": "person", "ORG": "org", "GPE": "place", "LOC": "place", "FAC": "place", "NORP": "group"}
MEDIA = {"cnn", "bbc", "reuters", "ap", "associated press", "daily mail", "mailonline", "hollywood reporter", "twitter", "facebook", "instagram", "youtube"}   # 新闻的署名来源，到处都出现，换进去不像话
STOPTOK = {"the", "of", "and", "a", "an", "for", "in", "on", "at", "de", "la", "s", "&", "'s"}
T_TAIL = ("If the sentence attributes an action, statement or property to a named person, organisation, place or group, first write the check on three lines:\n"
          "Claim subject: <the subject as the sentence has it>\nSource: \"<the source sentence that states this>\"\nSource subject: <who or what the source says it is>\n"
          "Then answer on the last line as `Answer: <letter>` (A, B or C). If no such check applies, write only the last line.")
ANS = re.compile(r"Answer:\s*$")


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 宿主句 ──

def sources() -> dict[str, str]:
    """摘要任务、RAGTruth 训练集、不在核心集留出里的原文：id → 助手收到的完整输入。"""
    split = {}
    for r in read_jsonl(RT / "response.jsonl"):
        split[str(r["source_id"])] = r["split"]
    held = {str(r["source_id"]) for r in read_jsonl(CORE / "core.jsonl") if r["split"] != "train"}
    return {str(r["source_id"]): r["prompt"] for r in read_jsonl(RT / "source_info.jsonl")
            if r["task_type"] == "Summary" and split.get(str(r["source_id"])) == "train" and str(r["source_id"]) not in held}


def hosts(src: dict[str, str], own_arms: list[str], own_per_source: int, seed: str) -> list[dict]:
    """→ [{kind, source_id, sentences, ok}]；ok[i]：第 i 句能不能当宿主（RAGTruth 的回答里被标错的句子不当）。"""
    from ragtruth.convert import split_sentences
    out, seen = [], set()

    def add(kind: str, sid: str, sents: list[str], ok: list[bool]) -> None:
        key = (sid, tuple(sents))
        if sid in src and len(sents) >= 2 and key not in seen:
            seen.add(key)
            out.append({"kind": kind, "source_id": sid, "sentences": sents, "ok": ok})

    for r in read_jsonl(CORE / "genpool_train.jsonl"):   # 核心集改正后的参考回答
        if r["task_type"] == "Summary":
            add("ref", str(r["source_id"]), list(r["sentences"]), [True] * len(r["sentences"]))
    own = collections.defaultdict(list)
    for arm in own_arms:   # RAG 模型训练时自己写的回答（没核实，原句当忠实句有少量噪声）
        for f in sorted(glob.glob(str(CORE / f"loop2_{arm}" / "run" / "samples" / "*.jsonl.gz"))):
            for l in gzip.open(f, "rt", encoding="utf-8"):
                d = json.loads(l)
                if d["task_type"] == "Summary" and d.get("finish") == "stop" and not d.get("refusal") and d.get("why") == "ok":
                    own[str(d["prompt_id"])].append(d["text"].strip())
    for sid in sorted(own):
        xs = sorted(set(own[sid]))
        random.Random(f"{seed}|own|{sid}").shuffle(xs)
        for t in xs[:own_per_source]:
            add("own", sid, [t[a:b] for a, b in split_sentences(t)], None)
    for r in read_jsonl(RT / "response.jsonl"):   # RAGTruth 训练集里各模型的回答
        sid, t = str(r["source_id"]), r["response"]
        if sid not in src or r.get("quality") != "good":
            continue
        spans = split_sentences(t)
        bad = [(x["start"], x["end"]) for x in r["labels"]]
        add("rt", sid, [t[a:b] for a, b in spans], [not any(a < e and s < b for s, e in bad) for a, b in spans])
    for h in out:
        if h["ok"] is None:
            h["ok"] = [True] * len(h["sentences"])
    return out


# ── 构造 ──

def toks(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if w not in STOPTOK]


def clean(ent_text: str) -> str:
    return re.sub(r"^(?:the|The)\s+", "", re.sub(r"(?:'s|’s)$", "", ent_text.strip())).strip()


class Src:
    """一篇原文：句子、每句的实词、各类主体。"""

    def __init__(self, doc):
        self.sents = [s for s in doc.sents if s.text.strip()]
        self.words = [{t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop and len(t) > 2} for s in self.sents]
        self.low = [s.text.lower() for s in self.sents]
        self.ents = collections.defaultdict(collections.Counter)   # 类别 → 写法 → 次数
        for e in doc.ents:
            k, t = KIND.get(e.label_), clean(e.text)
            if k and t and t[0].isupper() and len(t) > 2 and "\n" not in t:
                self.ents[k][t] += 1
        self.sent_ents = [[(KIND[e.label_], clean(e.text)) for e in s.ents if e.label_ in KIND] for s in self.sents]
        # 小模型的实体类别会标错（人标成机构、船名标成人）：同一个写法在这篇原文里七成以上是同一类才算数
        tot = collections.Counter()
        for k in self.ents:
            tot.update(self.ents[k])
        self.kind_of = {t: k for k in self.ents for t, n in self.ents[k].items() if n / tot[t] >= 0.7}
        self.count = tot
        self.name_pos = {}   # 人名里的单个词 → 它在全名里是名还是姓（0 / -1）
        for t, k in self.kind_of.items():
            ws = t.split()
            if k == "person" and len(ws) >= 2:
                self.name_pos.setdefault(ws[0].lower(), 0)
                self.name_pos[ws[-1].lower()] = -1
        self.text_low = doc.text.lower()


def key_of(kind: str, text: str) -> str:
    """在原文里找这个主体用的关键词：人取姓（最后一个词），其余取整个写法。"""
    return text.split()[-1].lower() if kind == "person" else text.lower()


def has(key: str, low: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", low) is not None


def swap_one(S: Src, sent_doc, rng: random.Random, min_overlap: int) -> dict | None:
    """给一句话换一个主体。→ {kind, orig, new, sentence, evidence, src_subject, overlap} 或 None。"""
    text = sent_doc.text
    ents = [(KIND[e.label_], e) for e in sent_doc.ents if e.label_ in KIND and clean(e.text) and clean(e.text)[0].isupper()]
    if not ents:
        return None
    ent_toks = {w for _, e in ents for w in toks(e.text)}
    words = {t.lemma_.lower() for t in sent_doc if t.is_alpha and not t.is_stop and len(t) > 2} - ent_toks
    score = [len(words & w) for w in S.words]
    support = [i for i, v in enumerate(score) if v >= min_overlap]   # 支持这句话的原文句子
    order = list(ents)
    rng.shuffle(order)
    order.sort(key=lambda ke: {"person": 0, "org": 1, "place": 1, "group": 2}[ke[0]])   # 人优先：张冠李戴主要是人，人名的类别也标得最准
    for kind, e in order:
        orig = clean(e.text)
        single = len(orig.split()) == 1
        pos = S.name_pos.get(orig.lower()) if kind == "person" and single else None
        if S.kind_of.get(orig) != kind and pos is None:   # 原文里这个写法不是稳定的同一类
            continue
        if (e.start > 1 and sent_doc[e.start - 1].is_title) or (e.end < len(sent_doc) and sent_doc[e.end].is_title):   # 只标到名字的一部分
            continue
        k0 = orig.lower() if pos == 0 else key_of(kind, orig)
        ev = [i for i in support if has(k0, S.low[i])]
        if not ev:
            continue
        best = max(ev, key=lambda i: (score[i], -i))
        cands = []
        for c, n in S.ents[kind].items():
            kc = key_of(kind, c)
            if set(toks(c)) & (ent_toks | set(toks(orig))) or has(kc, text.lower()) or any(has(kc, S.low[i]) for i in support):
                continue
            if S.kind_of.get(c) != kind or c.lower() in MEDIA:
                continue
            if kind == "person":   # 人只用原文里出现过全名、而且被提到不止一次的
                if len(c.split()) < 2 or len(re.findall(r"(?<![a-z0-9])" + re.escape(kc) + r"(?![a-z0-9])", S.text_low)) < 2:
                    continue
            elif S.count[c] < 2:
                continue
            cands.append(c)
        if not cands:
            continue
        new = rng.choice(sorted(cands))
        if kind == "person" and single:
            new = new.split()[0 if pos == 0 else -1]
            if has(new.lower(), text.lower()) or new.lower() in ent_toks:
                continue
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(orig) + r"(?![A-Za-z0-9])", new, text)
        if out == text:
            continue
        subj = next((t for kk, t in S.sent_ents[best] if kk == kind and has(k0, t.lower())), orig)
        return {"kind": kind, "orig": orig, "new": new, "sentence": out, "evidence": S.sents[best].text.strip(), "src_subject": subj, "overlap": score[best]}
    return None


def clip(ev: str, key: str, n: int = 60) -> str:
    """证据句太长就留关键词前后共 n 个词。"""
    ws = ev.split()
    if len(ws) <= n:
        return ev
    i = next((j for j, w in enumerate(ws) if key in w.lower()), 0)
    a = max(0, min(i - n // 2, len(ws) - n))
    return ("… " if a else "") + " ".join(ws[a:a + n]) + (" …" if a + n < len(ws) else "")


def check_text(claim: str, ev: str, subj: str, letter: str) -> str:
    return f"Claim subject: {claim}\nSource: \"{ev}\"\nSource subject: {subj}\nAnswer: {letter}"


def cmd_rows(a) -> None:
    import spacy
    from ragtruth.critic_harvest import row
    from ragtruth.rm_data import RM1_OPT
    nlp = spacy.load("en_core_web_sm")
    src = sources()
    hs = hosts(src, a.own_arms, a.own_per_source, a.round)
    used = sorted({h["source_id"] for h in hs})
    body = {sid: src[sid].split("\n", 1)[1] if "\n" in src[sid] else src[sid] for sid in used}   # 第一行是指令，主体只从正文里找
    S = {sid: Src(d) for sid, d in zip(used, nlp.pipe([body[s] for s in used], batch_size=16))}
    by = collections.defaultdict(list)
    for h in hs:
        by[h["source_id"]].append(h)
    items, stat = [], collections.Counter()
    for sid in used:
        rng = random.Random(f"{a.round}|{sid}")
        cand = [(h, i) for h in by[sid] for i in range(len(h["sentences"])) if h["ok"][i]]
        rng.shuffle(cand)
        cand.sort(key=lambda hi: {"ref": 0, "own": 1, "rt": 2}[hi[0]["kind"]])   # 轮流取三种来源
        buckets = collections.defaultdict(list)
        for hi in cand:
            buckets[hi[0]["kind"]].append(hi)
        mixed = [x for tup in zip(*[buckets[k] + [None] * (max(map(len, buckets.values())) - len(buckets[k])) for k in sorted(buckets)]) for x in tup if x] if buckets else []
        got, seen_sent, pair_n = 0, set(), collections.Counter()
        docs = nlp.pipe([h["sentences"][i] for h, i in mixed[: a.try_per_source]], batch_size=64)
        for (h, i), d in zip(mixed, docs):
            if got >= a.per_source:
                break
            if d.text.strip().lower() in seen_sent:
                continue
            s = swap_one(S[sid], d, rng, a.min_overlap)
            if not s or pair_n[(s["orig"], s["new"])] >= a.per_pair:
                continue
            seen_sent.add(d.text.strip().lower())
            pair_n[(s["orig"], s["new"])] += 1
            got += 1
            items.append({"id": f"attr-{h['kind']}-{sid}-{got}", "source_id": sid, "host": h["kind"], "idx": i + 1, "sentences": h["sentences"], "original": h["sentences"][i], **s})
            stat[f"host_{h['kind']}"] += 1
            stat[f"kind_{s['kind']}"] += 1
    B = O / a.round
    write(B / "items.jsonl", items)
    add = []
    for x in items:
        k0 = key_of(x["kind"], x["orig"])
        ev = clip(x["evidence"], k0)
        bad = list(x["sentences"])
        bad[x["idx"] - 1] = x["sentence"]
        e = row(x["id"] + "-err", src[x["source_id"]], bad, x["idx"], RM1_OPT["conflict"], "Summary", "attr-" + x["source_id"], "attr_fill_error")
        o = row(x["id"] + "-ok", src[x["source_id"]], x["sentences"], x["idx"], RM1_OPT["faithful"], "Summary", "attr-" + x["source_id"], "attr_fill_original")
        e["check"] = check_text(x["new"], ev, x["src_subject"], RM1_OPT["conflict"])
        o["check"] = check_text(x["orig"], ev, x["src_subject"], RM1_OPT["faithful"])
        add += [e, o]
    write(B / "add.jsonl", add)
    print("ATTR_FILL_ROWS", json.dumps({"sources_available": len(src), "sources_with_hosts": len(used), "sources_used": len({x["source_id"] for x in items}),
                                        "hosts": dict(collections.Counter(h["kind"] for h in hs)), "swaps": len(items), "rows": len(add), **dict(sorted(stat.items()))}))


# ── 训练集 ──

def to_think(r: dict) -> dict:
    """单字母样本 → 核对格式：system 的最后一句换成核对说明；completion 是核对过程 + Answer 行（没有核对过程的只有 Answer 行）。"""
    sys_ = r["prompt"][0]["content"]
    head, tail = sys_.rsplit("\n\n", 1)
    assert tail.startswith("Answer with a single letter"), tail[:60]
    comp = r.get("check") or f"Answer: {r['label']}"
    return {**{k: v for k, v in r.items() if k != "check"}, "prompt": [{"role": "system", "content": head + "\n\n" + T_TAIL}] + r["prompt"][1:], "completion": comp}


def think_messages(source: str, sentences: list[str], idx: int) -> list[dict]:
    from ragtruth.rm_data import RM1_SYSTEM
    from ragtruth.task import render_input
    head, _ = RM1_SYSTEM.rsplit("\n\n", 1)
    return [{"role": "system", "content": head + "\n\n" + T_TAIL}, {"role": "user", "content": render_input({"source": source, "sentences": sentences, "idx": idx})}]


def cmd_build(a) -> None:
    from ragtruth.rm_data import OUT
    base = OUT / a.base
    held = set()
    for sp in ("calib.jsonl", "test.jsonl"):
        for r in read_jsonl(base / sp):
            held |= set(re.findall(r"\d{3,}", str(r.get("group", ""))))
    accept = set().union(*json.load(open("configs/rt/accept_split.json", encoding="utf-8")).values())
    add = list(read_jsonl(O / a.round / "add.jsonl"))
    src = {r["group"].split("-", 1)[1] for r in add}
    assert not src & accept, f"{len(src & accept)} added sources are acceptance sources"
    drop = src & held
    add = [r for r in add if r["group"].split("-", 1)[1] not in drop]   # 和基线校准 / 测试集同原文的不加
    base_train = list(read_jsonl(base / "train.jsonl"))
    stat = {"dropped_sources_in_calib_test": len(drop)}
    for arm in a.arms:
        d = OUT / f"{a.name}{arm}"
        d.mkdir(parents=True, exist_ok=True)
        train = base_train + add
        random.Random(f"attr_fill|{a.round}|shuffle").shuffle(train)   # 两组同一个顺序，只差格式
        train = [to_think(r) for r in train] if arm == "t" else [{k: v for k, v in r.items() if k != "check"} for r in train]
        write(d / "train.jsonl", train)
        for sp in ("calib.jsonl", "test.jsonl"):
            shutil.copy(base / sp, d / sp)
        meta = {**json.load(open(base / "meta.json", encoding="utf-8")), "base": a.base, "arm": arm, "added": len(add), "format": "check_then_answer" if arm == "t" else "single_letter",
                "added_label": dict(sorted(collections.Counter(r["label"] for r in add).items())), "added_sources": len(src - drop),
                "train": {"n": len(train), "label": dict(sorted(collections.Counter(r["label"] for r in train).items()))}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        stat[arm] = {"added": len(add), "train": len(train)}
    print("ATTR_FILL_BUILD", json.dumps(stat), flush=True)


# ── t 组打分 ──

def answer_probs(choice) -> dict | None:
    """生成结果里最后一个 Answer: 后面那个 token 的 A / B / C 概率。"""
    acc, want, out = "", False, None
    for t in (choice.logprobs.content if choice.logprobs else None) or []:
        if want and t.token.strip():
            lp = {o: -30.0 for o in "ABC"}
            for c in t.top_logprobs:
                h = c.token.strip().rstrip(".").upper()
                if h in lp:
                    lp[h] = max(lp[h], c.logprob)
            out, want = lp, False
        acc += t.token
        if ANS.search(acc):
            want = True
    return out


def cmd_score(a) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from ragtruth.rm_eval import probs
    from train.ans_reward import NO_THINK
    cl = OpenAI(base_url=a.url, api_key="x", timeout=900)

    def one(src: str, sents: list[str], i: int) -> dict:
        r = cl.chat.completions.create(model=a.model, messages=think_messages(src, sents, i), max_tokens=a.max_tokens, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NO_THINK["extra_body"])
        lp = answer_probs(r.choices[0])
        txt = r.choices[0].message.content or ""
        p = probs(lp, a.temp) if lp else {"A": 0.5, "B": 0.5, "C": 0.0}
        return {"p": p["B"] + p["C"], "p_conflict": p["B"], "p_baseless": p["C"], "fmt": lp is not None, "checked": "Claim subject:" in txt, "text": txt[:400]}

    def resp(r: dict) -> list[dict]:
        with ThreadPoolExecutor(8) as ex:
            xs = list(ex.map(lambda i: one(r["source"], r["sentences"], i), range(1, len(r["sentences"]) + 1)))
        return [{"id": f"{r['id']}-S{i + 1}", **x} for i, x in enumerate(xs)]

    rows = [json.loads(l) for l in open(a.answers, encoding="utf-8")]
    rows = [r for r in rows if r.get("sentences") and r.get("finish") == "stop"]
    with ThreadPoolExecutor(8) as ex:
        out = [x for g in ex.map(resp, rows) for x in g]
    open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False))
    print("SCORED", a.out, len(rows), "answers", len(out), "sentences", "format ok", sum(x["fmt"] for x in out), "with check", sum(x["checked"] for x in out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rows")
    s.add_argument("--round", default="at1")
    s.add_argument("--own_arms", nargs="*", default=["cont9dyn", "cont8dyn", "satdyn"])
    s.add_argument("--own_per_source", type=int, default=12, help="每篇原文最多取几条 RAG 模型自己的回答当宿主")
    s.add_argument("--per_source", type=int, default=6, help="每篇原文最多造几条")
    s.add_argument("--per_pair", type=int, default=2, help="同一篇原文里同一对（原主体 → 新主体）最多几条")
    s.add_argument("--try_per_source", type=int, default=400)
    s.add_argument("--min_overlap", type=int, default=3, help="宿主句和原文证据句至少重合几个实词（主体之外）")
    s.set_defaults(fn=cmd_rows)
    s = sub.add_parser("build")
    s.add_argument("--round", default="at1")
    s.add_argument("--base", default="rm1__core_r2_bfcl")
    s.add_argument("--name", default="rm1__core_at1")
    s.add_argument("--arms", nargs="+", default=["l", "t"])
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("score")
    s.add_argument("--answers", required=True)
    s.add_argument("--url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--temp", type=float, default=1.0)
    s.add_argument("--max_tokens", type=int, default=200)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
