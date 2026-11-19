"""专项补 critic 的张冠李戴盲区里占比最大的一种形态：说话人 / 出处错位（这句话、这个看法是谁的弄错了）。

attr_fill 换的是叙述句里的人名地名，真实错误里这种只占很小一部分；真实的主要是三种写法，这里按规则各造一种：
  swap  原句是「X 说 ……」，把 X 换成原文里另一个说过话的人（A 的话写成 B 说的）
  tag   原句是记者自己的叙述（原文对应的句子不在引号里、也没有「某某说」），给它挂上附近一位说话人（叙述写成某人说的）
  merge 回答里相邻两句，前一句是「X 说 ……」，后一句是记者的叙述，合成一句「X 说 ……，and ……」（叙述被并进 X 的话里）
说话人不只是有名字的人，也包括「the second mate」「a senior analyst」这样的称呼（从原文里「某某说」的主语里取）。
宿主句、原文范围和 attr_fill 一样（摘要任务、RAGTruth 训练集、不含核心集留出和验收原文）。
原句只取一部分当忠实句（上一轮错句和原句一比一，critic 整体变保守），而且只取核实过的来源（参考回答、RAGTruth 没被标错的句子）。

  python -m ragtruth.spk_fill rows                      # 本地：→ outputs/rt/attr_fill/sp1/{items,add}.jsonl
  python -m ragtruth.attr_fill build --round sp1 --name rm1__core_sp1 --arms l      # 远程：和 attr_fill 共用
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from ragtruth.attr_fill import MEDIA, O, has, hosts, sources, write

REPORT = {"say", "tell", "state", "note", "explain", "argue", "claim", "add", "announce", "warn", "admit", "acknowledge", "write", "testify", "confirm", "insist", "recall",
          "emphasize", "stress", "assert", "contend", "declare", "comment", "remark", "reply", "respond"}
SAY = {"say", "tell", "add", "write", "explain", "note", "state"}
QUOTE = re.compile(r"[\"“”]")
BAD_HEAD = {"who", "which", "that", "it", "this", "one", "some", "many", "others", "people", "article", "report", "story", "statement", "post", "source", "sources", "spokesman", "spokeswoman",
            "man", "woman", "speaker", "person", "group", "site", "website", "page", "video", "complaint", "document", "letter", "study", "email", "show", "film", "revelation", "news", "narrator", "author"}
DET = {"the", "a", "an"}
LEAD = DET | {"one"}
WHO = {"PERSON", "ORG", "GPE", "NORP"}
NAME = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}


def lower_first(s: str, doc) -> str:
    t = doc[0]
    keep = t.text == "I" or (t.pos_ == "PROPN" and t.text.lower() not in LEAD) or (t.ent_type_ in NAME and t.text.lower() not in LEAD) or (len(t.text) > 1 and t.text.isupper())
    return s if keep else s[0].lower() + s[1:]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def low_det(s: str) -> str:
    w = s.split(" ", 1)
    return (w[0].lower() + " " + w[1]) if len(w) == 2 and w[0].lower() in LEAD else s


def speakers(doc) -> list[tuple]:
    """「某某说」的主语 → [(动词 token, 主语名词短语 span)]。代词、关系代词、太长或太泛的主语不要。"""
    chunk = {c.root.i: c for c in doc.noun_chunks}
    out = []
    for t in doc:
        if t.lemma_.lower() in REPORT and t.pos_ == "VERB" and (any(c.dep_ in ("ccomp", "parataxis") for c in t.children) or t.dep_ == "parataxis" or (t.lemma_.lower() in SAY and QUOTE.search(t.sent.text))):   # 后面要跟着说的内容（"claimed the life of" 不算）
            for c in t.children:
                if c.dep_ == "nsubj" and c.i in chunk:
                    sp = chunk[c.i]
                    if c.pos_ in ("NOUN", "PROPN") and len(sp) <= 5 and c.text.lower() not in BAD_HEAD and "\n" not in sp.text and (sp[0].text.lower() in LEAD or sp[0].pos_ == "PROPN") and (c.pos_ == "NOUN" or c.ent_type_ in WHO) and not (sp.end < len(doc) and doc[sp.end].text.lower() in ("of", "for", "at", "in")):
                        out.append((t, sp))
    return out


def plural(sp) -> bool:
    return sp.root.tag_ in ("NNS", "NNPS")


class Src:
    def __init__(self, doc):
        self.sents = [s for s in doc.sents if s.text.strip()]
        self.words = [{t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop and len(t) > 2} for s in self.sents]
        self.low = [s.text.lower() for s in self.sents]
        start = {s.start: i for i, s in enumerate(self.sents)}
        sid, cur = {}, 0
        for t in doc:
            cur = start.get(t.i, cur)
            sid[t.i] = cur
        # 这一句是不是记者自己的叙述：句子里没有引号、没有「某某说」、也不落在一段跨句的引语里面
        n_q, inside = 0, []
        for s in self.sents:
            inside.append(n_q % 2 == 1)
            n_q += len(QUOTE.findall(s.text))
        said = {sid[v.i] for v, _ in speakers(doc)} | {i for i, s in enumerate(self.sents) if "according to" in s.text.lower()}
        self.narr = [not inside[i] and not QUOTE.search(s.text) and i not in said and not any(t.lemma_.lower() in REPORT for t in s) for i, s in enumerate(self.sents)]
        self.spk = collections.defaultdict(lambda: {"text": collections.Counter(), "at": [], "plural": False, "proper": False})   # 关键词（主语的中心词）→ 写法、出现在哪几句
        for v, sp in speakers(doc):
            k = sp.root.text.lower()
            if k in MEDIA or sp.text.lower() in MEDIA or len(k) < 3:
                continue
            d = self.spk[k]
            d["text"][sp.text.strip()] += 1
            d["at"].append(sid[v.i])
            d["plural"] = plural(sp)
            d["proper"] = sp.root.pos_ == "PROPN"

    def support(self, words: set, n: int) -> tuple[list[int], list[int]]:
        score = [len(words & w) for w in self.words]
        return [i for i, v in enumerate(score) if v >= n], score

    def name(self, k: str, short: bool) -> str:
        """说话人的写法：有名字的人，宿主句用的是单个词就只写姓，否则写最常见的全称。"""
        d = self.spk[k]
        full = max(sorted(d["text"]), key=lambda t: (d["text"][t], len(t)))
        return full.split()[-1] if short and d["proper"] else full


def content(doc, drop: set = frozenset()) -> set:
    return {t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 2} - drop


def near(xs: list[int], i: int) -> int:
    return min(abs(x - i) for x in xs)


def do_swap(S: Src, d, rng, a) -> dict | None:
    sp = speakers(d)
    if len(sp) != 1:
        return None
    v, x = sp[0]
    kx = x.root.text.lower()
    if not has(kx, " ".join(S.low)):
        return None
    xt = {t.text.lower() for t in x}
    sup, score = S.support(content(d, {t.lemma_.lower() for t in x}), a.min_overlap)
    if not sup:
        return None
    zone = {j for i in sup for j in (i - 1, i, i + 1)} | {i for i, l in enumerate(S.low) if has(kx, l)}
    cands = []
    for k, s in S.spk.items():
        if k == kx or k in xt or not s["proper"] or has(k, d.text.lower()) or s["plural"] != plural(x) or any(0 <= j < len(S.low) and has(k, S.low[j]) for j in zone):
            continue
        if {w.lower() for t in s["text"] for w in t.split()} & (xt - DET):
            continue
        cands.append(k)
    if not cands:
        return None
    k = rng.choice(sorted(cands))
    new = S.name(k, len(x) == 1)
    new = cap(new) if x.start == 0 else low_det(new)
    out = d[: x.start].text_with_ws + new + d[x.end - 1].whitespace_ + d[x.end:].text_with_ws
    best = max(sup, key=lambda i: (score[i], -i))
    return {"form": "swap", "orig_speaker": x.text, "new_speaker": new, "sentence": out.strip(), "evidence": S.sents[best].text.strip(), "overlap": score[best]}


def narration_host(S: Src, d, a) -> tuple[int, int] | None:
    """这句话是不是只对应原文里记者的叙述：→ (证据句下标, 重合词数)。"""
    if speakers(d) or QUOTE.search(d.text) or "according to" in d.text.lower() or re.search(r"(article|passage|summary)", d.text.lower()) or len(d) < 8 or not d.text.strip().endswith("."):
        return None
    if any(t.lemma_.lower() in REPORT for t in d):
        return None
    sup, score = S.support(content(d), a.min_overlap)
    if not sup or not all(S.narr[i] for i, v in enumerate(score) if v >= max(2, a.min_overlap - 1)):   # 沾边的原文句子都得是叙述，不然这句话里可能有一部分真是某人说的
        return None
    best = max(sup, key=lambda i: (score[i], -i))
    return (best, score[best]) if score[best] >= a.min_overlap + 1 else None


def pick_speaker(S: Src, d, ev: int, rng, a, avoid: set = frozenset()) -> str | None:
    cands = [k for k, s in S.spk.items() if k not in avoid and not has(k, d.text.lower()) and not has(k, S.low[ev]) and near(s["at"], ev) <= a.window and not s["plural"]]   # 证据句里提到的人不挂（多半就是这句话的主角）
    if not cands:
        return None
    m = min(near(S.spk[k]["at"], ev) for k in cands)
    return rng.choice(sorted(k for k in cands if near(S.spk[k]["at"], ev) <= m + 1))


def do_tag(S: Src, d, rng, a) -> dict | None:
    h = narration_host(S, d, a)
    if not h:
        return None
    k = pick_speaker(S, d, h[0], rng, a)
    if not k:
        return None
    y, text = S.name(k, False), d.text.strip()
    verb = rng.choice(["said", "said", "said", "noted"])
    form = "back" if d[0].pos_ in ("ADV", "CCONJ", "SCONJ") else rng.choice(["front", "front", "back", "according"])
    if form == "front":
        out = f"{cap(y)} {verb} {lower_first(text, d)}"
    elif form == "back":
        out = f"{text[:-1]}, {low_det(y)} {verb}."
    else:
        out = f"According to {low_det(y)}, {lower_first(text, d)}"
    return {"form": "tag", "orig_speaker": "", "new_speaker": y, "sentence": out, "evidence": S.sents[h[0]].text.strip(), "overlap": h[1]}


def do_merge(S: Src, d, d2, rng, a) -> dict | None:
    sp = speakers(d)
    if len(sp) != 1 or sp[0][1].start > 2 or sp[0][0].i > sp[0][1].end + 1 or QUOTE.search(d.text) or not d.text.strip().endswith(".") or len(d) > 25 or ";" in d.text or ", and" in d.text:
        return None
    h = narration_host(S, d2, a)
    if not h or len(d2) > 25 or re.search(r"(however|but|though|although|meanwhile)", d2.text.lower()) or d2[0].pos_ in ("PRON", "CCONJ", "SCONJ", "ADV"):
        return None
    kx = sp[0][1].root.text.lower()
    if has(kx, d2.text.lower()) or kx not in S.spk:
        return None
    out = f"{d.text.strip()[:-1]}, and {lower_first(d2.text.strip(), d2)}"
    return {"form": "merge", "orig_speaker": "", "new_speaker": sp[0][1].text, "sentence": out, "evidence": S.sents[h[0]].text.strip(), "overlap": h[1]}


def cmd_rows(a) -> None:
    import spacy
    from ragtruth.critic_harvest import row
    from ragtruth.rm_data import RM1_OPT
    nlp = spacy.load("en_core_web_sm")
    src = sources()
    hs = hosts(src, a.own_arms, a.own_per_source, a.round)
    used = sorted({h["source_id"] for h in hs})
    body = {sid: src[sid].split("\n", 1)[1] if "\n" in src[sid] else src[sid] for sid in used}
    S = {sid: Src(d) for sid, d in zip(used, nlp.pipe([body[s] for s in used], batch_size=16))}
    by = collections.defaultdict(list)
    for h in hs:
        by[h["source_id"]].append(h)
    items, stat = [], collections.Counter()
    for sid in used:
        if not S[sid].spk:
            stat["source_no_speaker"] += 1
            continue
        rng = random.Random(f"{a.round}|{sid}")
        cand = [(h, i) for h in by[sid] for i in range(len(h["sentences"])) if h["ok"][i]]
        rng.shuffle(cand)
        cand = cand[: a.try_per_source]
        docs = {}
        need = sorted({h["sentences"][j] for h, i in cand for j in (i, i + 1) if j < len(h["sentences"])})
        for t, d in zip(need, nlp.pipe(need, batch_size=64)):
            docs[t] = d
        got, seen, pair_n = collections.Counter(), set(), collections.Counter()
        for h, i in cand:
            if sum(got.values()) >= a.per_source:
                break
            d = docs[h["sentences"][i]]
            if d.text.strip().lower() in seen or len(d) < 6:
                continue
            forms = sorted(("swap", "tag", "merge"), key=lambda f: (got[f], rng.random()))   # 三种写法轮着来
            for f in forms:
                if got[f] >= a.per_form:
                    continue
                if f == "swap":
                    s = do_swap(S[sid], d, rng, a)
                elif f == "tag":
                    s = do_tag(S[sid], d, rng, a)
                else:
                    s = do_merge(S[sid], d, docs[h["sentences"][i + 1]], rng, a) if i + 1 < len(h["sentences"]) and h["ok"][i + 1] else None
                if s and pair_n[(f, s["new_speaker"].lower())] < a.per_pair:
                    break
            else:
                continue
            seen.add(d.text.strip().lower())
            pair_n[(f, s["new_speaker"].lower())] += 1
            got[f] += 1
            sents = list(h["sentences"])
            bad = sents[:i] + [s["sentence"]] + sents[i + (2 if f == "merge" else 1):]
            items.append({"id": f"spk-{h['kind']}-{sid}-{sum(got.values())}", "source_id": sid, "host": h["kind"], "idx": i + 1, "sentences": sents, "bad_sentences": bad, "original": sents[i], **s})
            stat[f"host_{h['kind']}"] += 1
            stat[f"form_{f}"] += 1
    B = O / a.round
    write(B / "items.jsonl", items)
    add = []
    for x in items:
        g = "attr-" + x["source_id"]
        add.append(row(x["id"] + "-err", src[x["source_id"]], x["bad_sentences"], x["idx"], RM1_OPT["conflict"], "Summary", g, "spk_fill_error"))
        if x["host"] != "own" and random.Random(f"{a.round}|orig|{x['id']}").random() < a.orig_frac:
            add.append(row(x["id"] + "-ok", src[x["source_id"]], x["sentences"], x["idx"], RM1_OPT["faithful"], "Summary", g, "spk_fill_original"))
            stat["originals"] += 1
    write(B / "add.jsonl", add)
    print("SPK_FILL_ROWS", json.dumps({"sources_available": len(src), "sources_used": len({x["source_id"] for x in items}), "errors": len(items), "rows": len(add), **dict(sorted(stat.items()))}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rows")
    s.add_argument("--round", default="sp1")
    s.add_argument("--own_arms", nargs="*", default=["cont9dyn", "cont8dyn", "satdyn"])
    s.add_argument("--own_per_source", type=int, default=12)
    s.add_argument("--per_source", type=int, default=14, help="每篇原文最多造几条")
    s.add_argument("--per_form", type=int, default=7, help="每篇原文每种写法最多几条")
    s.add_argument("--per_pair", type=int, default=2, help="同一篇原文里同一种写法挂同一个说话人最多几条")
    s.add_argument("--try_per_source", type=int, default=400)
    s.add_argument("--min_overlap", type=int, default=3, help="宿主句和原文证据句至少重合几个实词")
    s.add_argument("--window", type=int, default=4, help="tag：说话人说话的句子离证据句最多几句")
    s.add_argument("--orig_frac", type=float, default=0.5, help="核实过的宿主句里，多大比例把原句也加进去当忠实句")
    s.set_defaults(fn=cmd_rows)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
