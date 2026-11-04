"""回答 GRPO 的奖励（2026-11-02，用户：用 rm1_gold_9b 当回报训 9B，看幻觉能不能降）。

critic 逐句给 P(有幻觉) = P(B) + P(C)（温度校准后）。每条回答：
  奖励 = 1 − (0.5 × 平均 P + 0.5 × 最大 P)
平均 P 让每句都有梯度，最大 P 盯最差的一句；前者靠多写安全句可以稀释，后者句子越多越容易变大，两项放一起不偏向写长或写短。
连续概率，不用阈值（这版 critic 的 0.5 阈值误报约 10%）。

硬规则（不满足 0 分）防的是不答题也能拿高分：
  空回答、截断；拒答（"Unable to answer"，只训 9B 自己通常会答的 prompt，见 ragtruth.ans_data）；
  长度超出这个 prompt 上 9B 原始回答中位数的 0.5–1.5 倍；
  照抄原文（连续 6 词都在原文里的词占比，QA ≥ 0.9、Summary ≥ 0.5，取自 9B 原始回答 p99 以上）；
  结尾没有句末标点（故意打断；9B 原始 300 条回答 0 条）；原文没有中文而回答夹中文（9B 原始约 0.5%，Claude 判这种回答不可用）。

满意度护栏（第二轮，用户：不需要很准，只是不让模型为了降幻觉去钻空子）：只罚不奖。sat_pair 和这个 prompt 上 9B 原始的参考回答比，P = 新回答更好；sat_abs 直接判好不好。
  奖励 = max(0, 幻觉项 − w × max(0, 0.5 − P))
"""
from __future__ import annotations

import re

from ragtruth.convert import split_sentences
from ragtruth.rm_data import RM1_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.sat_build import SAT_ABS_SYSTEM, SAT_PAIR_SYSTEM, abs_user, pair_user
from ragtruth.task import render_input

NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
WORD = re.compile(r"\w+")
REFUSAL = re.compile(r"unable to answer", re.I)
COPY_MAX = {"QA": 0.9, "Summary": 0.5}
LEN_BAND = (0.5, 1.5)
WORD_LIMIT = re.compile(r"within (\d+) words", re.I)
END = re.compile(r"[.!?。！？\"'”’)\]*:]\s*$")
CJK = re.compile(r"[\u4e00-\u9fff]")


def copy_frac(answer: str, source: str, n: int = 6) -> float:
    a, s = WORD.findall(answer.lower()), WORD.findall(source.lower())
    grams = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    hit = [False] * len(a)
    for i in range(len(a) - n + 1):
        if tuple(a[i:i + n]) in grams:
            hit[i:i + n] = [True] * n
    return sum(hit) / max(1, len(a))


def sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in split_sentences(text)]


def check(text: str, finish: str, row: dict) -> str:
    """→ "ok" 或没过的那条硬规则。row 要有 task_type、source、ref_chars。"""
    t = text.strip()
    if not t or not sentences(t):
        return "empty"
    if finish != "stop":
        return "truncated"
    if REFUSAL.search(t):
        return "refusal"
    ref = row.get("ref_chars")
    if ref and not LEN_BAND[0] * ref <= len(t) <= LEN_BAND[1] * ref:
        return "length"
    if copy_frac(t, row["source"]) >= COPY_MAX.get(row["task_type"], 1.0):
        return "copy"
    if not END.search(t):
        return "unfinished"
    if CJK.search(t) and not CJK.search(row["source"]):
        return "language"
    return "ok"


def combine(ps: list[float]) -> float:
    return 1.0 - 0.5 * sum(ps) / len(ps) - 0.5 * max(ps)


def guarded(r: float, sat: float | None, w: float) -> float:
    """满意度只罚不奖：低于 0.5 才扣，不低于 0（硬规则没过的是 0，不能比它还低）。"""
    return r if sat is None else max(0.0, r - w * max(0.0, 0.5 - sat))


def over_limit(text: str, source: str) -> bool:
    m = WORD_LIMIT.search(source[:300])
    return bool(m) and len(WORD.findall(text)) > int(m.group(1))


class AnswerRewarder:
    """model：当回报的 critic（vLLM 上的 LoRA 名）。先打第一句再并发其余（同一回答的句子共享前缀，同时发都还没算完，命中不了缓存）。"""

    def __init__(self, client, model: str, temp: float, pool, sat_model: str = "", sat_kind: str = "pair", sat_temp: float = 1.0, sat_w: float = 0.0,
                 refuse_model: str = "", refuse_temp: float = 1.0, refuse_w: float = 0.0):
        # client 可以是列表（两张卡各一个同样的服务）：按原文分到固定的服务，同一原文的请求共享前缀缓存
        self.clients = client if isinstance(client, list) else [client]
        self.client, self.model, self.temp, self.pool = self.clients[0], model, temp, pool
        self.sat_model, self.sat_kind, self.sat_temp, self.sat_w = sat_model, sat_kind, sat_temp, sat_w
        self.refuse_model, self.refuse_temp, self.refuse_w = refuse_model, refuse_temp, refuse_w

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, st: dict) -> None:
        pass

    def pick(self, source: str):
        import zlib
        return self.clients[zlib.crc32(source[:4000].encode("utf-8")) % len(self.clients)]

    def p_error(self, source: str, sents: list[str], idx: int, model: str | None = None, temp: float | None = None, priority: int = 0) -> float:
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        msgs = [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": source, "sentences": sents, "idx": idx})}]
        r = self.pick(source).chat.completions.create(model=model or self.model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(r.choices[0], "ABC")
        if not lp:
            return 1.0
        p = probs(lp, temp or self.temp)
        return p["B"] + p["C"]

    def p_answer(self, source: str, sents: list[str], model: str | None = None, temp: float | None = None, priority: int = 0) -> list[float]:
        first = self.p_error(source, sents, 1, model, temp, priority)
        rest = list(self.pool.map(lambda i: self.p_error(source, sents, i, model, temp, priority), range(2, len(sents) + 1)))
        return [first] + rest

    def p_sat(self, row: dict, answer: str, priority: int = 0) -> float | None:
        """P(满意)。sat_pair：参考回答放 A（同一 prompt 的前缀共享缓存），P = 新回答（B）更好。没配模型或没有参考回答时 None。"""
        if not self.sat_model or (self.sat_kind == "pair" and not row.get("ref_text")):
            return None
        sys_, user = (SAT_PAIR_SYSTEM, pair_user(row["source"], row["ref_text"], answer)) if self.sat_kind == "pair" else (SAT_ABS_SYSTEM, abs_user(row["source"], answer))
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        r = self.pick(row["source"]).chat.completions.create(model=self.sat_model, messages=[{"role": "system", "content": sys_}, {"role": "user", "content": user}], max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(r.choices[0], "AB")
        if not lp:
            return None
        p = probs(lp, self.sat_temp)
        return p["B"] if self.sat_kind == "pair" else p["A"]

    def p_refuse(self, row: dict, priority: int = 0) -> float | None:
        """拒答 RM：P(这道题原文答不了，可以拒答)。只看题目（原文 + 问题），和回答无关；没配模型或不是 QA 时 None。"""
        if not self.refuse_model or row.get("task_type") != "QA":
            return None
        from ragtruth.refuse_data import RM_SYSTEM as REFUSE_SYSTEM
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        r = self.pick(row["source"]).chat.completions.create(model=self.refuse_model, messages=[{"role": "system", "content": REFUSE_SYSTEM}, {"role": "user", "content": row["source"].strip()}],
                                                max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(r.choices[0], "AB")
        return probs(lp, self.refuse_temp)["B"] if lp else None

    def describe(self, text: str, finish: str, row: dict) -> dict:
        t = text.strip()
        return {"why": check(t, finish, row), "chars": len(t), "n_sent": len(sentences(t)), "copy": round(copy_frac(t, row["source"]), 4),
                "refusal": bool(REFUSAL.search(t)), "over_limit": over_limit(t, row["source"])}

    def score_group(self, row: dict, gens: list[dict]) -> list[dict]:
        """同一 prompt 的一组：先打一条把原文这段前缀放进缓存，再并发其余。过不了硬规则的不调 critic。"""
        out = [self.describe(g["text"], g["finish"], row) for g in gens]
        live = [i for i, o in enumerate(out) if o["why"] == "ok"]
        for o in out:
            o.update(reward=0.0, p_mean=None, p_max=None)

        def one(i: int) -> None:
            ps = self.p_answer(row["source"], sentences(gens[i]["text"].strip()))
            sat = self.p_sat(row, gens[i]["text"].strip())
            out[i].update(reward=guarded(combine(ps), sat, self.sat_w), p_mean=sum(ps) / len(ps), p_max=max(ps), ps=[round(p, 4) for p in ps], sat=sat)

        if live:
            one(live[0])
            list(self.group_pool.map(one, live[1:]))
        refused = [o for o in out if o["why"] == "refusal"]
        if refused and self.refuse_model:   # 用户：MiniCPM 偏向拒答，要降拒答率。拒答的奖励 = refuse_w × P(可以拒答)，最高 refuse_w，比一般的好回答低
            pr = self.p_refuse(row)
            for o in refused:
                o.update(reward=self.refuse_w * pr if pr is not None else 0.0, p_refuse=pr)
        return out

    group_pool = None
