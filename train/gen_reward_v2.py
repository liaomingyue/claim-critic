"""生成器 v2 的 GRPO 奖励（核心目标是 sev 奖励、提高合格样本量；自然度和 critic 作为门槛过滤）。

策略输出 = gen_v2 格式（{"idx", "sentence", "edit"}），prompt 行带 source / sentences / mechanism（要求的机制）。
  1. 硬规则（0 分）：gen_v2.parse 不过（解析不出、句号越界、原样输出、照抄别的句子）；改写句 > 原句 3 倍且多 300 字以上；点破题目的词。
  2. sev = sev v3（输入：要求的机制 + 规则改动清单；训练中不调 9B 写描述，描述留空）的 P(要求的那种·显然) + P(要求的那种·细微)。
  3. 门槛：自然度 RM2 v3b P(自然) ≥ nat_gate；--crit 时另加 critic r2 P(有幻觉) ∈ [crit_lo, crit_hi)。
     过了门槛：奖励 = sev；没过：奖励 = fail_scale × sev（低于任何过门槛的样本，留一点梯度）。
shape="hard"（核心是造 critic 盲区样本——critic 难度当主奖励，自然度和是否真错作为过滤）：
  门槛 = sev P(有错) ≥ err_gate（默认 0.9；难区 sev ≥ 0.5 精度只有约 55%，≥ 0.9 约 85%）且 RM2 自然 ≥ nat_gate；没过 0 分。
  过了：奖励 = 0.1 + 0.9 × h(critic P)，h 在 0.3–0.5 为 1，0.1→0.3 线性升、0.5→0.9 线性降，< 0.1 和 ≥ 0.9 为 0
  （不单调奖励「骗过 critic」：之前的 gated 版就是这样坍缩成造 sev 会误判的非错句，真错只剩 24%）。
组内优势值（gen_grpo.group_advantages）按「类型」= 要求的机制分组，同一 prompt 机制相同，等于组均值。
"""
from __future__ import annotations

import re

from train.gen_reward import NO_THINK
from ragtruth.rm_eval import option_logprobs, probs

BANNED = re.compile(r"hallucinat|faithful|\bthe source\b|\bsource text\b|\bnot (?:stated|mentioned) in\b|\berror\b", re.I)


def hardness_shape(p: float, shape: str) -> float:
    """难度系数的几种形状（0.7 以上的比例要压下去，往 0.5 以下推；做几组对比）。
    tri：0.1→0.3 升、0.3–0.5 为 1、0.5→0.9 降；low：只奖励 < 0.7（0.1–0.5 为 1、0.5→0.7 降）。"""
    if shape == "rank":   # 绝对部分同 tri；组内排名部分在 group_post 里加
        return hardness(p)
    if shape == "low":
        if p < 0.1 or p >= 0.7:
            return 0.0
        return 1.0 if p < 0.5 else (0.7 - p) / 0.2
    return hardness(p)


def hardness(p: float) -> float:
    """critic P(有幻觉) → 难度系数：0.3–0.5 最高，两头为 0（不单调奖励骗过 critic）。"""
    if p < 0.1 or p >= 0.9:
        return 0.0
    if p < 0.3:
        return (p - 0.1) / 0.2
    if p < 0.5:
        return 1.0
    return (0.9 - p) / 0.4


class V2SevRewarder:
    def __init__(self, client, temps: dict[str, float], nat_gate=0.5, crit=False, crit_lo=0.1, crit_hi=0.7, fail_scale=0.1, shape="sev", err_gate=0.9,
                 err_gate_start=None, warm_samples=0, hard_shape="tri", hard_base=0.1):
        import threading
        self.client, self.t = client, temps
        self.shape, self.err_gate = shape, err_gate
        self.hard_shape, self.hard_base = hard_shape, hard_base   # 过门槛的奖励 = base + (1 − base) × h
        # 动态门槛（前期 sev 门槛调低一点）：已打分样本数 0 → warm_samples，门槛从 err_gate_start 线性升到 err_gate
        self.err_gate_start, self.warm_samples, self.n_scored, self.lock = err_gate_start, warm_samples, 0, threading.Lock()
        if shape == "hard":
            crit = True
        self.nat_gate, self.crit, self.crit_lo, self.crit_hi, self.fail_scale = nat_gate, crit, crit_lo, crit_hi, fail_scale

    def state_dict(self) -> dict:   # 只有动态门槛的计数
        return {"n_scored": self.n_scored}

    def load_state_dict(self, st: dict) -> None:
        self.n_scored = int(st.get("n_scored", 0))

    def option_p(self, model: str, msgs: list[dict], opts: str, priority: int = 0) -> dict | None:
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        r = self.client.chat.completions.create(model=model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(r.choices[0], opts)
        return probs(lp, self.t[model]) if lp else None

    def raw(self, text: str, row: dict, priority: int = 0) -> dict:
        from ragtruth import discrim
        from ragtruth.gen_v2 import parse
        from ragtruth.rm_data import RM1_SYSTEM
        from ragtruth.rm2_v3 import RM2V3_SYSTEM
        from ragtruth.sev_gen import prompt_v3
        from ragtruth.task import render_input
        sents = row["sentences"]
        rec, why = parse(text, sents)
        if rec is None:
            return {"why": why}
        o, s = rec["original"], rec["sentence"]
        if len(s) > 3 * len(o) and len(s) - len(o) > 300:
            return {"why": "too_long"}
        if BANNED.search(s) and not BANNED.search(o):
            return {"why": "banned_word"}
        ctx = list(sents)
        ctx[rec["idx"] - 1] = s
        x = {"mechanism": row["mechanism"], "original": o, "sentence": s, "desc": "", "source": row["source"], "context": ctx, "idx": rec["idx"]}
        ps = self.option_p("sev", prompt_v3(x), "ABCD", priority)
        p2 = self.option_p("rm2v3", [{"role": "system", "content": RM2V3_SYSTEM}, {"role": "user", "content": discrim.render(row["source"], ctx, rec["idx"])}], "AB", priority)
        out = {"why": "ok", "idx": rec["idx"], "original": o, "sentence": s, "edit_gen": rec["edit_gen"],
               "p_req": (ps["A"] + ps["B"]) if ps else 0.0, "p_err": (ps["A"] + ps["B"] + ps["C"]) if ps else 0.0,
               "p_natural": p2["A"] if p2 else 0.0, "p_halluc": None}
        if self.crit:
            p1 = self.option_p("rm1", [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": row["source"], "sentences": ctx, "idx": rec["idx"]})}], "ABC", priority)
            out["p_halluc"] = (p1["B"] + p1["C"]) if p1 else 1.0
        return out

    def score(self, text: str, row: dict) -> dict:
        s = self.raw(text, row)
        m = row["mechanism"]
        base = {"type": m, "tactic": m, "tactic_raw": m, "op": "rewrite", "reason": "", "evidence": ""}
        if s["why"] != "ok":
            return {**base, "idx": 0, "sentence": "", "reward": 0.0, "why": s["why"], "qualified": False}
        if self.shape == "hard":
            with self.lock:
                self.n_scored += 1
                k = self.n_scored
            g = self.err_gate
            if self.err_gate_start is not None and self.warm_samples:
                g = self.err_gate_start + (self.err_gate - self.err_gate_start) * min(1.0, k / self.warm_samples)
            gate = s["p_natural"] >= self.nat_gate and s["p_err"] >= g
            h = hardness_shape(s["p_halluc"], self.hard_shape)
            r = (self.hard_base + (1 - self.hard_base) * h) if gate else 0.0
            return {**base, **s, "gate": gate, "err_gate_now": round(g, 3), "hardness": h, "reward": r, "qualified": bool(gate and h > 0),
                    "p_correct": s["p_err"], "hard": h, "real": s["p_natural"], "div": 0.0}
        gate = s["p_natural"] >= self.nat_gate
        if self.crit:
            gate = gate and self.crit_lo <= s["p_halluc"] < self.crit_hi
        r = s["p_req"] if gate else self.fail_scale * s["p_req"]
        ph = s["p_halluc"] if s["p_halluc"] is not None else 0.0
        return {**base, **s, "p_halluc": ph, "gate": gate, "reward": r, "qualified": bool(gate and s["p_req"] >= 0.5),
                "p_correct": s["p_req"], "hard": 0.0, "real": s["p_natural"], "div": 0.0}

    def group_post(self, scores: list[dict]) -> None:
        """组内相对难度（方向①）：同一提示的一组里，过门槛且 critic ≥ 0.1 的样本按 critic P 从低到高排名，
        奖励 = 0.5 × 排名分（比它更容易被 critic 判出的同伴占比）+ 0.5 × 绝对难度 tri(P)；critic < 0.1、没过门槛仍是 0（不奖励完全骗过 critic）。
        这样 0.99 和 0.93 之间也拉得开，每组都有梯度，模型一步步往下推；不设保底（h1 证明保底会鼓励明显错）。"""
        if self.shape != "hard" or self.hard_shape != "rank":
            return
        cand = [x for x in scores if x.get("gate") and x.get("p_halluc") is not None and x["p_halluc"] >= 0.1]
        if len(cand) < 2:
            for x in cand:
                x["rank"] = 0.5
                x["reward"] = 0.5 * 0.5 + 0.5 * hardness(x["p_halluc"])
            return
        ps = [x["p_halluc"] for x in cand]
        for x in cand:
            hi = sum(q > x["p_halluc"] for q in ps) + 0.5 * (sum(q == x["p_halluc"] for q in ps) - 1)
            x["rank"] = hi / (len(cand) - 1)
            x["reward"] = 0.5 * x["rank"] + 0.5 * hardness(x["p_halluc"])
