"""按操作清单造错的生成器的 GRPO 奖励，分三个阶段逐步加难（一上来就要求「critic 认不出」时只有约 6% 的样本拿得到分）。

策略输出 = gen_ops 的格式（{"plan"?, "idx", "from"?, "to"?, "sentence", "edit"}），prompt 行带 source / sentences / op（要求的操作）。
硬规则（0 分）：gen_v2.parse 不过（解析不出、句号越界、原样输出、照抄别的句子）；改写句长到原句 3 倍以上且多 300 字；点破题目的词。
四个条件：
  真错      sev4（ragtruth.rm_ops 的 sev__v4）P(明显错) + P(轻微错)
  自然      nat4（rm2__v4）P(自然)
  操作一致  opm（opm__v1）P(改动做的是要求的操作)
  不是整句替换  改写句和原句的词序列相似度 ≥ min_sim（程序算，默认 0.3）。只是底线：改多改少本身不是目标
阶段一：奖励 = 三个概率的几何平均（相似度不够时 × fail_scale）。不看 critic，先把「按指令造出干净的真错」练稳。
阶段二、三：主目标换成难度，但阶段一的目标不许下降（前一阶段的目标是后一阶段的约束）。
        奖励 = 阶段一的分 + w × 阶段一的分 × h（h 只给四个条件都过门槛的样本）。阶段一的分原样留着、权重不变，难度是加在上面的。
        w 自动调：最近一批样本的阶段一平均分低于进入本阶段时的水平（ref × tol）就把 w 调小，回到水平之上再慢慢调大。
        这样难度只在「不拖累前面几项」的范围内被优化；前面几项一掉，难度项自己让路。
        阶段二的 h：critic P(有幻觉) 从 0.9 到 0.1 线性升到 1，低于 0.1 的停在 0.5（不单调奖励「完全骗过 critic」，之前那样做生成器塌成了造非错句）。
        阶段三的 h：只在 critic 拿不准的区间给分（0.1→0.3 升、0.3–0.5 为 1、0.5→0.9 降）。
"""
from __future__ import annotations

from ragtruth.rm_eval import option_logprobs, probs
from train.gen_reward import NO_THINK, edit_sim
from train.gen_reward_v2 import BANNED, hardness


def soft_hard(p: float) -> float:
    """阶段二的难度系数：critic P 越低越高，0.9 以上为 0，0.1 以下停在 0.5。"""
    if p >= 0.9:
        return 0.0
    if p < 0.1:
        return 0.5
    return (0.9 - p) / 0.8


class OpsRewarder:
    def __init__(self, client, temps: dict[str, float], stage=1, err_gate=0.5, nat_gate=0.5, op_gate=0.5, min_sim=0.3, fail_scale=0.1,
                 ref=0.0, w_hard=0.5, w_min=0.02, tol=0.97, window=512):
        import collections
        import threading
        self.client, self.t = client, temps
        self.stage, self.err_gate, self.nat_gate, self.op_gate, self.min_sim, self.fail_scale = stage, err_gate, nat_gate, op_gate, min_sim, fail_scale
        # ref = 进入本阶段时阶段一的平均分（同一批提示上量的）；0 = 不自动调，w 固定为 w_hard
        self.ref, self.w_max, self.w_min, self.tol, self.window = ref, w_hard, w_min, tol, window
        self.w = w_hard if not ref else w_hard / 4   # 从小的权重起步，前面几项稳得住再放大
        self.recent, self.n_seen, self.lock = collections.deque(maxlen=window), 0, threading.Lock()

    def state_dict(self) -> dict:
        return {"w": self.w, "recent": list(self.recent), "n_seen": self.n_seen}

    def load_state_dict(self, st: dict) -> None:
        self.w, self.n_seen = float(st.get("w", self.w)), int(st.get("n_seen", 0))
        self.recent.extend(st.get("recent", []))

    def track(self, base: float) -> float:
        """记下这条样本的阶段一得分；每满半个窗口看一次：最近的平均分掉到 ref × tol 以下就把难度权重调小，否则慢慢调大。返回当前权重。"""
        with self.lock:
            self.recent.append(base)
            self.n_seen += 1
            if self.ref and self.n_seen % (self.window // 2) == 0 and len(self.recent) >= self.window // 2:
                m = sum(self.recent) / len(self.recent)
                self.w = max(self.w_min, self.w * 0.7) if m < self.ref * self.tol else min(self.w_max, self.w * 1.15)
            return self.w

    def option_p(self, model: str, msgs: list[dict], opts: str, priority: int = 0) -> dict | None:
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        r = self.client.chat.completions.create(model=model, messages=msgs, max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(r.choices[0], opts)
        return probs(lp, self.t[model]) if lp else None

    def raw(self, text: str, row: dict, priority: int = 0) -> dict:
        from ragtruth.gen_v2 import parse
        from ragtruth.rm2_v3 import prompt as nat_prompt
        from ragtruth.rm_data import RM1_SYSTEM
        from ragtruth.rm_ops import opm_prompt
        from ragtruth.sev_gen import prompt_dne
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
        x = {"original": o, "sentence": s, "source": row["source"], "context": ctx, "idx": rec["idx"]}
        pe = self.option_p("sev4", prompt_dne(x), "ABC", priority)
        pn = self.option_p("nat4", nat_prompt(x), "AB", priority)
        po = self.option_p("opm", opm_prompt(x, row["op"]), "AB", priority)
        out = {"why": "ok", "idx": rec["idx"], "original": o, "sentence": s, "edit_gen": rec["edit_gen"], "sim": edit_sim(o, s),
               "p_err": (pe["A"] + pe["B"]) if pe else 0.0, "p_clear": pe["A"] if pe else 0.0, "p_natural": pn["A"] if pn else 0.0, "p_op": po["A"] if po else 0.0, "p_halluc": None}
        if self.stage >= 2:
            p1 = self.option_p("rm1", [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": row["source"], "sentences": ctx, "idx": rec["idx"]})}], "ABC", priority)
            out["p_halluc"] = (p1["B"] + p1["C"]) if p1 else 1.0
        return out

    def score(self, text: str, row: dict) -> dict:
        s = self.raw(text, row)
        op = row["op"]
        base = {"type": op, "tactic": op, "tactic_raw": op, "op": "rewrite", "reason": "", "evidence": ""}
        if s["why"] != "ok":
            if self.stage >= 2:
                self.track(0.0)
            return {**base, "idx": 0, "sentence": "", "reward": 0.0, "why": s["why"], "qualified": False}
        geo = (s["p_err"] * s["p_natural"] * s["p_op"]) ** (1 / 3)
        near = s["sim"] >= self.min_sim
        gate = near and s["p_err"] >= self.err_gate and s["p_natural"] >= self.nat_gate and s["p_op"] >= self.op_gate
        r1 = geo if near else self.fail_scale * geo   # 阶段一的分：三个阶段都原样在奖励里
        if self.stage == 1:
            r, h, q, w = r1, 0.0, gate, 0.0
        else:
            w = self.track(r1)
            h = (soft_hard if self.stage == 2 else hardness)(s["p_halluc"]) if gate else 0.0
            r, q = r1 + w * r1 * h, bool(gate and h > 0)
        return {**base, **s, "p_halluc": s["p_halluc"] if s["p_halluc"] is not None else 0.0, "gate": gate, "geo": geo, "r1": r1, "w_hard": w, "hardness": h, "reward": r, "qualified": bool(q),
                "p_correct": s["p_err"], "hard": h, "real": s["p_natural"], "div": 0.0}
