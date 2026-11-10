"""生成器 GRPO 的奖励（RM1 / 2 / 3 打分；考虑 DPO 的规则，避免 reward hack、不让模型走捷径）。

三个 RM 都是同一个 9B 底座上的 LoRA，挂在推理卡的同一个 vLLM 服务上（模型名 rm1 / rm2 / rm3），读选项概率、用各自的
全局温度校准。每条输出的奖励：

1. 硬规则（不满足直接 0 分，是「没造出合格错误」，不是格式扣分；不在意格式规则本身）：
   能解析出句号、改写句、类型；句号在范围内；改写句不能和原句相同（规范化后）；不能整句照抄回答里的另一句；
   不能写成一大段（> 原句 3 倍且多出 300 字符以上，防止「一句里塞好几个错」骗过 RM）；改写句里不能出现
   hallucination / faithful / "the source" 这类点破题目的词（防止「明说这里有错」骗 RM3）。
2. RM3 门槛：P(合格) < 0.5 时只给 0.1 × P(合格)（比任何合格样本都低，但保留一点梯度）。
   原句本身被 RM1 判有幻觉（P > 0.5）时给 0 分（Claude 抽检发现模型更常改本来就有问题的句子）。
3. 合格时：(w_hard × 难度 + w_real × 真实性 + w_div × 多样性) / 权重和，三项都在 [0, 1]：
   - 难度：4p(1−p)，p = RM1 P(有幻觉)；
   - 真实性：RM2 P(自然)；
   - 两项都在同一错误类型内用滑动均值 / 标准差标准化后过 sigmoid（DPO 第二轮的「按类型分组」：类型之间不互相挤压，
     模型不能靠只写某一类拿高分）；
   - 多样性：1/√(1 + N(类型, 改法))，N 是最近 window（512）条合格样本里这一类出现的次数（按实时分布算；累计计数到第 50 步就只剩
     0.06，等于没有）。改法封闭到标准清单，对不上的归 other，other 不给多样性分（防编名字）；
     另外某一类型在最近窗口里占比低于 type_floor 时，这一类型的样本多样性记满分（防类型塌缩）。
组内优势值（train.gen_grpo）按「同一 prompt、同一类型」的均值减（DPO 第二轮的「chosen 和 rejected 同一类型」）。
"""
from __future__ import annotations

import collections
import json
import math
import re
import threading

from ragtruth import audit_synth, discrim
from ragtruth.realgen import norm_tactic
from ragtruth.synth import TACTICS
from ragtruth.rm_data import RM1_SYSTEM, RM2_SYSTEM, RM2G_SYSTEM, RM3_SYSTEM
from ragtruth.sev_rm import SEV_SYSTEM
from ragtruth.rm_eval import option_logprobs, probs
from ragtruth.synth import parse_json_obj
from ragtruth.task import render_input

NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
BANNED = re.compile(r"hallucinat|faithful|\bthe source\b|\bsource text\b|\bnot (?:stated|mentioned) in\b", re.I)
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
OTHER_TACTIC = "other"


def closed_tactic(t: str, kind: str) -> str:
    """改法封闭到标准清单，对不上的一律归 other（共用一个计数）。改法是模型自填的字符串，开放时每编一个新名字
    N(类型, 改法) 都是 0、多样性拿满分——第一版正式训练第 50–100 步模型就学会了这个（改法 18 → 71 种）。"""
    n = norm_tactic(t, kind)
    return n if n in TACTICS.get(kind, []) else OTHER_TACTIC


def parse(text: str, row: dict) -> tuple[dict | None, str]:
    """→ (rec, 原因)。rec 为 None 时原因是没过哪条硬规则。"""
    obj = parse_json_obj(text)
    if not isinstance(obj, dict):
        return None, "unparsable"
    try:
        i = int(obj.get("idx"))
    except (TypeError, ValueError):
        return None, "bad_idx"
    sents = row["sentences"]
    op = "insert" if str(obj.get("op") or "").strip().lower() == "insert" else "rewrite"
    if not 1 <= i <= len(sents) + (op == "insert"):
        return None, "bad_idx"
    sent = obj.get("sentence")
    if not isinstance(sent, str) or not sent.strip():
        return None, "empty_sentence"
    sent = sent.strip()
    kind = str(obj.get("type") or "").strip().lower()
    if kind not in ("conflict", "baseless"):
        return None, "bad_type"
    if row.get("want_type") and kind != row["want_type"]:   # prompt 指定了类型却造了另一种
        return None, "wrong_type"
    orig = "" if op == "insert" else sents[i - 1]
    if op == "rewrite" and norm(sent) == norm(orig):
        return None, "identical"
    if any(norm(sent) == norm(s) for k, s in enumerate(sents, 1) if k != i or op == "insert"):
        return None, "copied_other_sentence"
    if len(sent) > max(3 * len(orig), len(orig) + 300):
        return None, "too_long"
    if BANNED.search(sent):
        return None, "leaks_label"
    rec = {"source": row["source"], "sentences": sents, "idx": i, "op": op, "type": kind, "sentence": sent,
           "original": orig or "(none: this is a newly inserted sentence)", "tactic": closed_tactic(str(obj.get("tactic") or ""), kind),
           "tactic_raw": str(obj.get("tactic") or ""),
           "evidence": str(obj.get("evidence") or ""), "reason": str(obj.get("reason") or ""), "task_type": row["task_type"]}
    return rec, "ok"


def context(rec: dict) -> list[str]:
    i, s = rec["idx"], rec["sentences"]
    return s[:i - 1] + [rec["sentence"]] + (s[i - 1:] if rec["op"] == "insert" else s[i:])


class RunningStat:
    """按 key 的滑动均值 / 方差（指数平均），用于类型内标准化。"""

    def __init__(self, decay: float = 0.99):
        self.decay, self.m, self.v, self.n = decay, {}, {}, collections.Counter()

    def norm(self, key, x: float) -> float:
        m, v = self.m.get(key, x), self.v.get(key, 0.05)
        self.n[key] += 1
        d = self.decay if self.n[key] > 20 else 0.8   # 起步时更新快一点
        self.m[key] = d * m + (1 - d) * x
        self.v[key] = d * v + (1 - d) * (x - m) ** 2
        return 1 / (1 + math.exp(-(x - m) / math.sqrt(v + 1e-4)))


class Rewarder:
    def __init__(self, client, temps: dict[str, float], w_hard=1.0, w_real=1.0, w_div=1.0, type_floor=0.3, window=512, orig_max=0.5):
        self.orig_max = orig_max
        self.client, self.t = client, temps
        self.w = (w_hard, w_real, w_div)
        self.type_floor = type_floor
        self.stat = RunningStat()
        self.cls = collections.Counter()   # 最近窗口里每个（类型, 改法）的次数，和 recent 同步
        self.recent = collections.deque(maxlen=window)
        self.lock = threading.Lock()

    def state_dict(self) -> dict:
        with self.lock:
            return {"m": dict(self.stat.m), "v": dict(self.stat.v), "n": dict(self.stat.n), "recent": list(self.recent)}

    def load_state_dict(self, st: dict) -> None:
        self.stat.m, self.stat.v, self.stat.n = dict(st["m"]), dict(st["v"]), collections.Counter(st["n"])
        self.recent.clear()
        self.recent.extend(st["recent"])
        self.cls = collections.Counter(self.recent)

    def option_p(self, model: str, system: str, user: str, opts: str, priority: int = 0) -> dict | None:
        body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
        resp = self.client.chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                                   {"role": "user", "content": user}], max_tokens=8, temperature=0.0,
                                                   logprobs=True, top_logprobs=20, extra_body=body)
        lp = option_logprobs(resp.choices[0], opts)
        return probs(lp, self.t[model]) if lp else None

    def raw(self, text: str, row: dict, priority: int = 0) -> dict:
        """只打 RM 原始概率、不动滑动统计和多样性计数（训练中的快速评测也用它）。合格时才补 RM1 / RM2。"""
        rec, why = parse(text, row)
        if rec is None:
            return {"why": why}
        p3 = self.option_p("rm3", RM3_SYSTEM, audit_synth.render(rec), "ABCD", priority)
        pc = p3["A"] if p3 else 0.0
        out = {"why": "ok", "type": rec["type"], "op": rec["op"], "idx": rec["idx"], "tactic": rec["tactic"], "tactic_raw": rec["tactic_raw"],
               "p_correct": pc, "sentence": rec["sentence"], "reason": rec["reason"], "evidence": rec["evidence"]}
        if pc >= 0.5:
            if rec["op"] == "rewrite":   # 原句本身有问题就不算造出了错误（模型自己改不了原句，骗不了这一项）
                p0 = self.option_p("rm1", RM1_SYSTEM, render_input({"source": rec["source"], "sentences": rec["sentences"], "idx": rec["idx"]}), "ABC", priority)
                out["p_orig_halluc"] = (p0["B"] + p0["C"]) if p0 else 0.0
            p1 = self.option_p("rm1", RM1_SYSTEM, render_input({"source": rec["source"], "sentences": context(rec), "idx": rec["idx"]}), "ABC", priority)
            p2 = self.option_p("rm2", RM2_SYSTEM, discrim.render(rec["source"], context(rec), rec["idx"]), "AB", priority)
            out["p_halluc"] = (p1["B"] + p1["C"]) if p1 else 1.0
            out["p_natural"] = p2["A"] if p2 else 0.0
        return out

    def score(self, text: str, row: dict) -> dict:
        s = self.raw(text, row)
        if s["why"] != "ok":
            return {"reward": 0.0, "why": s["why"]}
        out = {k: v for k, v in s.items() if k not in ("p_halluc", "p_natural")}
        pc = s["p_correct"]
        if pc < 0.5:
            return {**out, "reward": 0.1 * pc, "qualified": False}
        if s.get("p_orig_halluc", 0.0) > self.orig_max:
            return {**out, "reward": 0.0, "qualified": False, "orig_flagged": True}
        rec = s
        ph, pn = s["p_halluc"], s["p_natural"]
        hard_raw = 4 * ph * (1 - ph)
        with self.lock:
            hard = self.stat.norm(("hard", rec["type"]), hard_raw)
            real = self.stat.norm(("real", rec["type"]), pn)
            c = (rec["type"], rec["tactic"])
            div = 1 / math.sqrt(1 + self.cls[c])
            if len(self.recent) == self.recent.maxlen:
                self.cls[self.recent[0]] -= 1
            self.recent.append(c)
            self.cls[c] += 1
            share = sum(t == rec["type"] for t, _ in self.recent) / len(self.recent)
            if len(self.recent) >= 64 and share < self.type_floor:
                div = 1.0
            if rec["tactic"] == OTHER_TACTIC:   # 没有多样性加分：否则 other 一少，编个新名字又能拿满分
                div = 0.0
        wh, wr, wd = self.w
        r = (wh * hard + wr * real + wd * div) / (wh + wr + wd)
        return {**out, "reward": r, "qualified": True, "p_halluc": ph, "p_natural": pn, "hard": hard, "real": real, "div": div}


def focus_tactic(t: str, kind: str) -> str:
    """定向版的改法封闭到 synth.FOCUS_TACTICS（prompt 里列的就是这几条）：模型照抄清单时前 40 个字符对得上，否则归 other。"""
    from ragtruth.synth import FOCUS_TACTICS
    n = norm(t)
    for x in FOCUS_TACTICS.get(kind, []):
        if n and (norm(x)[:40] in n or n[:40] in norm(x)):
            return x
    return OTHER_TACTIC


class TargetRewarder(Rewarder):
    """定向造错（合成 2B critic 漏掉的那几类错误，生成器也给出类型和理由；GRPO 训出和真实模型一致的错误）。
    主力奖励是真实性（RM2 P(自然)，同类型内标准化）；正确性（RM3）和 critic（RM1）只扣分：
      - 硬规则没过：0；RM3 P(合格) < gate：0.1 × P（过滤，和旧版一样保留一点梯度）；原句本身有错：0。
        原句有没有错优先用 prompt 里预先算好的 orig_p（训练外用 9B 金标 critic 打的，训练外调用用 9B），没有才用 RM1 现场打；
      - 合格时 r = (真实性 + w_div × 多样性) / (1 + w_div) − w_corr × (1 − P(合格)) − w_crit × 2 × max(0, P(critic 判出) − 0.5)，截到 [0, 1]：
        critic 一眼看出的错没有训练价值（它已经会了），扣分；判不出的不加分，避免生成器去钻 critic 的噪声（早先定向版失败的原因之一）。
    多样性只看同一 prompt 这组里的位置（place_diversity_target），改法和类型是模型自填的，不进奖励。"""

    def __init__(self, client, temps, w_div=0.2, w_corr=0.5, w_crit=0.3, gate=0.7, err_mode="a", shape="v1", w_real2=0.45, w_hard2=0.45, crit_max=0.98, crit_floor=0.4, rm2_ens=(), gate_lo=None, min_sim=0.0, base_fail=0.1, obvious_pen=0.1, blind_bonus=0.0, blind_thr=0.5, blind_lo=0.0, **kw):
        """第二版奖励（GRPO 打分拉不开差距，要改奖励；骗不过 critic 要想办法；只是理由或证据不对、句子确实错了的不该罚）：
        err_mode="ab"：P(真错) = RM3 的 P(A 合格) + P(B 句子错了但解释不对)，门槛和扣分都用它；
        shape="v2"：真实性和骗过 critic 都用对数几率映射到 [0, 1]（logodds01）的绝对分、都是加分项，按 w_real2 / w_hard2 / w_div 加权平均，
        再减 w_corr × (1 − P(真错))。v1 的真实性是类型内标准化的排名、critic 只在 P > 0.5 时线性扣分，合格样本 P(判出) 平均约 0.89，扣分几乎一样，拉不开。
        shape="filter"（几项分数有的好拿有的难拿，加权相加有问题；真实性做主奖励，其他用过滤）：没过滤掉的样本奖励只看真实性
        r = (logodds01(P自然) + w_div × 多样性) / (1 + w_div)；过滤：规则、原句干净、P(真错) ≥ gate、critic P(判出) ≤ crit_max，没过一律 0
        （门槛以下不再给 0.1 × P）。grpo3 合格样本 critic P(判出) 中位 0.982，≤ 0.98 约留一半；骗过 critic 的样本判自然的比例高得多
        （≤ 0.5 的一档 50% vs 全体 8%），真实性做主奖励也会顺带把 critic 判出的概率往下推。
        shape="soft"（过不了 critic 这关就永远拿不到自然度奖励，门卡不能太高；P < 0.5 就算骗过；正确性只保证句子真的错）：
        critic 不做过滤，改成乘在真实性上的系数：P(判出) < 0.5 记 1，0.5 → 1 线性降到 crit_floor；
        r = 系数 × (logodds01(P自然) + w_div × 多样性) / (1 + w_div)。只有 P(真错) = A + B < gate、原句有错、规则不过记 0。
        奖励设计对比（先把可能的实验列上去，各跑 50 步看数据）：
        shape="mult"：r = (sqrt(logodds01(P自然) × logodds01(1 − P判出)) + w_div × 多样性) / (1 + w_div)，两项都连续、相乘（实验 A5）；
        rm2_ens：再用这几个 RM2 各打一次 P(自然)，和 rm2 取平均（实验 B3，降尖锐度、少钻单个 RM2 的空子）。
        gate_lo（实验 C3：「是真错」的硬门槛改成连续系数）：P(真错) < gate_lo 记 0，gate_lo → gate 线性升到 1，再乘到奖励上。
        a0 上 50 步合格率 16.7% → 30.7%、骗过 9B 16.7% → 1.1%：没过门槛 0、过了奖励中位约 0.07，组内「过没过门槛」的分差
        压过了自然度的分差，GRPO 主要学会了过门槛（明显的错最容易过）。
        min_sim（实验 C4）：改写句和原句的词序列相似度（difflib ratio）低于它的记 0（几乎整句换掉）。Claude 抽查 a0 的 20 条 B 样本：
        18 条是真错，但第 50 步 15 条里 10 条和原句相似度 < 0.4（换成一句不相干的话，"S1: Some oral infections cause bad breath."）；
        2B 真实错句和改对句相似度 < 0.4 的只有 10%。
        shape="boundary"（实验 A6：critic 很大很小都不意味着好，越接近 0.5 越合适）：critic 系数 = max(crit_floor, 4P(1 − P))，
        0.5 处最高。a0/a1/a4 的 1031 条合格样本按 2B P(判出) 分段：0.3–0.7 的 9B 平均 P 0.81（确实是错、2B 拿不准）、判自然 49%；
        < 0.1 的 9B 也有 85% 判不出、判自然 97%（可能根本不算错）；≥ 0.98 的 428 条 9B 全判出、判自然 1%。
        shape="sev"（奖励显然错误、尤其张冠李戴 / 与原文矛盾）：用严重程度 RM（sev，6 选 1）代替 RM3，
        门槛 P(显然) = A + B ≥ gate（建议 0.5：盲区独立测试过门槛的 93% 确实有错），原句有错用 sev 的 P(F) > orig_max 判；
        合格的 r = (1 − P(critic 判出)) × (logodds01(P自然) + w_div × 多样性) / (1 + w_div)；没过门槛给 base_fail × P(显然) / gate。
        shape="boundary2"（实验 A7）：a6 放大评测（每 prompt 16 次）里 0.9 以上的样本涨了约 2.9 倍、0.3–0.7 只涨 1.05–1.3 倍——
        没过门槛记 0、组里六七成过不了，明显的错（P 0.95 → 系数 0.19、奖励约 0.03）仍高于 0，照样拿正优势。
        改成：P(真错) 没过门槛给基线 base_fail（规则不过、原句有错仍是 0）；合格的 r = 自然度项 × 4P(1 − P)，P ≥ 0.98 记 0。
        base_fail 0.1：用 a6 放大评测样本模拟，0.9–0.98 只有 17% 高于基线，0.3–0.7 全部高于，0.7–0.9 92%；造一个明显的错比没造出错还亏。"""
        super().__init__(client, temps, w_hard=0.0, w_real=1.0, w_div=w_div, **kw)
        self.w_corr, self.w_crit, self.gate = w_corr, w_crit, gate
        self.err_mode, self.shape, self.w_real2, self.w_hard2, self.crit_max, self.crit_floor = err_mode, shape, w_real2, w_hard2, crit_max, crit_floor
        self.rm2_ens = tuple(rm2_ens)
        self.gate_lo, self.min_sim, self.base_fail, self.obvious_pen = gate_lo, min_sim, base_fail, obvious_pen
        self.blind_bonus, self.blind_thr, self.blind_lo = blind_bonus, blind_thr, blind_lo

    def raw(self, text: str, row: dict, priority: int = 0) -> dict:
        rec, why = parse(text, row)
        if rec is None:
            return {"why": why}
        rec["tactic"] = focus_tactic(rec["tactic_raw"], rec["type"])
        if self.shape in ("sev", "gated", "add"):   # 严重程度 RM 代替 RM3：输入和 sev RM 训练时同格式（回答里已经换成改写句，见 ragtruth.sev_rm.prompt）
            ps = self.option_p("sev", SEV_SYSTEM, audit_synth.render({**rec, "sentences": context(rec)}), "ABCDEF", priority)
            pa = ps["A"] if ps else 0.0
            pc = (ps["A"] + ps["B"]) if ps else 0.0   # P(显然错误)
            p3 = ps
        else:
            p3 = self.option_p("rm3", RM3_SYSTEM, audit_synth.render(rec), "ABCD", priority)
            pa = p3["A"] if p3 else 0.0
            pc = (pa + p3["B"]) if p3 and self.err_mode == "ab" else pa   # ab：解释不对但句子确实错了也算
        out = {"why": "ok", "type": rec["type"], "op": rec["op"], "idx": rec["idx"], "tactic": rec["tactic"], "tactic_raw": rec["tactic_raw"],
               "p_correct": pc, "p_a": pa, "p_rm3": p3, "original": rec["original"], "sentence": rec["sentence"], "reason": rec["reason"], "evidence": rec["evidence"]}
        if self.shape in ("sev", "gated", "add"):
            out["p_orig_halluc"] = p3["F"] if p3 else 0.0   # 原句本身有错：用 sev 的 F（全 2B；不用 9B 预先算的 orig_p：9B 不能用于筛选）
        if pc >= (self.gate if self.gate_lo is None else self.gate_lo):
            if rec["op"] == "rewrite" and self.shape not in ("sev", "gated", "add"):
                if row.get("orig_p"):
                    out["p_orig_halluc"] = row["orig_p"][rec["idx"] - 1]
                else:
                    p0 = self.option_p("rm1", RM1_SYSTEM, render_input({"source": rec["source"], "sentences": rec["sentences"], "idx": rec["idx"]}), "ABC", priority)
                    out["p_orig_halluc"] = (p0["B"] + p0["C"]) if p0 else 0.0
            p1 = self.option_p("rm1", RM1_SYSTEM, render_input({"source": rec["source"], "sentences": context(rec), "idx": rec["idx"]}), "ABC", priority)
            u2 = discrim.render(rec["source"], context(rec), rec["idx"])
            p2s = [self.option_p(m, RM2G_SYSTEM, u2, "AB", priority) for m in ("rm2",) + self.rm2_ens]   # 定向版 RM2 不假定句子有错
            out["p_halluc"] = (p1["B"] + p1["C"]) if p1 else 1.0
            out["p_natural"] = sum(p["A"] if p else 0.0 for p in p2s) / len(p2s)
        return out

    def combine(self, s: dict) -> float:
        if self.shape == "gated":   # 真错(sev，score() 已卡 pc≥gate)+ critic 盲区 = 硬门；自然度 = 唯一优化目标，不设门（门槛放松）。
            ph = s["p_halluc"]
            if not (self.blind_lo <= ph <= self.blind_thr):   # critic 盲区硬门：判出 P 要落在 [blind_lo, blind_thr]（盲区/边界），否则无训练价值记 0
                return 0.0
            return (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])   # 自然度（p_natural 用 rm2__coreerr 服务）+ 多样性
        if self.shape == "add":   # critic 盲区 = 主目标，自然度 = 次目标，相加；sev门已卡真错。w_hard2>w_real2 让盲区为主
            blind = max(0.0, 1.0 - s["p_halluc"])   # critic 盲区深度：判不出（P 低）越深
            return (self.w_hard2 * blind + self.w_real2 * logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (self.w_hard2 + self.w_real2 + self.w[2])
        if self.shape == "sev":   # 新循环：门槛 P(显然) ≥ gate 之后，critic 越判不出越高、读起来越自然越高
            ph = s["p_halluc"]
            # 区间模式（blind_lo > 0：不要 critic 评分很低的，要离 0.5 近的）：难度项用离 0.5 的接近度（0.5 → 1，0 / 1 → 0）代替 1 − P
            hard = max(0.0, 1.0 - abs(ph - 0.5) / 0.5) if self.blind_lo > 0 else 1.0 - ph
            r = hard * (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            # 盲区加分（方案 A）：50 步里合格样本 P(critic) 约 0.9、自然度约 0.15，乘出来 0.01–0.02，组内拉不开，
            # 梯度主要来自没过门槛的保底项（推着过门槛 → 多造 critic 一眼看出的错）。critic 判不出（P < blind_thr）的合格样本固定加 blind_bonus
            # 对 critic 最有用的是离 0.5 近的边界样本（0.3–0.7），加分只给 [blind_lo, blind_thr] 区间
            return r + (self.blind_bonus if self.blind_lo <= s["p_halluc"] <= self.blind_thr else 0.0)
        if self.shape == "mult":
            return (math.sqrt(logodds01(s["p_natural"]) * logodds01(1 - s["p_halluc"])) + self.w[2] * s["div"]) / (1 + self.w[2])
        if self.shape in ("filter", "soft"):
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * (1.0 if self.shape == "filter" else crit_factor(s["p_halluc"], self.crit_floor))
        if self.shape == "boundary7":   # a12：明显的错（P ≥ 0.9）给负分 obvious_pen；P < 0.1 记 0；其余同 a11；保底同 a9（按离真错多近给）
            ph = s["p_halluc"]
            if ph >= 0.9:
                return -self.obvious_pen
            if ph < 0.1:
                return 0.0
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * 4 * ph * (1 - ph)
        if self.shape == "boundary6":   # a11（critic 基本能判清楚的两头，阈值用 0.1 和 0.9）：4P(1 − P) 只在 [0.1, 0.9) 给，外面记 0
            ph = s["p_halluc"]
            if ph < 0.1 or ph >= 0.9:
                return 0.0
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * 4 * ph * (1 - ph)
        if self.shape == "boundary5":   # a10：a7 的系数 4P(1 − P)（峰值 0.5），P < 0.02 或 ≥ 0.98 记 0；保底同 a9（按离真错多近给）
            ph = s["p_halluc"]
            if ph < 0.02 or ph >= 0.98:
                return 0.0
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * 4 * ph * (1 - ph)
        if self.shape == "boundary4":   # a9：帐篷同 a8；0.9–0.98 给 0.03（低于差一点过门槛的保底、高于不造错），≥ 0.98 记 0
            ph = s["p_halluc"]
            if ph >= 0.98:
                return 0.0
            if ph >= 0.9:
                return 0.03
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * tent(ph)
        if self.shape == "boundary3":
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * tent(s["p_halluc"])
        if self.shape == "boundary2":
            ph = s["p_halluc"]
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return 0.0 if ph >= 0.98 else r * 4 * ph * (1 - ph)
        if self.shape == "boundary":
            r = (logodds01(s["p_natural"]) + self.w[2] * s["div"]) / (1 + self.w[2])
            return r * max(self.crit_floor, 4 * s["p_halluc"] * (1 - s["p_halluc"]))
        if self.shape == "v2":
            base = (self.w_real2 * logodds01(s["p_natural"]) + self.w_hard2 * logodds01(1 - s["p_halluc"]) + self.w[2] * s["div"]) / (self.w_real2 + self.w_hard2 + self.w[2])
            return min(1.0, max(0.0, base - self.w_corr * (1 - s["p_correct"])))
        base = (s["real"] + self.w[2] * s["div"]) / (1 + self.w[2])
        pen = self.w_corr * (1 - s["p_correct"]) + self.w_crit * 2 * max(0.0, s["p_halluc"] - 0.5)
        return min(1.0, max(0.0, base - pen))

    def score(self, text: str, row: dict) -> dict:
        s = self.raw(text, row)
        if s["why"] != "ok":
            return {"reward": 0.0, "why": s["why"]}
        out = {k: v for k, v in s.items() if k not in ("p_halluc", "p_natural", "p_rm3")}
        pc = s["p_correct"]
        if self.min_sim and s["op"] == "rewrite" and edit_sim(s["original"], s["sentence"]) < self.min_sim:
            return {**out, "reward": 0.0, "qualified": False, "why_zero": "rewrote_whole"}
        if pc < (self.gate if self.gate_lo is None else self.gate_lo):
            if self.shape in ("boundary2", "boundary3"):
                return {**out, "reward": self.base_fail, "qualified": False}
            if self.shape in ("boundary4", "boundary5", "boundary6", "boundary7", "sev", "gated", "add"):   # a9 / a10：保底按离真错多近给。a8 固定 0.1：81–100 步 86% 的样本拿保底、合格 22.5% → 13.4%，平均奖励反而升
                return {**out, "reward": self.base_fail * pc / self.gate, "qualified": False}
            return {**out, "reward": 0.0 if self.shape in ("filter", "soft", "mult", "boundary") else 0.1 * pc, "qualified": False}
        if s.get("p_orig_halluc", 0.0) > self.orig_max:
            return {**out, "reward": 0.0, "qualified": False, "orig_flagged": True}
        ph, pn = s["p_halluc"], s["p_natural"]
        with self.lock:
            real = self.stat.norm(("real", s["type"]), pn)
        d = {**out, "qualified": True, "p_halluc": ph, "p_natural": pn, "real": real, "div": 1.0, "hard": 1 - ph}
        if self.shape == "filter" and ph > self.crit_max:   # critic 一眼看出的错：过滤掉（仍算合格，快评照常统计）
            return {**d, "reward": 0.0, "crit_filtered": True}
        d["reward"] = self.combine(d)
        if self.gate_lo is not None:   # 连续正确性系数；门槛以下的不算合格（快评口径不变）
            d["reward"] *= min(1.0, (pc - self.gate_lo) / max(1e-9, self.gate - self.gate_lo))
            d["qualified"] = pc >= self.gate
        return d


def edit_sim(a: str, b: str) -> float:
    """两句的词序列相似度（difflib ratio，小写词）。"""
    import difflib
    t = lambda x: re.findall(r"\w+", x.lower())  # noqa: E731
    return difflib.SequenceMatcher(a=t(a), b=t(b), autojunk=False).ratio()


def tent(p: float, lo: float = 0.02, peak: float = 0.3, hi: float = 0.9) -> float:
    """a8 的 critic 系数：lo → peak 线性升到 1，peak → hi 线性降到 0，两头外面记 0。
    Claude 判真错（5 次调用，48 条）：a7 的 2B P 0.1–0.3 真错 11/16、0.3–0.7 14/16，起点 0.3–0.7 14/16；< 0.1 先前 11/33。
    0.1–0.3 多半是真错、判自然的更多，峰值从 0.5 挪到 0.3；a7 在 0.9–0.98 还剩 84 条，P ≥ 0.9 记 0。"""
    if p < lo or p >= hi:
        return 0.0
    return (p - lo) / (peak - lo) if p <= peak else (hi - p) / (hi - peak)


def crit_factor(p_halluc: float, floor: float = 0.4) -> float:
    """critic 系数：P(判出) < 0.5 算骗过，记 1；0.5 → 1 线性降到 floor。"""
    return 1.0 if p_halluc < 0.5 else 1.0 - (1.0 - floor) * (p_halluc - 0.5) / 0.5


def logodds01(p: float, span: float = 8.0) -> float:
    """概率 → 对数几率线性映射到 [0, 1]（±span 截断）：0.0003 → 0、0.05 → 0.32、0.5 → 0.5、0.95 → 0.68。
    RM2 / critic 的分数大多挤在 0 或 1 附近，直接用概率或组内排名拉不开差距。"""
    p = min(1 - 1e-9, max(1e-9, p))
    return min(1.0, max(0.0, (math.log(p / (1 - p)) + span) / (2 * span)))


def place_diversity_target(scores: list[dict], rw: TargetRewarder) -> None:
    """同 place_diversity，但按 TargetRewarder.combine 重算奖励。"""
    q = [s for s in scores if s.get("qualified")]
    n = collections.Counter((s["op"], s["idx"]) for s in q)
    for s in q:
        s["div"] = 1 / math.sqrt(n[(s["op"], s["idx"])])
        s["reward"] = rw.combine(s)


def place_diversity(scores: list[dict], w: tuple[float, float, float]) -> None:
    """按同一 prompt 的一组输出算多样性：同一位置（操作, 句号）在这组合格样本里出现 n 次，每条 1/√n，替换 Rewarder.score 里按改法算的那一项。
    只看位置（要奖励换位置，不是在一个位置上换标签）：改法和类型都是模型自填，不进这一项。改法归 other 的仍记 0。原地改 div 和 reward。"""
    q = [s for s in scores if s.get("qualified")]
    n = collections.Counter((s["op"], s["idx"]) for s in q)
    wh, wr, wd = w
    for s in q:
        s["div"] = 0.0 if s["tactic"] == OTHER_TACTIC else 1 / math.sqrt(n[(s["op"], s["idx"])])
        s["reward"] = (wh * s["hard"] + wr * s["real"] + wd * s["div"]) / (wh + wr + wd)


def group_advantages(rewards: list[float], types: list[str | None], eps: float = 1e-4) -> list[float]:
    """同一 prompt 的一组输出：减去同类型输出的均值（同类型只有自己时减组均值），除以组标准差。
    全组奖励一样时优势全为 0（这组不贡献梯度）。"""
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    if std < eps:
        return [0.0] * n
    out = []
    for r, t in zip(rewards, types):
        same = [x for x, y in zip(rewards, types) if t is not None and y == t]
        base = sum(same) / len(same) if len(same) >= 2 else mean
        out.append((r - base) / (std + eps))
    return out


def load_temps(spec: str = "") -> dict[str, float]:
    """spec：vLLM 上的名字=RM 目录，逗号分隔（2B 版：rm1=rm1__mcgold,rm2=rm2v2__mc,rm3=rm3__mc）；空 = 9B 版。"""
    t = lambda rm: json.load(open(f"outputs/rt/rm/{rm}/eval/report.json", encoding="utf-8"))["temperature"]
    m = dict(kv.split("=", 1) for kv in spec.split(",")) if spec else {"rm1": "rm1", "rm2": "rm2v2", "rm3": "rm3"}
    return {k: t(v) for k, v in m.items()}
