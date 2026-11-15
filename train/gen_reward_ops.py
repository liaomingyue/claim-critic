"""按操作清单造错的生成器的 GRPO 奖励，分三个阶段逐步加难（一上来就要求「critic 认不出」时只有约 6% 的样本拿得到分）。

策略输出 = gen_ops 的格式（{"plan"?, "idx", "from"?, "to"?, "sentence", "edit"}），prompt 行带 source / sentences / op（要求的操作）。
硬规则（0 分）：gen_v2.parse 不过（解析不出、句号越界、原样输出、照抄别的句子）；改写句长到原句 3 倍以上且多 300 字；点破题目的词。
四个条件：
  真错      sev4（ragtruth.rm_ops 的 sev__v4）P(明显错) + P(轻微错)
  自然      nat4（rm2__v4）P(自然)
  操作一致  opm（opm__v1）P(改动做的是要求的操作)
  （改动大小不设条件：min_sim 默认 0。改写句和原句的相似度只记下来观察，不进奖励）
阶段一：奖励 = 三个概率的几何平均 × 「不明显」系数。主目标是把「按指令造出干净的真错」练稳，critic 不是主目标，但不能不管：
        验证模型和 critic 的打分相关 0.78，只追「真错」会顺带把样本推向 critic 早就认得出的那一类
        （实测不看 critic 训 50 步，critic ≥ 0.9 的占比 36% → 40%），到阶段二就拉不回来了。
        系数：critic P(有幻觉) ≤ obv_from（0.7）时为 1，往上线性降到 P = 1 时的 1 − obv_pen（0.5）。只压「明显」的那一头，不奖励骗过 critic。
阶段二、三：主目标换成难度，但阶段一的目标不许下降（前一阶段的目标是后一阶段的约束）。
        奖励 = 阶段一的分 + w × h（h 只给几个条件都过门槛的样本）。阶段一的分原样留着、权重不变，难度是加在上面的。
        约束（refs：进入本阶段时真错、自然、操作一致各自的平均水平）：某一项的近期平均掉到基准线下，就给这一项加补偿权重
        （奖励 += λ × 这一项的概率，λ 每次检查加 0.25、回到线上后慢慢减），难度权重 w 不低于起步值。
        最初的做法是前面几项一掉就把 w 调小：实测 w 被压到 0.2–0.4 后停住，难度项等于被关掉。难度项和约束都不能让步。
        阶段二的 h：critic P(有幻觉) 从 0.9 到 0.1 线性升到 1，低于 0.1 的停在 0.5（不单调奖励「完全骗过 critic」，之前那样做生成器塌成了造非错句）。
        阶段三的 h：只在 critic 拿不准的区间给分（0.1→0.3 升、0.3–0.5 为 1、0.5→0.9 降）。

判别器项（disc=True，思路同 GAN 的判别模型）：再加一个把关模型 disc（ragtruth.rm_disc 的 rm2__d1：这句错是模型自己写错的，还是事后改出来的）。
        现在的合成错句离真实错句很远（留出原文上判别器给真实错句 93% 判为真、合成的只有一成，AUC 0.95），绝对分数几乎全在低位，
        直接乘进奖励会把所有样本压扁。所以只用组内的相对高低：同一个提示的一组样本里，按判别器的分从低到高排名，
        奖励 = 阶段一的分 + w × 阶段一的分 × 排名分（比它更不像真的同伴占比）。w 和阶段二、三的难度权重一样自动调。
        判别器要跟着生成器更新：每训完一段，用新生成器的样本重训。

三个阶段训完后 Claude 抽查发现两个偏向，各加一项来纠（都默认关）：
  明显错系数（clear_ref > 0）：难度项再乘 min(1, P(明显错) / clear_ref)²。三项达标且 critic 低于 0.5 的样本里六成是轻微错
        （should → may 这类），明显错只有 18%：过门槛的样本里难度分和 P(明显错) 的相关是 −0.41，难度项实际在奖励「错得轻」。
        不用硬门槛（P(明显错) ≥ 0.5 才给分）：那样只有 15% 的任务组内有可比较的难度分，没有东西可学。
  组内多样性（div_sim > 0）：同一个任务的一组样本里，达标样本之间比「改动签名」（相对原句删掉和新加的实词），
        签名重合度 ≥ div_sim 的算同一种改法，同一种改法的 k + 1 条每条奖励乘 1 / √(1 + k)（不低于 div_floor）。
        只扣不加、只在达标样本之间比：给「不一样」加分会鼓励离谱的句子。训到阶段三，同一任务两两相似度从 0.44 升到 0.61，
        改法向几个套路收敛（add_hedge 里 91% 是加 may，generalize 里 34% 是加 many）。
  组内多样性只管得到「同一个任务里重复」，管不到「每个任务都用同一招」。再加两项跨任务的，都只扣不加、只作用在达标样本上：
  手段节点（node_base）：每条样本按 ragtruth.edit_node 归到手段节点（新加或去掉了哪一类词）。按要求的操作分别滚动统计达标样本的节点占比，
        某个节点的占比超过起点模型在这种操作上占比的 1.5 倍（起点占比不足 5% 的按 5% 算），奖励乘 √(1.5 × 起点占比 / 现占比)，不低于 0.5。
        以起点为基准而不是均匀：加限定语本来就只有一个节点，不该因为集中被罚；要罚的是套路渗进别的操作
        （要求改数字，起点一半是换成另一个数，训到后来三分之二带上了 several / at least）。
  类别分布（cls）：事后分类模型（ragtruth.rm_opc 的 opl、opc，不看要求的操作）判这条样本实际是哪一种操作。滚动统计全部达标样本的类别占比，
        某个类别的占比超过它在出题里占比的 1.5 倍，奖励按同样的公式缩小。出题是各种操作均匀的，产出的类别分布偏了，
        说明有些操作被做成了别的操作，或者被判为不是错。
  cls = 2：换成多维度的事后分类模型（ragtruth.rm_opd 的 opd，一次前向给出层、操作、动作、手段）。类别占比照旧用它判的操作；
        手段节点改用它判的「动作 / 手段」，不再用词表规则：规则认不出换实词和换语序的改法（达标样本里约两成），词表外的同义说法也认不出。
        这时 node_base 要给模型节点的起点占比（scripts/analysis/genops_classify_opd.py 在起点模型的样本上算出来）。
"""
from __future__ import annotations

from ragtruth.rm_eval import option_logprobs, probs
from train.gen_reward import NO_THINK, edit_sim
from train.gen_reward_v2 import BANNED, hardness

STOP = frozenset("the a an to of in on at for and or is are was were be been it that this with by as from s".split())


def edit_sig(original: str, sentence: str) -> frozenset:
    """改动签名：相对原句删掉的实词（-词）和新加的实词（+词）。"""
    import collections
    import re
    a, b = (collections.Counter(re.findall(r"\w+", t.lower())) for t in (original, sentence))
    return frozenset("-" + w for w in (a - b) if w not in STOP) | frozenset("+" + w for w in (b - a) if w not in STOP)


def soft_hard(p: float) -> float:
    """阶段二的难度系数：critic P 越低越高，0.9 以上为 0，0.1 以下停在 0.5。"""
    if p >= 0.9:
        return 0.0
    if p < 0.1:
        return 0.5
    return (0.9 - p) / 0.8


class OpsRewarder:
    def __init__(self, client, temps: dict[str, float], stage=1, err_gate=0.5, nat_gate=0.5, op_gate=0.5, min_sim=0.0, fail_scale=0.1,
                 ref=0.0, w_hard=0.5, w_min=0.02, tol=0.97, window=2048, obv_pen=0.5, obv_from=0.7, disc=False, refs=None, w_disc=0.5, w_fixed=False,
                 clear_ref=0.0, div_sim=0.0, div_floor=0.4, node_base=None, cls=False, cat_target=None):
        import collections
        import threading
        self.client, self.t = client, temps
        self.stage, self.err_gate, self.nat_gate, self.op_gate, self.min_sim, self.fail_scale = stage, err_gate, nat_gate, op_gate, min_sim, fail_scale
        # ref = 进入本阶段时阶段一的平均分（同一批提示上量的）；0 = 不自动调，w 固定为 w_hard
        self.ref, self.w_max, self.w_min, self.tol, self.window = ref, w_hard, w_min, tol, window
        self.obv_pen, self.obv_from, self.disc, self.w_disc = obv_pen, obv_from, disc, w_disc
        self.clear_ref, self.div_sim, self.div_floor = clear_ref, div_sim, div_floor
        # node_base = {操作: {"share": {节点: 起点模型的占比}}}；cat_target = {类别: 出题里的占比}（没给的类别按 0.02）
        self.node_base, self.cls, self.cat_target = node_base, cls, cat_target or {}
        self.node_recent = {}                                                                # 每种操作最近 300 条达标样本的节点
        self.cat_recent = collections.deque(maxlen=2048)                                     # 最近的达标样本的类别
        # 窗口 2048 条 ≈ 8 步：窗口只有 2 步（16 个任务）时，抽到哪几种操作就能让某一项的均值上下 0.05，
        # 明明没有下降也频繁触发回调（实测四分之一的检查触发），权重被按在起步值附近，难度项等于没加
        self.w = w_hard if not (ref or refs) else w_hard / 2   # 从一半起步，前面几项稳得住再放大
        self.recent, self.n_seen, self.lock = collections.deque(maxlen=window), 0, threading.Lock()
        # refs = 进入本阶段时真错 / 自然 / 操作一致三项各自的平均概率。给了就逐项看：任何一项掉下去都把难度权重调小。
        # 只看综合分不够：s2d 里自然度涨、真错率 74% → 70%，综合分没变，约束没有察觉。
        self.refs = tuple(refs) if refs else None
        self.recent3 = [collections.deque(maxlen=window) for _ in range(3)]
        self.low = ""   # 最近一次检查时掉到基准线下面的项（记进样本，方便看是哪一项在拖）
        self.lam, self.lam_max = [0.0, 0.0, 0.0], 2.0   # 真错 / 自然 / 操作一致三项的补偿权重（逐项守基准线时用）
        self.w_floor = w_hard / 2 if refs else w_min    # 逐项守基准线时，难度权重不低于起步值
        if w_fixed:   # 难度权重整段训练固定为 w_hard，不升不降；前面几项靠各自的补偿权重守
            self.w = self.w_floor = w_hard

    def state_dict(self) -> dict:
        return {"w": self.w, "recent": list(self.recent), "n_seen": self.n_seen, "lam": list(self.lam)}

    def load_state_dict(self, st: dict) -> None:
        self.w, self.n_seen = float(st.get("w", self.w)), int(st.get("n_seen", 0))
        self.recent.extend(st.get("recent", []))
        self.lam = [float(v) for v in st.get("lam", self.lam)]

    def track(self, base: float, comps: tuple = (0.0, 0.0, 0.0)) -> float:
        """记下这条样本的阶段一得分；每满半个窗口看一次：最近的平均分掉到 ref × tol 以下就把难度权重调小，否则慢慢调大。返回当前权重。"""
        with self.lock:
            self.recent.append(base)
            for q, v in zip(self.recent3, comps):
                q.append(v)
            self.n_seen += 1
            if (self.ref or self.refs) and self.n_seen % (self.window // 2) == 0 and len(self.recent) >= self.window // 2:
                low = []
                if self.ref and sum(self.recent) / len(self.recent) < self.ref * self.tol:
                    low.append("score")
                for name, q, r in zip(("real", "natural", "op"), self.recent3, self.refs or ()):
                    if sum(q) / len(q) < r * self.tol:
                        low.append(name)
                self.low = ",".join(low)
                if self.refs:
                    # 哪一项掉了，就给哪一项加补偿权重，难度权重不动。之前的做法是一掉就把难度权重调小：
                    # 阶段二里权重从 2.0 被压到 0.2–0.4 就停在那里，难度项等于被关掉，critic 那边 100 步没有任何进展。
                    # 两头都要：难度不让步，掉下去的那一项靠加它自己的权重拉回来（要的正是 critic 认不出、验证模型又确信的样本）
                    for k, name in enumerate(("real", "natural", "op")):
                        self.lam[k] = min(self.lam_max, self.lam[k] + 0.25) if name in low else max(0.0, self.lam[k] - 0.1)
                    self.w = max(self.w_floor, self.w * 0.9) if "score" in low else min(self.w_max, self.w * 1.2)
                else:
                    self.w = max(self.w_min, self.w * 0.7) if low else min(self.w_max, self.w * 1.2)
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
        if self.disc:
            from ragtruth.rm_disc import prompt as disc_prompt
            pd = self.option_p("disc", disc_prompt(row["source"], ctx, rec["idx"]), "AB", priority)
            out["p_disc"] = pd["A"] if pd else 0.0
        if self.cls == 2:
            from ragtruth.rm_opd import decode, multi_logprobs, prompt as opd_prompt
            body = {**NO_THINK["extra_body"], **({"priority": priority} if priority else {})}
            r = self.client.chat.completions.create(model="opd", messages=opd_prompt(x), max_tokens=12, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=body)
            lps = multi_logprobs(r.choices[0])
            d = decode(lps) if lps else {}
            out["c_layer"], out["c_op"], out["c_node"] = d.get("layer"), d.get("op"), (f"{d['action']}/{d['device']}" if d else None)
        elif self.cls:
            from ragtruth.rm_opc import LAYER_LETTER, LAYER_OPS, opc_prompt, opl_prompt
            pl = self.option_p("opl", opl_prompt(x), "ABCDE", priority)
            layer = {v: k for k, v in LAYER_LETTER.items()}.get(max(pl, key=pl.get), "none") if pl else None
            out["c_layer"], out["c_op"] = layer, "none" if layer == "none" else None
            if layer and layer != "none":
                ops = LAYER_OPS[layer]
                pq = self.option_p("opc", opc_prompt(x, layer), "ABCDEFGH"[:len(ops)], priority)
                out["c_op"] = ops[ord(max(pq, key=pq.get)) - 65] if pq else None
        if self.stage >= 2 or self.obv_pen:
            p1 = self.option_p("rm1", [{"role": "system", "content": RM1_SYSTEM}, {"role": "user", "content": render_input({"source": row["source"], "sentences": ctx, "idx": rec["idx"]})}], "ABC", priority)
            out["p_halluc"] = (p1["B"] + p1["C"]) if p1 else 1.0
        return out

    def score(self, text: str, row: dict) -> dict:
        s = self.raw(text, row)
        op = row["op"]
        base = {"type": op, "tactic": op, "tactic_raw": op, "op": "rewrite", "reason": "", "evidence": ""}
        if s["why"] != "ok":
            if self.stage >= 2 or self.disc:
                self.track(0.0)
            return {**base, "idx": 0, "sentence": "", "reward": 0.0, "why": s["why"], "qualified": False}
        geo = (s["p_err"] * s["p_natural"] * s["p_op"]) ** (1 / 3)
        near = s["sim"] >= self.min_sim
        gate = near and s["p_err"] >= self.err_gate and s["p_natural"] >= self.nat_gate and s["p_op"] >= self.op_gate
        pc = s["p_halluc"]
        obv = 1.0 if pc is None or pc <= self.obv_from else 1.0 - self.obv_pen * (pc - self.obv_from) / (1.0 - self.obv_from)
        r1 = (geo if near else self.fail_scale * geo) * obv   # 阶段一的分：三个阶段都原样在奖励里
        if self.stage == 1:
            r, h, q, w = r1, 0.0, gate, (self.track(r1, (s["p_err"], s["p_natural"], s["p_op"])) if self.disc else 0.0)   # 判别器的那一项在 group_post 里按组内排名加
        else:
            w = self.track(r1, (s["p_err"], s["p_natural"], s["p_op"]))
            h = (soft_hard if self.stage == 2 else hardness)(s["p_halluc"]) if gate else 0.0
            if self.clear_ref:
                h *= min(1.0, s["p_clear"] / self.clear_ref) ** 2
            # 难度项直接加，不再乘阶段一的分：过了三道门槛之后，能拿多少难度分只看 critic。
            # 乘法时难度项和阶段一的分天然相关，权重小的时候训练继续往「三项更高」走，而那三项和 critic 同向——
            # 实测权重 0.25 跑 27 步，达标样本里 critic 拿不准的从 24% 降到 17%，样本反而更好认了
            lam = list(self.lam)
            r, q = r1 + w * h + lam[0] * s["p_err"] + lam[1] * s["p_natural"] + lam[2] * s["p_op"], bool(gate and h > 0)
        return {**base, **s, "p_halluc": s["p_halluc"] if s["p_halluc"] is not None else 0.0, "gate": gate, "geo": geo, "obv": obv, "r1": r1, "w_hard": w, "low": self.low, "lam": list(self.lam), "hardness": h, "reward": r, "qualified": bool(q),
                "p_correct": s["p_err"], "hard": h, "real": s["p_natural"], "div": 1.0, "div_k": 0}

    def group_post(self, scores: list[dict]) -> None:
        """判别器项：同一个提示的一组里，格式合格的样本按判别器的分排名，奖励加上 w × 阶段一的分 × 排名分。
        组内多样性：达标样本里同一种改法的分摊奖励（在判别器项之后，对整条奖励按比例缩小）。"""
        if self.disc:
            self.disc_post(scores)
        P = [x for x in scores if x.get("why") == "ok" and x.get("gate")]
        for x in P:
            x["reward_raw"], x["div"] = x["reward"], 1.0
        if self.div_sim:
            S = [edit_sig(x["original"], x["sentence"]) for x in P]
            for i, x in enumerate(P):
                k = sum(1 for j in range(len(P)) if j != i and (len(S[i] & S[j]) / len(S[i] | S[j]) if (S[i] | S[j]) else 1.0) >= self.div_sim)
                x["div_k"], x["div"] = k, (1 + k) ** -0.5
        if self.node_base or self.cls:
            import collections
            from ragtruth.edit_node import FALLBACK, node
            from ragtruth.error_ops import OPS

            def shrink(p: float, q: float) -> float:
                return 1.0 if p <= 1.5 * q else max(0.5, (1.5 * q / p) ** 0.5)

            with self.lock:
                for x in P:
                    if self.node_base:
                        x["node"] = (x.get("c_node") or "none/none") if self.cls == 2 else node(x["original"], x["sentence"])
                        self.node_recent.setdefault(x["type"], collections.deque(maxlen=300)).append(x["node"])   # 兜底节点也记：它是分母的一部分
                    if self.cls and x.get("c_op"):
                        self.cat_recent.append(x["c_op"])
                for x in P:
                    # 兜底节点（规则说不出是哪类词）不扣；挂接层不用这一项：这一层改的是实词和语序，规则大半认不出，套路交给事后分类看
                    if self.node_base and (self.cls == 2 or (x["node"] not in FALLBACK and OPS[x["type"]]["layer"] != "binding")):
                        q = self.node_recent[x["type"]]
                        if len(q) >= 30:   # 这种操作的达标样本攒够了再算占比
                            base = max(0.05, self.node_base.get(x["type"], {}).get("share", {}).get(x["node"], 0.0))
                            x["node_mult"] = shrink(sum(n == x["node"] for n in q) / len(q), base)
                    if self.cls and x.get("c_op") and len(self.cat_recent) >= 300:
                        x["cat_mult"] = shrink(sum(c == x["c_op"] for c in self.cat_recent) / len(self.cat_recent), self.cat_target.get(x["c_op"], 0.02))
        # 几项多样性的系数相乘，合起来不低于 div_floor：扣得再狠，达标样本的分也不该掉到没达标的样本下面，否则是在把模型往不达标推
        for x in P:
            x["div_total"] = max(self.div_floor, x["div"] * x.get("node_mult", 1.0) * x.get("cat_mult", 1.0))
            x["reward"] = x["reward_raw"] * x["div_total"]

    def disc_post(self, scores: list[dict]) -> None:
        cand = [x for x in scores if x.get("why") == "ok" and "p_disc" in x]
        ps = [x["p_disc"] for x in cand]
        for x in cand:
            lo = sum(q < x["p_disc"] for q in ps) + 0.5 * (sum(q == x["p_disc"] for q in ps) - 1)
            x["disc_rank"] = lo / (len(cand) - 1) if len(cand) > 1 else 0.5
            # 阶段一里判别器项用自动调的权重；阶段二、三里那个权重给了难度项，判别器项用固定的 w_disc
            wd = x["w_hard"] if self.stage == 1 else self.w_disc
            x["reward"] = x["reward"] + wd * x["r1"] * x["disc_rank"]
