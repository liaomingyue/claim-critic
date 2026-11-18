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
  和真实错误的分布比（real_target，要 cls = 2）：上面两项拿起点模型的分布当目标，而起点模型本身就在用套路（加限定语 92% 是 may），
        查训练样本发现用了套路词的样本几乎没被罚，换了写法的反而因为被判成别的操作罚得更重。给了 real_target 就换成这一套，上面两项不再用：
        节点：分类模型不取最可能的一格，整个概率分布都用（一半的准确率下，硬判的一格有一半是错的）。目标 = 同一个分类模型在真实错误句上给出的平均分布
              （两边用同一把尺子，分类模型的偏差大体抵消），实际 = 这种操作最近 300 条达标样本的平均分布。
              系数 = Σ 这条样本在各格的概率 × (目标占比 / 实际占比)，夹在 [0.5, div_cap]：落在占比偏高的格子里的罚，落在偏低的格子里的加分。
        词面：节点比套路粗（many、most、some 是同一格）。同一种操作里统计「新加进句子的词」的占比，
              超过它在真实错误里占比的 1.5 倍（不足 10% 的按 10% 算）就乘 允许占比 / 实际占比，不低于 0.3。只罚不奖：给少见的词加分会鼓励怪词。
        真实错误不足 30 对的操作（加限定语只有 3 对）两项都不设目标。目标文件由 scripts/analysis/genops_classify_opd.py real 生成。
  第二版（real_v = 2）：上一版把 many、because、not 压下去了，留下两个问题，这一版各补一项，再把词面项加重：
        近义词并组：改数字时 about、around、several 轮着用，每个词单看都在门槛附近。同一组的词（WORD_GROUPS）合起来再算一次占比，
              超过真实错误里这一组占比的 1.5 倍（不足 15% 的按 15% 算）就罚，和单个词的系数取小的。
        操作系数：没被罚的样本里操作做对的比例明显更低（扩大范围 50% 对 82%），躲开套路词的办法之一是改去做别的操作。
              乘一项 0.75 + 0.5 × 分类模型给「要求的操作」的概率（用概率，不用硬判：硬判只有一半准；以 1 为中心，免得达标样本整体的分被压到没达标的样本下面）；节点的加分部分也按这个概率打折。
        没有目标的操作：新加的词按固定上限算（单个词不超过 30%），不设节点目标。
        单个词的门槛 10% → 8%；词面系数的下限 0.3 → 0.2，不过几项相乘后仍不低于 div_floor（0.4），实际多罚的有限。
  按占比给的稀有奖励（share_w = (组内权重, 全局权重)，要 cls = 2）：只加不扣、只给达标样本，而且只在操作一致的样本之间算
        （分类模型给「要求的操作」的概率 ≥ share_pop，默认 0.5；其余样本不进分布、不加分；过了的按这个概率加权：进分布按它算，加分也按它打折）。
        类别 = 分类模型判的手段节点（动作 / 手段），用整个概率分布：两条样本算不算同一类按两个分布的余弦相似度算，硬判时就是 0 或 1。
        不用它判的操作当类别，也不让操作不一致的样本参与：同一个任务要求的是同一种操作，按操作算稀有等于奖励做成别的操作
        （起点模型的样本上模拟：组内 1 − 占比，做对的平均 0.10，做成别的操作的 0.42）。
        组内：1 − 所在节点在这一组里的占比。一组的平均值 = 这一组节点分布的多样性（1 − Σ 占比²）；整组的多样性对组内每条样本是同一个数，
              减掉组内均值后没有信号，所以按每条样本的贡献分下去。只有一条合条件的样本时是 0。
        全局：有真实目标的操作由上面「和真实错误的分布比」管（高于真实占比的罚、低于的加分）；这一项只管没有真实目标的操作：
              每种操作一份累计的节点分布，起点 = 起点模型在这种操作上的节点占比（node_base，折成 share_n0 条样本的量），
              这一项 = 1 − 这条样本的节点在累计分布里的占比。没给 real_target 时所有操作都用这一项。
        奖励乘 (1 + 组内权重 × 组内项) × (1 + 全局权重 × 全局项)。开了这一项，旧的节点 / 类别占比系数（以起点为目标的那两项）不再算。
"""
from __future__ import annotations

from ragtruth.rm_eval import option_logprobs, probs
from train.gen_reward import NO_THINK, edit_sim
from train.gen_reward_v2 import BANNED, hardness

STOP = frozenset("the a an to of in on at for and or is are was were be been it that this with by as from s".split())
# 词面占比不看的词：代词和助动词是改写时顺带出现的，不是套路。情态词（may、can、might）留着：它们正是套路
FUNC = frozenset("have has had he she they his her their them you we i its which who what there here do does did".split())
# 近义词组（real_v = 2）：同一组的词合起来算占比，防止一个词被压下去、换一个意思相同的词顶上
WORD_GROUPS = {
    "approx": "about around approximately roughly nearly almost over under least up",
    "vague_quantity": "many several few some most numerous various multiple countless dozens hundreds thousands",
    "cause": "because since due therefore thus hence so led leading caused causing resulting",
    "hedge": "may might could possibly potentially perhaps likely probably reportedly allegedly apparently",
    "contrast": "not no never but yet still however although though",
}
WORD_GROUP = {w: g for g, ws in WORD_GROUPS.items() for w in ws.split()}


def word_groups(words) -> frozenset:
    return frozenset(WORD_GROUP[w] for w in words if w in WORD_GROUP)


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
                 clear_ref=0.0, div_sim=0.0, div_floor=0.4, node_base=None, cls=False, cat_target=None, div_rare=0.0, div_cap=2.0, real_target=None, real_v=1, share_w=(0.0, 0.0), share_n0=200.0, share_pop=0.5):
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
        # div_rare > 0（要 cls = 2）：占比系数改成 (目标占比 / 实际占比) ** div_rare，占比高的重罚、占比低的加分。分两种情况：
        #   分类模型判的操作 = 要求的操作：按节点算，目标是起点模型在这种操作上的节点占比，系数夹在 [0.5, div_cap]，可以大于 1；
        #   判成了别的操作：按它落进的类别算，目标是出题占比，系数夹在 [0.4, 1]，只罚不奖。
        # 稀有的加分只给做对了的样本：在留出样本上模拟过不分情况的版本，稀有的格子大半是做成了别的操作的样本，
        # 做错的平均系数（0.82）反而高过做对的（0.61），等于奖励跑偏
        self.div_rare, self.div_cap = div_rare, div_cap
        # real_target = {操作: {"n": 真实错误的对数, "node": {节点: 分类模型在真实错误上的平均概率}, "word": {新加的词: 占比}}}
        self.real_target, self.real_v = real_target, real_v
        self.share_w, self.share_n0, self.share_pop, self.share_cnt = tuple(share_w), share_n0, share_pop, {}   # share_cnt = {操作: {节点: 累计的量}}
        self.group_recent = {}                                                               # real_v = 2：每种操作最近 300 条达标样本用到的近义词组
        self.soft_recent, self.word_recent = {}, {}                                          # 每种操作最近 300 条达标样本的节点概率分布、新加的词
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
        return {"w": self.w, "recent": list(self.recent), "n_seen": self.n_seen, "lam": list(self.lam), "share_cnt": {k: dict(v) for k, v in self.share_cnt.items()}}

    def load_state_dict(self, st: dict) -> None:
        self.w, self.n_seen = float(st.get("w", self.w)), int(st.get("n_seen", 0))
        self.recent.extend(st.get("recent", []))
        self.lam = [float(v) for v in st.get("lam", self.lam)]
        self.share_cnt = {k: dict(v) for k, v in st.get("share_cnt", {}).items()}

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
            if (self.real_target or any(self.share_w)) and lps:
                from ragtruth.rm_opd import soft_nodes
                out["c_soft"] = soft_nodes(lps)
                from ragtruth.rm_opd import op_prob
                out["c_pop"] = op_prob(lps, row["op"])
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
        if self.real_target:
            self.real_post(P)
        elif (self.node_base or self.cls) and not any(self.share_w):
            import collections
            from ragtruth.edit_node import FALLBACK, node
            from ragtruth.error_ops import OPS

            def shrink(p: float, q: float) -> float:
                return 1.0 if p <= 1.5 * q else max(0.5, (1.5 * q / p) ** 0.5)

            def rare(p: float, q: float, lo: float, hi: float) -> float:
                return min(hi, max(lo, (q / max(p, 1e-6)) ** self.div_rare))

            def node_target(op: str, n: str) -> float:
                return max(0.05, self.node_base.get(op, {}).get("share", {}).get(n, 0.0))

            right = lambda x: x.get("c_op") == x["type"]

            with self.lock:
                for x in P:
                    if self.node_base:
                        x["node"] = (x.get("c_node") or "none/none") if self.cls == 2 else node(x["original"], x["sentence"])
                        if not self.div_rare or right(x):   # 双向的版本只统计做对了的样本
                            self.node_recent.setdefault(x["type"], collections.deque(maxlen=300)).append(x["node"])   # 兜底节点也记：它是分母的一部分
                    if self.cls and x.get("c_op"):
                        self.cat_recent.append(x["c_op"])
                for x in P if self.div_rare else []:
                    if right(x) and self.node_base and len(self.node_recent.get(x["type"], ())) >= 30:
                        q = self.node_recent[x["type"]]
                        x["node_mult"] = rare(sum(n == x["node"] for n in q) / len(q), node_target(x["type"], x["node"]), 0.5, self.div_cap)
                    elif not right(x) and x.get("c_op") and len(self.cat_recent) >= 300:
                        x["cat_mult"] = rare(sum(c == x["c_op"] for c in self.cat_recent) / len(self.cat_recent), self.cat_target.get(x["c_op"], 0.02), 0.4, 1.0)
                for x in [] if self.div_rare else P:
                    # 兜底节点（规则说不出是哪类词）不扣；挂接层不用这一项：这一层改的是实词和语序，规则大半认不出，套路交给事后分类看
                    if self.node_base and (self.cls == 2 or (x["node"] not in FALLBACK and OPS[x["type"]]["layer"] != "binding")):
                        q = self.node_recent[x["type"]]
                        if len(q) >= 30:   # 这种操作的达标样本攒够了再算占比
                            x["node_mult"] = shrink(sum(n == x["node"] for n in q) / len(q), node_target(x["type"], x["node"]))
                    if self.cls and x.get("c_op") and len(self.cat_recent) >= 300:
                        x["cat_mult"] = shrink(sum(c == x["c_op"] for c in self.cat_recent) / len(self.cat_recent), self.cat_target.get(x["c_op"], 0.02))
        if any(self.share_w):
            self.share_post(P)
        # 几项多样性的系数相乘，合起来不低于 div_floor：扣得再狠，达标样本的分也不该掉到没达标的样本下面，否则是在把模型往不达标推
        for x in P:
            x["div_total"] = min(self.div_cap if (self.div_rare or self.real_target or any(self.share_w)) else 1.0, max(self.div_floor, x["div"] * x.get("node_mult", 1.0) * x.get("cat_mult", 1.0) * x.get("word_mult", 1.0) * x.get("op_mult", 1.0) * x.get("share_mult", 1.0)))
            x["reward"] = x["reward_raw"] * x["div_total"]

    def share_post(self, P: list[dict], min_n: float = 30.0) -> None:
        """按占比给的稀有奖励。P = 同一个任务的达标样本；只在「做的是要求的操作」的概率 ≥ share_pop 的样本之间算，其余的不参与也不加分。
        组内项 = 1 − 所在节点在这一组里的占比。一组的平均值就是这一组节点分布的多样性（1 − Σ 占比²）：
        整组的多样性是一个数，组内减均值后对每条样本都一样，学不到东西，所以按每条样本对它的贡献分下去。
        全局项 = 1 − 所在节点在这种操作累计分布里的占比；给了 real_target 时只用在没有真实目标的操作上（有目标的由 real_post 和真实分布比）。"""
        vec = lambda x: x.get("c_soft") or {x.get("c_node") or "none/none": 1.0}
        pop = lambda x: x["c_pop"] if x.get("c_pop") is not None else float(x.get("c_op") == x["type"])
        dot = lambda a, b: sum(v * b.get(k, 0.0) for k, v in a.items())
        E = [x for x in P if pop(x) >= self.share_pop]
        V = [vec(x) for x in E]
        N = [dot(v, v) ** 0.5 or 1.0 for v in V]
        W = [pop(x) for x in E]   # 操作一致的概率当权重：进分布按它算，加分也按它打折
        wg, wa = self.share_w
        with self.lock:
            for x in P:
                x["grp_div"], x["all_div"] = 0.0, 0.0
            for i, x in enumerate(E):
                x["grp_div"] = (1.0 - sum(W[j] * dot(V[i], V[j]) / (N[i] * N[j]) for j in range(len(E))) / sum(W)) * W[i]
                op = x["type"]
                tg = (self.real_target or {}).get(op)
                if not (tg and tg["n"] >= min_n):
                    if op not in self.share_cnt:   # 起点：起点模型在这种操作上的节点占比，折成 share_n0 条样本
                        base = (self.node_base or {}).get(op, {}).get("share", {})
                        self.share_cnt[op] = {k: v * self.share_n0 for k, v in base.items()}
                    c = self.share_cnt[op]
                    n = sum(c.values())
                    x["all_div"] = (1.0 - sum(v * c.get(k, 0.0) for k, v in V[i].items()) / n) * W[i] if n >= min_n else 0.0
                x["share_mult"] = (1.0 + wg * x["grp_div"]) * (1.0 + wa * x["all_div"])
            for x, v, w in zip(E, V, W):   # 这一组算完再记进累计分布
                if x["type"] in self.share_cnt:
                    c = self.share_cnt[x["type"]]
                    for k, q in v.items():
                        c[k] = c.get(k, 0.0) + w * q

    def real_post(self, P: list[dict], eps: float = 0.02, min_n: int = 30) -> None:
        """和真实错误的分布比：节点按概率分布算（node_mult，可以大于 1），新加的词按占比算（word_mult，只罚）。
        real_v = 2 时再加近义词组的占比、操作系数（op_mult），没有目标的操作按固定上限算词面。"""
        import collections
        v2 = self.real_v >= 2
        floor, thr = (0.2, 0.08) if v2 else (0.3, 0.10)
        with self.lock:
            todo = []
            for x in P:
                tg = self.real_target.get(x["type"])
                has = bool(tg) and tg["n"] >= min_n
                if not has and not v2:
                    continue
                x["add_words"] = sorted(t[1:] for t in edit_sig(x["original"], x["sentence"]) if t[0] == "+" and t[1:] not in FUNC)
                self.word_recent.setdefault(x["type"], collections.deque(maxlen=300)).append(frozenset(x["add_words"]))
                if v2:
                    self.group_recent.setdefault(x["type"], collections.deque(maxlen=300)).append(word_groups(x["add_words"]))
                if has and x.get("c_soft"):
                    self.soft_recent.setdefault(x["type"], collections.deque(maxlen=300)).append(x["c_soft"])
                todo.append((x, tg if has else None))
            share = {}   # 每种操作这一批只算一次平均分布
            for x, tg in todo:
                op = x["type"]
                pr = x.get("c_pop") if v2 else None
                q = self.soft_recent.get(op, ())
                if tg and x.get("c_soft") and len(q) >= min_n:
                    if op not in share:
                        acc = collections.Counter()
                        for s in q:
                            acc.update(s)
                        share[op] = {k: v / len(q) for k, v in acc.items()}
                    m = sum(w * min(4.0, max(0.25, (tg["node"].get(k, 0.0) + eps) / (share[op].get(k, 0.0) + eps))) for k, w in x["c_soft"].items())
                    m = min(self.div_cap, max(0.5, m))
                    if pr is not None and m > 1.0:   # 加分只按「做的是要求的操作」的概率给
                        m = 1.0 + (m - 1.0) * pr
                    x["node_mult"] = m
                wq = self.word_recent[op]
                if len(wq) >= min_n and x["add_words"]:
                    wm, worst = 1.0, ""
                    for w in x["add_words"]:
                        allow = max(1.5 * tg["word"].get(w, 0.0), thr) if tg else 0.30
                        p = sum(w in s for s in wq) / len(wq)
                        if p > allow and allow / p < wm:
                            wm, worst = max(floor, allow / p), w
                    if v2 and tg:
                        gq = self.group_recent[op]
                        for g in word_groups(x["add_words"]):
                            allow, p = max(1.5 * tg.get("group", {}).get(g, 0.0), 0.15), sum(g in s for s in gq) / len(gq)
                            if p > allow and allow / p < wm:
                                wm, worst = max(floor, allow / p), "group:" + g
                    x["word_mult"], x["word_hit"] = wm, worst
                if pr is not None:
                    x["op_mult"] = 0.75 + 0.5 * pr

    def disc_post(self, scores: list[dict]) -> None:
        cand = [x for x in scores if x.get("why") == "ok" and "p_disc" in x]
        ps = [x["p_disc"] for x in cand]
        for x in cand:
            lo = sum(q < x["p_disc"] for q in ps) + 0.5 * (sum(q == x["p_disc"] for q in ps) - 1)
            x["disc_rank"] = lo / (len(cand) - 1) if len(cand) > 1 else 0.5
            # 阶段一里判别器项用自动调的权重；阶段二、三里那个权重给了难度项，判别器项用固定的 w_disc
            wd = x["w_hard"] if self.stage == 1 else self.w_disc
            x["reward"] = x["reward"] + wd * x["r1"] * x["disc_rank"]
