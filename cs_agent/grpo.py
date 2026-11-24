"""客服 Agent 的 GRPO（τ² retail 训练任务，策略从 Qwen3.5-9B 原版起步）。

两台机器：
  推理机：一个 vLLM 服务，策略 LoRA（每步热加载新版本）+ 9B 原版当用户模拟器（model=base，关闭思考）。
  训练机（本程序）：一个 9B 底座挂四套参数——policy（训练的 LoRA，起点 B 全零 = 原版）、参考模型（关掉 LoRA 的原版）、
    cstep（过程 critic，cs_agent.critic --task step）、cout（全流程 reward，--task outcome）。
采样（线程，只发 HTTP 请求）和训练（GPU）并行，采样是流式的：并发槽一直填满，同一任务的 n 条到齐就整组交给训练，
训练凑够 tasks_per_step 组训一步；最早一条落后超过 max_stale 个版本的组丢掉。单条对话有墙钟上限（ep_timeout），超时按没办完算。重要性比值用采样时 vLLM 返回的 logprob，延迟一个版本是 PPO 截断管得住的范围。

奖励（和用户对过的设计）：
  全局（结果监督，默认 --outcome_source model）：R = 结果模型的 P(正确) = p(GOOD) / (p(GOOD) + p(FAIL))（三分类模型去掉 FLAWED 重新归一）；
        执行了 S1 严重违规 R = 0。真实数据库结果只记录（每步对照模型给的分，看结果模型有没有被钻空子）。
        过程好坏只由过程 critic 管（步骤优势）。--outcome_source db：R = 真实数据库比对（对照用）。A_glob = 同一任务 G 条采样组内标准化。
  步骤：分值 OK +1 / ERR −2 / AFTER_ERR −1 / FIX +3 / TRY +0.5（cstep 的类别概率）。
        默认去底噪（--step_mode floor，同 RAG 句子层）：每个非 OK 类别减掉同组全部步骤的平均概率，只有超出平均的部分计分，
        A_step = clip(Σ_c (分值_c / 3) × max(0, p_c − b_c) / σ_c, ±3)；处在平均水平的步骤优势为 0。
        底噪 b_c 不超过 critic 在留出集正常步骤上的平均概率（--floor_caps）；FIX / TRY 只在更早有错可纠时计分
        （更早某步 ERR / AFTER_ERR 超出底噪 ≥ 1σ，或工具报过错）。
  每个 token 的优势 = A_glob + β_step · A_step(所在步骤)。
损失：PPO 截断（clip）+ k3 KL（相对原版，系数 kl）；只对策略生成的 token 算（lmask），除以整批策略 token 数。
显存：反向只算前 max_len 个 token（9B 单卡 16k 放得下、20k OOM），更长轨迹靠后的步骤不参与这一步的更新（打分时看全长）。

  python -m cs_agent.grpo --base <9B> --init <B 全零 LoRA> --critic_step <目录> --critic_outcome <目录> \
      --policy_url <推理机 /v1> --user_url <推理机 /v1> --out <目录> [--steps 60]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cs_agent  # noqa: F401
from cs_agent.step_cls import CLASSES as STEP_CLASSES
from cs_agent.step_cls import REWARD as STEP_REWARD


# ---------------------------------------------------------------- 采样
class Producer(threading.Thread):
    """后台线程，流式采样（只发 HTTP 请求，不碰 GPU）：并发槽一直填满，一条对话结束就补下一条，不等整批。
    任务按组排（每个任务 n 条），一组的 n 条到齐就整组放进队列；训练凑够 tasks_per_step 组训一步。
    整批同步时，一步的耗时由 64 条里最慢的那条决定（实测中位数 137 秒、最长 85 分钟），并发槽大半时间空着。
    每条对话记下开始时的策略版本，用那个版本采到底（重要性比值按它的 logprob 算）；领先训练最多 ahead_groups 组。"""

    def __init__(self, a, tok, tasks, servers, version_ref):
        super().__init__(daemon=True)
        self.a, self.tok, self.tasks, self.servers, self.version_ref = a, tok, tasks, servers, version_ref
        self.active = [0] * len(servers)   # 每台推理机上正在跑的对话数：新对话派给最闲的那台
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.order: list = []
        self.pos = 0
        self.epoch = 0
        self.stop = False
        self.err = None
        self.consumed = 0   # 训练取走的组数（run 里更新）

    def next_task(self):
        if self.pos >= len(self.order):
            self.order = list(self.tasks)
            random.Random(self.a.seed + self.epoch).shuffle(self.order)
            self.epoch += 1
            self.pos = 0
        self.pos += 1
        return self.order[self.pos - 1]

    def one(self, gid: int, t, j: int) -> dict:
        from cs_agent import rollout as R
        ver = self.version_ref[0]
        with self.lock:   # 一条对话的策略和模拟器都在同一台推理机上（前缀缓存留在一台机器里）
            k = min(range(len(self.servers)), key=lambda i: self.active[i])
            self.active[k] += 1
        try:
            return self._one(gid, t, j, ver, k)
        finally:
            with self.lock:
                self.active[k] -= 1

    def _one(self, gid: int, t, j: int, ver: int, k: int) -> dict:
        from cs_agent import rollout as R
        sample = R.vllm_sampler(self.servers[k], f"policy_v{ver}", self.a.temperature)
        user_args = {"api_base": self.servers[k].url, "api_key": self.a.user_key, "temperature": 0.0,
                     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        t0, err = time.time(), ""
        for attempt in range(2):
            try:
                ep = R.run_episode(self.tok, t, sample, user_llm=f"openai/{self.a.user_model}",
                                   user_args={**user_args, "seed": gid * 100 + j * 10 + attempt},
                                   gen_max_tokens=self.a.gen_max_tokens, max_ctx=self.a.max_ctx,
                                   deadline=t0 + self.a.ep_timeout if self.a.ep_timeout else 0.0)
                ep.update(trial=j, prompt_id=f"{gid}:{t.id}", version=ver, wall=round(time.time() - t0, 1), node=k)
                return ep
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:200]
        return {"task_id": t.id, "prompt_id": f"{gid}:{t.id}", "trial": j, "error": err, "version": ver}

    def run(self) -> None:
        from concurrent.futures import FIRST_COMPLETED, wait
        try:
            groups, pending, gid, started = {}, {}, 0, collections.deque()
            cap = self.a.concurrency * len(self.servers)
            with ThreadPoolExecutor(cap) as ex:
                while not self.stop:
                    # 补满并发槽：按组顺序发，一组 n 条；领先训练太多组就先停一停
                    while len(pending) < cap and (gid - self.consumed) < self.a.ahead_groups:
                        if not started:
                            t = self.next_task()
                            groups[gid] = {"task": t, "eps": [], "t0": time.time()}
                            started.extend((gid, t, j) for j in range(self.a.n))
                            gid += 1
                        g, t, j = started.popleft()
                        pending[ex.submit(self.one, g, t, j)] = g
                    if not pending:
                        time.sleep(1)
                        continue
                    done, _ = wait(list(pending), timeout=5, return_when=FIRST_COMPLETED)
                    for f in done:
                        g = pending.pop(f)
                        groups[g]["eps"].append(f.result())
                        if len(groups[g]["eps"]) == self.a.n:
                            G = groups.pop(g)
                            self.q.put({"gid": g, "task_id": G["task"].id, "episodes": G["eps"], "t_gen": time.time() - G["t0"],
                                        "version": min(e.get("version", 0) for e in G["eps"])})
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)
            raise


# ---------------------------------------------------------------- 模型
def load_models(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, a.init, adapter_name="policy", is_trainable=True)
    model.load_adapter(a.critic_step, adapter_name="cstep", is_trainable=False)
    if a.critic_outcome:
        model.load_adapter(a.critic_outcome, adapter_name="cout", is_trainable=False)
    H = base.config.get_text_config().hidden_size if hasattr(base.config, "get_text_config") else base.config.hidden_size
    heads = {}
    a.out_classes = []
    if a.critic_outcome:   # 类别清单：训完的有 manifest；epoch 中间的检查点没有，按分类头的输出维度认（3 = GOOD/FLAWED/FAIL，2 = GOOD/FLAWED）
        mf = Path(a.critic_outcome) / "critic_manifest.json"
        if mf.exists():
            a.out_classes = json.loads(mf.read_text())["classes"]
        else:
            n_out = torch.load(Path(a.critic_outcome) / "head.pt", map_location="cpu")["weight"].shape[0]
            a.out_classes = {3: ["GOOD", "FLAWED", "FAIL"], 2: ["GOOD", "FLAWED"]}[n_out]
    for name, d, n in (("cstep", a.critic_step, len(STEP_CLASSES)),) + ((("cout", a.critic_outcome, len(a.out_classes)),) if a.critic_outcome else ()):
        h = torch.nn.Linear(H, n).cuda().float()
        h.load_state_dict(torch.load(Path(d) / "head.pt", map_location="cuda"))
        h.requires_grad_(False)
        heads[name] = h
    use_policy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    return tok, model, heads


def policy_params(model):
    return [p for n, p in model.named_parameters() if ".policy." in n and "lora_" in n]


def use_policy(model) -> None:
    model.set_adapter("policy")
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(".policy." in n)


# ---------------------------------------------------------------- 系统提示共享（train.prefix_share）
# 所有轨迹开头的系统提示（零售政策 + 工具定义，5,066 token）完全相同，约占整条轨迹的四成。
# 参考模型、两个奖励模型参数冻结：系统提示的状态开训时算一次，整个训练复用；策略模型每步算一次（带梯度），
# 这一步的全部轨迹接在后面，最后对系统提示统一反向一次（数学上和整段拼接相同，见 train.prefix_share）。
class PrefixCache:
    def __init__(self, prompt_ids: list[int]):
        self.pids, self.frozen = list(prompt_ids), {}

    def frozen_prefix(self, model, name: str):
        """name：ref（关掉 LoRA 的原版）/ cstep / cout。"""
        import torch
        from train.prefix_share import prompt_pass
        if name not in self.frozen:
            with torch.no_grad():
                if name == "ref":
                    with model.disable_adapter():
                        self.frozen[name] = prompt_pass(model, self.pids, grad=False)
                else:
                    model.set_adapter(name)
                    self.frozen[name] = prompt_pass(model, self.pids, grad=False)
                    use_policy(model)
        return self.frozen[name]


def shared_hidden(model, prefix: list, comp: list[int]):
    """接在系统提示状态后面算 completion，返回 [C+1, H] 的归一化隐状态：第 0 行是系统提示最后一个位置，第 t 行是 completion 第 t-1 个 token。
    所以：预测绝对位置 q 的 token 用第 q-P 行；绝对位置 p 的 token 自己的输出（critic 读分）是第 p-P+1 行。"""
    import torch
    from torch.utils.checkpoint import checkpoint
    from train.prefix_share import KEYS, _Shim, _rope, parts  # noqa: F401
    text, _ = parts(model)
    dev = text.embed_tokens.weight.device
    comp = list(comp) if len(comp) >= 2 else list(comp) + [comp[-1]]   # 至少 2 个 token（单 token 会走线性注意力的原地更新分支）
    C = len(comp)
    ids = torch.tensor([comp], device=dev)
    h_last, states = prefix[0], prefix[1:]
    P = states[2 * text.config.layer_types.index("full_attention")].shape[-2]
    attn = (torch.arange(P + C, device=dev)[None, :] <= P + torch.arange(C, device=dev)[:, None])[None, None]
    h = text.embed_tokens(ids)
    pe, tpos = _rope(text, h, P)
    grad = torch.is_grad_enabled()
    si = 0
    for i, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
        kind = text.config.layer_types[i]
        a_, b_ = states[si], states[si + 1]
        si += 2

        def f(h_, a_, b_, i=i, layer=layer, kind=kind):
            if kind == "full_attention":
                pre, m = {"k": a_, "v": b_}, attn
            else:
                pre, m = {"conv": a_, "rec": b_.contiguous()}, None
            return layer.forward(h_, position_embeddings=pe, attention_mask=m, position_ids=tpos, past_key_values=_Shim(i, pre))
        h = checkpoint(f, h, a_, b_, use_reentrant=False) if grad else f(h, a_, b_)
    return text.norm(torch.cat([h_last, h], 1))[0]


def _lp_rows(model, hs, tgt, grad: bool):
    import torch
    from torch.utils.checkpoint import checkpoint
    head = model.base_model.model.lm_head

    def lp_chunk(hc, tc):
        lg = head(hc).float()
        return lg.gather(1, tc[:, None])[:, 0] - torch.logsumexp(lg, -1)
    out = [checkpoint(lp_chunk, hs[c:c + 512], tgt[c:c + 512], use_reentrant=False) if grad else lp_chunk(hs[c:c + 512], tgt[c:c + 512])
           for c in range(0, len(tgt), 512)]
    return torch.cat(out) if out else torch.zeros(0, device="cuda")


def shared_logprobs(model, prefix, ids: list[int], P: int, pos: list[int], grad: bool):
    """绝对位置 pos 上 token 的 logprob（pos 都 ≥ P），系统提示用共享状态。"""
    import torch
    hs = shared_hidden(model, prefix, ids[P:max(pos) + 1])
    return _lp_rows(model, hs[torch.tensor([q - P for q in pos], device="cuda")], torch.tensor([ids[q] for q in pos], device="cuda"), grad)


def score_episode(model, heads, ep: dict, cache=None) -> tuple[list[list[float]], list[float]]:
    """cstep：每个客服步骤结尾的 5 类概率；cout：最后一个 token 的 3 类概率。都看全长（不反向，显存够）。"""
    import torch
    inner = model.base_model.model.model
    ids = ep["prompt_ids"] + ep["ids"]
    P = len(ep["prompt_ids"])
    pos = [P + t["b"] - 1 for t in ep["turns"]]
    if cache is not None and ep["prompt_ids"] == cache.pids:
        with torch.no_grad():
            model.set_adapter("cstep")
            hs = shared_hidden(model, cache.frozen_prefix(model, "cstep"), ids[P:])
            ps = torch.softmax(heads["cstep"](hs[torch.tensor([q - P + 1 for q in pos], device="cuda")].float()), -1).cpu().tolist() if pos else []
            del hs
            po = []
            if "cout" in heads:
                model.set_adapter("cout")
                hs = shared_hidden(model, cache.frozen_prefix(model, "cout"), ids[P:])
                po = torch.softmax(heads["cout"](hs[-1:].float()), -1)[0].cpu().tolist()
                del hs
        use_policy(model)
        return ps, po
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        model.set_adapter("cstep")
        h = inner(input_ids=x).last_hidden_state[0]
        ps = torch.softmax(heads["cstep"](h[torch.tensor(pos, device="cuda")].float()), -1).cpu().tolist() if pos else []
        del h
        po = []
        if "cout" in heads:   # 全流程 reward 只做监控（λ = 0 时不进奖励）
            model.set_adapter("cout")
            h = inner(input_ids=x).last_hidden_state[0]
            po = torch.softmax(heads["cout"](h[-1:].float()), -1)[0].cpu().tolist()
            del h
    use_policy(model)
    return ps, po


# ---------------------------------------------------------------- 奖励
def turn_tool_errors(ep: dict) -> list[bool]:
    """每个客服步骤之后紧跟的工具返回里有没有报错（报错说明这一步有问题，之后的修正算纠错）。"""
    ms = ep["messages"]
    out = []
    for t in ep["turns"]:
        i, err = t["msg_idx"] + 1, False
        while i < len(ms) and ms[i].get("role") == "tool":
            err = err or bool(ms[i].get("error"))
            i += 1
        out.append(err)
    return out


def caps_from_scores(path: str) -> dict:
    """底噪上限：critic 在留出集（cs_agent.critic eval 的 scores.jsonl，split=test）真实为 OK 的步骤上，各类别的平均概率。"""
    xs = collections.defaultdict(list)
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("split") == "test" and r["gold"] == "OK":
            for c, v in r["probs"].items():
                xs[c].append(v)
    return {c: round(sum(v) / len(v), 4) for c, v in xs.items() if c != "OK" and v}


def rewards(batch: dict, model, heads, a, cache=None) -> list[dict]:
    from cs_agent.rules import check
    samples = []
    for ep in batch["episodes"]:
        if ep.get("error") or not ep.get("turns"):
            continue
        ps, po = score_episode(model, heads, ep, cache)
        viol = check(ep["messages"])
        s1 = any(v["sev"] == "S1" and v.get("executed") for v in viol)
        db = ep.get("reward") == 1.0
        q_flaw, p_ok = 0.0, None
        if po:
            g, f = po[a.out_classes.index("GOOD")], po[a.out_classes.index("FLAWED")]
            q_flaw = f / max(1e-6, g + f)
            if "FAIL" in a.out_classes:   # 结果模型的「正确」概率：去掉 FLAWED，只用 GOOD 和 FAIL 两类重新归一
                x = po[a.out_classes.index("FAIL")]
                p_ok = g / max(1e-6, g + x)
        if a.outcome_source == "model":   # 结果由模型监督：R = P(正确)，执行了 S1 严重违规一票否决；真实数据库结果只记录
            if p_ok is None:
                raise RuntimeError("--outcome_source model needs a 3-class outcome critic (GOOD/FLAWED/FAIL)")
            R = 0.0 if s1 else p_ok
        else:
            R = 0.0 if (not db or s1) else 1.0 - a.lam * q_flaw
        r_step = [sum(STEP_REWARD[c] * p[k] for k, c in enumerate(STEP_CLASSES)) for p in ps]
        tool_err = sum(1 for m in ep["messages"] if m.get("role") == "tool" and m.get("error"))
        samples.append({"ep": ep, "prompt_id": ep["prompt_id"], "R": R, "db": db, "s1": s1, "q_flaw": q_flaw, "p_ok": p_ok, "p_out": po,
                        "p_step": ps, "r_step": r_step, "tool_err": tool_err, "n_turns": len(ep["turns"])})
    groups = collections.defaultdict(list)
    for s in samples:
        groups[s["prompt_id"]].append(s)
    for g in groups.values():
        Rs = [s["R"] for s in g]
        mu, sd = sum(Rs) / len(Rs), (sum((x - sum(Rs) / len(Rs)) ** 2 for x in Rs) / len(Rs)) ** 0.5
        if a.step_mode == "floor":
            # 去底噪（同 RAG 句子层奖励 train.ans_reward）：每个非 OK 类别，底噪 b_c = 同组（同一任务 G 条采样的全部步骤）的平均概率，
            # σ_c = 标准差（下限 sigma_floor）；只有超出底噪的部分 e = max(0, p − b_c) 计分：A_step = Σ_c 分值_c × e_c / σ_c。
            # 处在平均水平的步骤 = 0（不因为多走几步拿分，也不因为 critic 对某类的普遍偏高——如 FIX 误报——白拿分）。
            bs = {}
            for k, c in enumerate(STEP_CLASSES):
                if c == "OK":
                    continue
                xs = [p[k] for s in g for p in s["p_step"]]
                b = sum(xs) / len(xs) if xs else 0.0
                sdv = (sum((x - b) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0
                if a.floor_caps and c in a.floor_caps:   # 底噪上限：最多扣到 critic 在正常步骤上的噪声水平，整组犯同一个错时照样罚得到
                    b = min(b, a.floor_caps[c])
                bs[k] = (b, max(sdv, a.sigma_floor))
            ki = {c: STEP_CLASSES.index(c) for c in STEP_CLASSES}
            for s in g:
                # 纠错的前提：同一条轨迹更早的步骤里确实有错可纠——ERR / AFTER_ERR 超出底噪至少一个 σ，或者工具报过错；否则 FIX / TRY 计 0
                ex = lambda p, c: (p[ki[c]] - bs[ki[c]][0]) / bs[ki[c]][1]
                terr = turn_tool_errors(s["ep"])
                gate, seen = [], False
                for t, p in enumerate(s["p_step"]):
                    gate.append(seen)
                    seen = seen or ex(p, "ERR") >= 1.0 or ex(p, "AFTER_ERR") >= 1.0 or terr[t]
                s["fix_gate"] = gate
            for s in g:
                # 分值按最大绝对值归一（FIX = 1），再截断到 ±step_clip：和全局优势（组内标准化，约 ±1–2）同一量级
                M = max(abs(v) for v in STEP_REWARD.values())
                s["A_step"] = [max(-a.step_clip, min(a.step_clip, sum(STEP_REWARD[STEP_CLASSES[k]] / M * max(0.0, p[k] - b) / sdv
                                                                       for k, (b, sdv) in bs.items()
                                                                       if s["fix_gate"][t] or STEP_CLASSES[k] not in ("FIX", "TRY"))))
                               for t, p in enumerate(s["p_step"])]
                s["floors"] = {STEP_CLASSES[k]: round(b, 4) for k, (b, _) in bs.items()}
        else:   # center：期望步骤分整组标准化（初版）
            rs = [x for s in g for x in s["r_step"]]
            smu = sum(rs) / len(rs) if rs else 0.0
            ssd = (sum((x - smu) ** 2 for x in rs) / len(rs)) ** 0.5 if rs else 0.0
            for s in g:
                s["A_step"] = [(x - smu) / (ssd + 1e-6) if ssd > 1e-6 else 0.0 for x in s["r_step"]]
        for s in g:
            s["A_glob"] = (s["R"] - mu) / (sd + 1e-6) if sd > 1e-6 else 0.0
            adv = [0.0] * len(s["ep"]["ids"])
            for t, ast in zip(s["ep"]["turns"], s["A_step"]):
                for k in range(t["a"], t["b"]):
                    adv[k] = s["A_glob"] + a.step_beta * ast
            s["adv"] = adv
    return samples


# ---------------------------------------------------------------- 训练
def token_logprobs(model, ids: list[int], pos: list[int], grad: bool):
    """位置 pos（预测 ids[pos] 的 token）的 logprob：隐状态只在这些位置过输出层，分块 + 反向时重算（同 cs_agent.sft）。"""
    import torch
    from torch.utils.checkpoint import checkpoint
    inner = model.base_model.model.model
    head = model.base_model.model.lm_head
    x = torch.tensor([ids], device="cuda")
    h = inner(input_ids=x).last_hidden_state[0]
    src = torch.tensor([p - 1 for p in pos], device="cuda")
    tgt = torch.tensor([ids[p] for p in pos], device="cuda")

    def lp_chunk(hc, tc):
        lg = head(hc).float()
        return lg.gather(1, tc[:, None])[:, 0] - torch.logsumexp(lg, -1)
    out = []
    for c in range(0, len(pos), 512):
        hc, tc = h[src[c:c + 512]], tgt[c:c + 512]
        out.append(checkpoint(lp_chunk, hc, tc, use_reentrant=False) if grad else lp_chunk(hc, tc))
    return torch.cat(out) if out else torch.zeros(0, device="cuda")


def check_share(model, heads, cache, samples: list[dict], n: int = 2, tol: float = 0.05) -> bool:
    """开训前自检：前 n 条轨迹，整段计算和共享系统提示各算一遍参考 / 策略 logprob 和过程 critic 概率，最大差值超过 tol 就不用共享。"""
    import torch
    worst = {"ref": 0.0, "policy": 0.0, "cstep": 0.0}
    for s in samples[:n]:
        ep = s["ep"]
        P = len(ep["prompt_ids"])
        ids = ep["prompt_ids"] + ep["ids"]
        pos = [P + k for k, m in enumerate(ep["lmask"]) if m][:2000]
        L = max(pos) + 1
        with torch.no_grad():
            with model.disable_adapter():
                a1 = token_logprobs(model, ids[:L], pos, grad=False)
                a2 = shared_logprobs(model, cache.frozen_prefix(model, "ref"), ids[:L], P, pos, grad=False)
            use_policy(model)
            b1 = token_logprobs(model, ids[:L], pos, grad=False)
            from train.prefix_share import prompt_pass
            b2 = shared_logprobs(model, prompt_pass(model, cache.pids, grad=False), ids[:L], P, pos, grad=False)
        c1, _ = score_episode(model, heads, ep, None)
        c2, _ = score_episode(model, heads, ep, cache)
        worst["ref"] = max(worst["ref"], (a1 - a2).abs().max().item())
        worst["policy"] = max(worst["policy"], (b1 - b2).abs().max().item())
        worst["cstep"] = max(worst["cstep"], max((abs(x - y) for p1, p2 in zip(c1, c2) for x, y in zip(p1, p2)), default=0.0))
    ok = all(v <= tol for v in worst.values())
    print(f"SHARE_CHECK max abs diff {json.dumps({k: round(v, 4) for k, v in worst.items()})} -> {'use shared prefix' if ok else 'FALLBACK to full sequences'}", flush=True)
    return ok


def train_step(model, opt, params, samples: list[dict], a, cache=None) -> dict:
    import torch
    keep = [s for s in samples if any(abs(x) > 1e-6 for x in s["adv"])]
    items = []
    for s in keep:
        ep = s["ep"]
        P = len(ep["prompt_ids"])
        ks = [k for k, m in enumerate(ep["lmask"]) if m and P + k < a.max_len]
        if ks:
            items.append((s, ks))
    n_tok = sum(len(ks) for _, ks in items) or 1
    st = collections.Counter()
    t0 = time.time()
    share = cache is not None and items and all(s["ep"]["prompt_ids"] == cache.pids for s, _ in items)
    if share:   # 策略的系统提示这一步只算一次（带梯度），全部轨迹接在它后面，最后统一反向
        from train.prefix_share import backward_prompt, leaves, prompt_pass
        use_policy(model)
        pre = prompt_pass(model, cache.pids, grad=True)
        leaf = leaves(pre)
        ref_pre = cache.frozen_prefix(model, "ref")
    for s, ks in items:
        ep = s["ep"]
        P = len(ep["prompt_ids"])
        L = P + max(ks) + 1
        ids = (ep["prompt_ids"] + ep["ids"])[:L]
        pos = [P + k for k in ks]
        with torch.no_grad(), model.disable_adapter():
            ref = shared_logprobs(model, ref_pre, ids, P, pos, grad=False) if share else token_logprobs(model, ids, pos, grad=False)
        use_policy(model)
        lp = shared_logprobs(model, leaf, ids, P, pos, grad=True) if share else token_logprobs(model, ids, pos, grad=True)
        old = torch.tensor([ep["old_lp"][k] for k in ks], device="cuda")
        adv = torch.tensor([s["adv"][k] for k in ks], device="cuda")
        ratio = torch.exp(lp - old)
        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv)
        kl = torch.exp(ref - lp) - (ref - lp) - 1
        loss = (-surr + a.kl * kl).sum() / n_tok
        loss.backward()
        st["kl"] += kl.detach().sum().item()
        st["clip"] += ((ratio.detach() - 1).abs() > a.clip).sum().item()
        st["old_absdiff"] += (lp.detach() - old).abs().sum().item()
        st["truncated_tok"] += sum(1 for k, m in enumerate(ep["lmask"]) if m) - len(ks)
    if share:
        backward_prompt(pre, leaf)
        del pre, leaf
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item() if items else 0.0
    if items:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return {"train_samples": len(items), "train_tokens": n_tok, "kl": st["kl"] / n_tok, "clip_frac": st["clip"] / n_tok,
            "vllm_hf_absdiff": st["old_absdiff"] / n_tok, "truncated_tokens": st["truncated_tok"], "grad_norm": gn, "t_train": round(time.time() - t0, 1),
            "shared_prefix": bool(share)}


def batch_stats(samples: list[dict], batch: dict, out_classes: list[str]) -> dict:
    n = max(1, len(samples))
    m = lambda f: round(sum(f(s) for s in samples) / n, 4)
    exp_cls = {c: m(lambda s, k=k: sum(p[k] for p in s["p_step"])) for k, c in enumerate(STEP_CLASSES)}   # 每条轨迹各类步骤的期望个数
    return {"episodes": len(batch["episodes"]), "scored": len(samples), "errors": sum(1 for e in batch["episodes"] if e.get("error")),
            "R": m(lambda s: s["R"]), "db": m(lambda s: s["db"]),
            "p_ok_db1": round(sum(s["p_ok"] or 0 for s in samples if s["db"]) / max(1, sum(s["db"] for s in samples)), 4),
            "p_ok_db0": round(sum(s["p_ok"] or 0 for s in samples if not s["db"]) / max(1, sum(not s["db"] for s in samples)), 4), "s1": m(lambda s: s["s1"]), "q_flaw_if_db": round(
                sum(s["q_flaw"] for s in samples if s["db"]) / max(1, sum(s["db"] for s in samples)), 4),
            "p_out": {c: m(lambda s, k=k: s["p_out"][k]) for k, c in enumerate(out_classes)}, "steps_per_traj": exp_cls,
            "r_step": m(lambda s: sum(s["r_step"]) / max(1, len(s["r_step"]))), "turns": m(lambda s: s["n_turns"]),
            "tool_err": m(lambda s: s["tool_err"]), "zero_adv_groups": sum(1 for g in {s["prompt_id"] for s in samples}
                                                                         if all(abs(s["A_glob"]) < 1e-6 for s in samples if s["prompt_id"] == g)),
            "t_gen": round(batch["t_gen"], 1), "staleness_version": batch["version"]}


def save_samples(samples: list[dict], step: int, out: Path) -> None:
    d = out / "samples"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"step{step}.jsonl.gz", "wt", encoding="utf-8") as f:
        for s in samples:
            ep = s["ep"]
            f.write(json.dumps({"task_id": ep["task_id"], "trial": ep.get("trial"), "R": s["R"], "db": s["db"], "s1": s["s1"], "q_flaw": s["q_flaw"],
                                "p_out": s["p_out"], "r_step": s["r_step"], "A_glob": s["A_glob"], "A_step": s["A_step"],
                                "p_step": [[round(x, 4) for x in p] for p in s["p_step"]], "messages": ep["messages"]}, ensure_ascii=False) + "\n")


def run(a) -> None:
    import torch
    from cs_agent.rollout import load_tasks
    from train.gen_grpo import Server
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, heads = load_models(a)
    params = policy_params(model)
    print(f"GRPO policy params {sum(p.numel() for p in params) / 1e6:.1f}M trainable", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    servers = [Server(u) for u in a.policy_url.split(",") if u.strip()]   # 多台推理机：逗号分隔，LoRA 每步推给每一台
    for server in servers:
        for v in range(1, a.keep_versions + 3):   # 新开一轮：推理机上同名的旧版本（冒烟测试、上一次中断的训练）先卸掉
            try:
                server.unload_lora(f"policy_v{v}")
            except Exception:  # noqa: BLE001
                pass
    print(f"inference nodes: {len(servers)}", flush=True)
    tasks = load_tasks("train")
    version = [0]
    prod = Producer(a, tok, tasks, servers, version)
    prod.start()
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    step = 0
    t_wait0, cache = time.time(), None
    while step < a.steps:
        got, n_stale = [], 0
        while len(got) < a.tasks_per_step:   # 凑够 tasks_per_step 组；太旧的组（最早的一条落后超过 max_stale 个版本）丢掉
            try:
                G = prod.q.get(timeout=60)
            except queue.Empty:
                if prod.err:
                    raise RuntimeError(f"producer died: {prod.err}")
                continue
            prod.consumed += 1
            if version[0] - G["version"] > a.max_stale:
                n_stale += 1
                continue
            got.append(G)
        batch = {"episodes": [e for G in got for e in G["episodes"]], "version": min(G["version"] for G in got),
                 "t_gen": time.time() - t_wait0, "stale_groups_dropped": n_stale,
                 "versions": dict(collections.Counter(e.get("version", 0) for G in got for e in G["episodes"]))}
        if a.share and cache is None:
            eps0 = [e for e in batch["episodes"] if not e.get("error") and e.get("turns")]
            if eps0:
                cache = PrefixCache(eps0[0]["prompt_ids"])
                try:
                    ok = check_share(model, heads, cache, [{"ep": e} for e in eps0])
                except Exception as e:  # noqa: BLE001  共享实现依赖 transformers 内部结构，出错就退回整段计算，不让训练停下
                    print(f"SHARE_CHECK error {repr(e)[:300]} -> FALLBACK to full sequences", flush=True)
                    ok = False
                if not ok:
                    a.share, cache = 0, None
                    import torch
                    torch.cuda.empty_cache()
        t0 = time.time()
        samples = rewards(batch, model, heads, a, cache)
        t_rm = time.time() - t0
        st = batch_stats(samples, batch, a.out_classes)
        tr = train_step(model, opt, params, samples, a, cache)
        step += 1
        d = out / "lora" / f"v{step}"
        model.save_pretrained(str(d), selected_adapters=["policy"])
        path = d / "policy" if (d / "policy").exists() else d
        for server in servers:
            server.load_lora(f"policy_v{step}", str(path))
        version[0] = step   # 推理机加载好新版本之后，采样线程才开始用它
        if step - a.keep_versions >= 1:
            for server in servers:
                server.unload_lora(f"policy_v{step - a.keep_versions}")
            old = out / "lora" / f"v{step - a.keep_versions}"
            if (step - a.keep_versions) % a.save_every:
                import shutil
                shutil.rmtree(old, ignore_errors=True)
        save_samples(samples, step, out)
        eps_ok = [e for e in batch["episodes"] if not e.get("error")]
        walls = sorted(e.get("wall", 0) for e in eps_ok)
        rec = {"step": step, **st, **tr, "t_rm": round(t_rm, 1), "staleness": step - 1 - batch["version"], "versions": batch["versions"],
               "stale_groups_dropped": batch["stale_groups_dropped"], "timeouts": sum(e.get("term") == "timeout" for e in eps_ok),
               "wall_p50": walls[len(walls) // 2] if walls else None, "wall_max": walls[-1] if walls else None,
               "t_step": round(time.time() - t_wait0, 1), "time": time.strftime("%F %T")}
        t_wait0 = time.time()
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print("STEP", json.dumps(rec, ensure_ascii=False), flush=True)
    prod.stop = True
    model.save_pretrained(str(out / "final"), selected_adapters=["policy"])
    (out / "grpo_manifest.json").write_text(json.dumps({**vars(a), "steps_done": step}, indent=1), encoding="utf-8")
    print("GRPO_DONE", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--init", required=True, help="策略起点 LoRA（B 全零 = 原版；scripts/make_lora_init.py）")
    ap.add_argument("--critic_step", required=True)
    ap.add_argument("--critic_outcome", default="", help="结果模型（cs_agent.critic --task outcome 的检查点）")
    ap.add_argument("--outcome_source", choices=["model", "db"], default="model",
                    help="model：全局奖励 = 结果模型的 P(正确)（GOOD 和 FAIL 两类重新归一，不看 FLAWED）；db：真实数据库比对")
    ap.add_argument("--policy_url", required=True)
    ap.add_argument("--user_url", default="", help="（不再使用：用户模拟器和策略在同一台推理机上）")
    ap.add_argument("--user_model", default="base")
    ap.add_argument("--user_key", default="x")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tasks_per_step", type=int, default=8)
    ap.add_argument("--n", type=int, default=8, help="每个任务采几条（组大小 G）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=20, help="每台推理机同时跑的对话数（每条对话策略 + 模拟器两个上下文；32 条时 KV 满、前缀缓存被挤掉，生成反而慢一半）")
    ap.add_argument("--ahead_groups", type=int, default=16, help="采样最多领先训练多少组（已发出但还没被训练取走）")
    ap.add_argument("--ep_timeout", type=float, default=900.0, help="一条对话的墙钟上限（秒），超时按没办完算；0 = 不限")
    ap.add_argument("--gen_max_tokens", type=int, default=4096)
    ap.add_argument("--max_ctx", type=int, default=32000)
    ap.add_argument("--max_len", type=int, default=16384, help="反向传播的最大长度")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.0, help="全局奖励里全流程 reward 过程扣分的系数 λ（默认 0：过程只由过程 critic 把控）")
    ap.add_argument("--step_beta", type=float, default=0.5, help="步骤优势的权重 β")
    ap.add_argument("--step_mode", choices=["floor", "center"], default="floor", help="floor：各类别减同组平均（底噪）后只计超出部分；center：期望步骤分组内标准化")
    ap.add_argument("--sigma_floor", type=float, default=0.1, help="去底噪时标准差的下限（一组都很干净时别把噪声放大）")
    ap.add_argument("--step_clip", type=float, default=3.0)
    ap.add_argument("--floor_caps", default="", help="过程 critic 留出集打分（scores.jsonl）：取真实 OK 步骤上各类别平均概率作底噪上限；空 = 不设上限")
    ap.add_argument("--max_stale", type=int, default=2)
    ap.add_argument("--keep_versions", type=int, default=4, help="推理机上保留几个策略版本：要大于 max_stale，进行中的对话还在用旧版本")
    ap.add_argument("--save_every", type=int, default=10, help="每多少步保留一个 LoRA 版本（其余旧版本删掉）")
    ap.add_argument("--share", type=int, default=1, help="1 = 系统提示共享（参考模型、两个奖励模型的系统提示状态只算一次，策略每步算一次）；开训前自检不过自动退回整段")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.floor_caps = caps_from_scores(a.floor_caps) if a.floor_caps else {}
    print("floor caps", a.floor_caps, flush=True)
    run(a)


if __name__ == "__main__":
    main()
