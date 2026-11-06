"""回答 GRPO：9B 生成回答，critic 逐句打分当回报，看幻觉能不能降（2026-11-02，用户）。训练框架同 train.gen_grpo：

  GPU1：vLLM 挂基座、当回报的 critic、对照 critic（只评测用，不进奖励）和策略 LoRA，训练中热加载；
  GPU0：HF + PEFT 训策略 LoRA（起点 B 全零，等于 9B 原始模型），prompt 共享（--share 1），PPO 截断 − β·KL（相对起点）。

  python -m train.ans_grpo --make_init --template <已有 r32 LoRA> --out <目录>   # B 全零、A 随机的起点
  python -m train.ans_grpo --init <起点> --out <目录> [--steps 300] [--lr …]

每步 16 prompt × 8 条（温度 1.0，策略梯度要求）；每 50 步在 dev（100 个没参与训练的原文）× 2 次上评（第 50 步之前是 8 次，用户：评测样本减到 1/4）（温度 0.7，同 9B 摸底；种子固定，和第 0 步逐原文配对），
RM 分同时看对照 critic：回报 critic 的分降、对照不降，就是在骗回报。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import gen_grpo
from train.ans_reward import AnswerRewarder, combine, sentence_adv, sentences
from ragtruth.ans_data import policy_prompt
from train.gen_grpo import Server, load_policy, load_state, push, save_samples, save_state, train_step, vllm_headers, vllm_key


def load_causal_policy(args):
    """纯文本的 CausalLM 策略（MiniCPM5-2B，Llama 结构）：和 gen_grpo.load_policy 一样挂 policy / ref 两个 LoRA。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.init, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.init, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    for p in gen_grpo.lora_params(model, "ref"):
        p.requires_grad_(False)
    for p in gen_grpo.lora_params(model, "policy"):
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    return tok, model


def make_init_causal(args) -> None:
    """CausalLM 的零起点：PEFT 默认初始化（A 随机、B 全零），r32 / alpha 64，所有线性层。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    m = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0))
    m.save_pretrained(args.out)
    print("INIT_CAUSAL ->", args.out, flush=True)


def make_init(args) -> None:
    """按已有 LoRA 的键和配置造起点：lora_B 全零（输出和 9B 完全相同），lora_A 随机（PEFT 默认的 kaiming）。"""
    import math
    import torch
    from safetensors.torch import load_file, save_file
    src, out = Path(args.template), Path(args.out)
    w = load_file(str(src / "adapter_model.safetensors"))
    torch.manual_seed(0)
    new = {}
    for k in sorted(w):
        t = torch.zeros_like(w[k])
        if ".lora_A." in k:
            torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5))
        new[k] = t.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(new, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_bytes((src / "adapter_config.json").read_bytes())
    print("INIT", len(new), "tensors ->", out, flush=True)


def push_all(model, out: Path, servers: list, version: int, keep_versions: int, prefix: str = "") -> None:
    """同 gen_grpo.push，但新版本 LoRA 推给所有服务（两张卡各一个同样的服务）。"""
    import shutil
    d = out / "lora" / f"v{version}"
    model.save_pretrained(str(d), selected_adapters=["policy"])
    path = d / "policy" if (d / "policy").exists() else d
    for s in servers:
        s.load_lora(f"{prefix}policy_v{version}", str(path))
    old = version - keep_versions
    if old >= 1:
        for s in servers:
            s.unload_lora(f"{prefix}policy_v{old}")
        shutil.rmtree(out / "lora" / f"v{old}", ignore_errors=True)


class AsyncPusher:
    """后台推送 LoRA（2026-11-04，双机模式：推送要上传到推理机，约 1.5–3 秒，不能挡训练）。
    主线程只做 save_pretrained 到本地目录（内存盘约 0.2 秒，存完的快照不受下一步训练影响），上传 / 热加载 / 卸载旧版本在后台线程；
    后台加载成功后才把 prod.version 指过去，所以采样永远只用已加载的版本（最多晚一步，在 max_stale 之内）。
    同一时刻只有一个推送在飞：下一次推送前先等上一次结束；后台出错就在下一次 push / wait 时抛出，训练停下，不会悄悄沿用旧版本。"""

    def __init__(self, servers: list, prod, keep_versions: int, prefix: str = ""):
        self.servers, self.prod, self.keep, self.prefix = servers, prod, keep_versions, prefix
        self.th, self.err, self.last_bg = None, None, 0.0

    def wait(self) -> None:
        if self.th is not None:
            self.th.join()
            self.th = None
        if self.err is not None:
            raise RuntimeError("async LoRA push failed") from self.err

    def _run(self, out: Path, version: int, path: Path) -> None:
        import shutil
        t0 = time.time()
        try:
            for s in self.servers:
                s.load_lora(f"{self.prefix}policy_v{version}", str(path))
            self.prod.version = version
            old = version - self.keep
            if old >= 1:
                for s in self.servers:
                    s.unload_lora(f"{self.prefix}policy_v{old}")
                shutil.rmtree(out / "lora" / f"v{old}", ignore_errors=True)
        except BaseException as e:  # noqa: BLE001
            self.err = e
        self.last_bg = time.time() - t0

    def push(self, model, out: Path, version: int) -> None:
        self.wait()
        d = out / "lora" / f"v{version}"
        model.save_pretrained(str(d), selected_adapters=["policy"])
        path = d / "policy" if (d / "policy").exists() else d
        self.th = threading.Thread(target=self._run, args=(out, version, path), daemon=True)
        self.th.start()


def cache_hits(url: str) -> tuple[float, float]:
    """vLLM /metrics 的前缀缓存累计（命中 token，查询 token）。url 可以是逗号分隔的几个服务，加总。"""
    import requests
    if "," in url:
        hs = [cache_hits(u) for u in url.split(",")]
        return sum(h for h, _ in hs), sum(q for _, q in hs)
    txt = requests.get(url.rsplit("/v1", 1)[0] + "/metrics", headers=vllm_headers(), timeout=10).text
    get = lambda n: sum(float(m) for m in re.findall(rf"^vllm:{n}(?:{{[^}}]*}})? ([\d.e+]+)$", txt, re.M))
    return get("prefix_cache_hits_total"), get("prefix_cache_queries_total")


def token_sentence_index(tok, ids: list[int], text: str, spans: list[list[int]]) -> list[int]:
    """每个回答 token 落在第几句（spans 是 strip 后文本里的字符区间）；不在任何句子里的（空白、结尾符号外）= −1。
    逐 token 累计解码得到字符终点；解码长度和 text 对不上时按比例缩放（不同分词器的空格处理）。"""
    lead = len(text) - len(text.lstrip())
    ends = [len(tok.decode(ids[:k + 1], skip_special_tokens=True)) for k in range(len(ids))]
    full = ends[-1] if ends else 0
    if full and full != len(text):
        ends = [round(e * len(text) / full) for e in ends]
    out, prev = [], 0
    for e in ends:
        mid = (prev + e) / 2 - lead   # token 的中点，换到 strip 后的坐标
        out.append(next((j for j, (a, b) in enumerate(spans) if a <= mid < b), -1))
        prev = e
    return out


def token_advantages(a_out: float, sent_adv: list[float], tok_idx: list[int], beta: float) -> list[float]:
    """每个 token 的优势 = 整段层（组内归一化的结果奖励）+ beta × 所在句子的句子层优势。"""
    return [a_out + (beta * sent_adv[j] if 0 <= j < len(sent_adv) else 0.0) for j in tok_idx]


class Producer:
    def __init__(self, args, tok, servers: list, rw: AnswerRewarder):
        self.a, self.tok, self.servers, self.rw = args, tok, servers, rw
        self.rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
        self.pos, self.version, self.stop, self.dead = 0, 0, False, None
        self.q = queue.Queue(maxsize=max(1, args.max_stale - 1))
        self.gen_pool = ThreadPoolExecutor(args.prompts_per_step)
        self.score_pool = ThreadPoolExecutor(args.prompts_per_step)

    def make_batch(self) -> dict:
        from train.gen_reward import group_advantages
        src = getattr(self, "version_src", None)   # 子进程模式：版本号在主进程和子进程共享的整数里
        v = src.value if src is not None else self.version
        rows = [self.rows[(self.pos + i) % len(self.rows)] for i in range(self.a.prompts_per_step)]
        self.pos += self.a.prompts_per_step
        name = "policy_v0" if v == 0 else f"{self.a.version_prefix}policy_v{v}"
        prompts = [policy_prompt(self.tok, r, self.a.sys_qa) for r in rows]
        t0 = time.time()
        # 发 token id（含分词器默认的特殊符，MiniCPM 是开头的 <s>），训练侧用同一份 id。vLLM 自己分词会加 <s>、训练侧原来不加，第 1 步截断比例 22%（2026-11-02）
        ids = [self.tok(p)["input_ids"] for p in prompts]
        if getattr(self.a, "pipeline", 0):
            # 每个 prompt 生成完立刻打分，生成和打分重叠（2026-11-05，用户：采样端也优化；原来整批生成完再整批打分，约 7 + 12 秒）
            def one(k):
                a = time.time()
                g = self.servers[k % len(self.servers)].sample(name, ids[k], self.a.n, self.a.max_tokens, self.a.temperature)
                b = time.time()
                return g, self.rw.score_group(rows[k], g), b - a, time.time() - b
            res = list(self.gen_pool.map(one, range(len(rows))))
            gens, scored = [r[0] for r in res], [r[1] for r in res]
            t1 = t0 + max(r[2] for r in res)   # t_gen = 最慢的生成；t_rm = 其余的墙钟时间
        else:
            gens = list(self.gen_pool.map(lambda kp: self.servers[kp[0] % len(self.servers)].sample(name, kp[1], self.a.n, self.a.max_tokens, self.a.temperature), enumerate(ids)))
            t1 = time.time()
            scored = list(self.score_pool.map(lambda x: self.rw.score_group(*x), zip(rows, gens)))
        samples = []
        beta = getattr(self.a, "sent_beta", 0.0)
        for r, p, gs, ss in zip(rows, prompts, gens, scored):
            adv = group_advantages([s["reward"] for s in ss], [None] * len(ss))
            pids = self.tok(p)["input_ids"]
            for g, s, a in zip(gs, ss, adv):
                if beta > 0 and s.get("ps") and s.get("spans") and g.get("ids"):
                    # 分步奖励（2026-11-05，用户：单独优化出错的那一句）：错句只罚自己的 token
                    sa = sentence_adv(s["ps"], s["sent_base"], s["sent_sigma"], self.a.sent_mode, self.a.sent_gamma)
                    a = token_advantages(a, sa, token_sentence_index(self.tok, g["ids"], g["text"], s["spans"]), beta)
                samples.append({"prompt_id": r["id"], "task_type": r["task_type"], "prompt_ids": pids, **g, **s, "adv": a})
        return {"version": v, "samples": samples, "t_gen": t1 - t0, "t_rm": time.time() - t1, "pos": self.pos}

    def run(self) -> None:
        fails = 0
        while not self.stop:
            try:
                b = self.make_batch()
                fails = 0
            except Exception as e:
                if self.stop:
                    return
                fails += 1
                print("PRODUCER_ERROR", fails, repr(e)[:500], flush=True)
                if fails >= 5:   # 主线程在等样本，不能让它静默卡死
                    self.dead = e
                    return
                time.sleep(30)
                continue
            while not self.stop:
                try:
                    self.q.put(b, timeout=5)
                    break
                except queue.Full:
                    continue

    def get(self) -> dict:
        while True:
            try:
                return self.q.get(timeout=30)
            except queue.Empty:
                if self.dead is not None:
                    raise RuntimeError("producer died") from self.dead


def build_rewarder(args) -> AnswerRewarder:
    from openai import OpenAI
    temp = json.load(open(args.critic_report, encoding="utf-8"))["temperature"]
    sat_temp = json.load(open(args.sat_report, encoding="utf-8"))["temperature"] if args.sat_model else 1.0
    refuse_temp = json.load(open(args.refuse_report, encoding="utf-8"))["temperature"] if args.refuse_model else 1.0
    rw = AnswerRewarder([OpenAI(base_url=u, api_key=vllm_key(), timeout=600, max_retries=3) for u in args.vllm_url.split(",")], args.critic, temp,
                        ThreadPoolExecutor(args.rm_concurrency), args.sat_model, args.sat_kind, sat_temp, args.sat_w, args.refuse_model, refuse_temp, args.refuse_w,
                        getattr(args, "reward_shape", "legacy"), getattr(args, "count_k", 3.0), getattr(args, "lam_any", 0.5), getattr(args, "lam_cnt", 0.5),
                        getattr(args, "sigma_floor", 0.1), getattr(args, "count_ratio", 0.0), getattr(args, "count_min", 2.0))
    rw.group_pool = ThreadPoolExecutor(args.rm_concurrency)
    return rw


def _producer_proc(args, out: str, start_pos: int, vval, bq, cmd, ack) -> None:
    """子进程（2026-11-05，用户：训练慢、怀疑是锁）：采样 / 打分 / 快评都在这里，训练进程独占自己的 GIL。
    实测同进程里 16 个解析 JSON 的线程就能让训练线程拿不到锁、GPU 0%；真实训练 t_train 24 秒，单独跑同样的计算 10.4 秒。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    servers = [Server(u) for u in (args.policy_url or args.vllm_url).split(",")]
    rw = build_rewarder(args)
    prod = Producer(args, tok, servers, rw)
    prod.pos, prod.q, prod.version_src = start_pos, bq, vval
    qe = QuickEval(args, tok, servers, rw, Path(out)) if args.eval_k > 0 else None

    def run_prod():
        prod.run()
        if prod.dead is not None:
            bq.put({"__dead__": repr(prod.dead)[:500]})
    threading.Thread(target=run_prod, daemon=True).start()
    while True:
        c = cmd.get()
        if c[0] == "eval" and qe:
            qe.submit(c[1], Path(c[2]) if c[2] else None)
        elif c[0] == "first" and qe:
            qe.first = first_from_samples(Path(c[1]), args.eval_k)
        elif c[0] == "stop":
            prod.stop = True
            if qe:
                qe.wait()
            ack.put("done")
            return


class ProcProducer:
    """主进程这边的代理，接口同 Producer（get / version / pos），另管子进程的启动和收尾。"""

    def __init__(self, args, out: Path):
        import multiprocessing as mp
        self.ctx, self.a, self.out = mp.get_context("spawn"), args, out
        self._v = self.ctx.Value("i", 0)
        self.pos, self.p = 0, None
        self.q = self.ctx.Queue(maxsize=max(1, args.max_stale - 1))
        self.cmd, self.ack = self.ctx.Queue(), self.ctx.Queue()
        self.target = _producer_proc   # 测试里换成假的子进程

    @property
    def version(self) -> int:
        return self._v.value

    @version.setter
    def version(self, v: int) -> None:
        self._v.value = int(v)

    def start(self) -> None:   # load_state 设好 pos 之后再启动
        self.p = self.ctx.Process(target=self.target, args=(self.a, str(self.out), self.pos, self._v, self.q, self.cmd, self.ack), daemon=True)
        self.p.start()

    def get(self) -> dict:
        while True:
            try:
                b = self.q.get(timeout=30)
            except queue.Empty:
                if self.p is not None and not self.p.is_alive():
                    raise RuntimeError(f"producer process died (exit {self.p.exitcode})")
                continue
            if "__dead__" in b:
                raise RuntimeError("producer died: " + b["__dead__"])
            self.pos = b.get("pos", self.pos)
            return b

    def shutdown(self, timeout: float = 7200) -> None:
        self.cmd.put(("stop",))
        try:
            self.ack.get(timeout=timeout)   # 等子进程把排着的快评做完
        except queue.Empty:
            pass
        self.p.join(60)
        if self.p.is_alive():
            self.p.terminate()


class EvalProxy:
    """主进程这边的快评代理：submit / first / wait 转成发给子进程的命令。"""

    def __init__(self, prod: ProcProducer):
        self.prod = prod

    def submit(self, step: int, path: Path | None) -> None:
        self.prod.cmd.put(("eval", step, str(path) if path else ""))

    @property
    def first(self):
        return None

    @first.setter
    def first(self, _v) -> None:   # 主进程原来直接算好赋值；子进程自己从文件读
        self.prod.cmd.put(("first", str(self.prod.out / f"quick_samples_{self.prod.a.version_prefix}s0.jsonl")))

    def wait(self) -> None:   # 由 shutdown 等
        pass


def sat_stats(xs: list[dict]) -> dict:
    if not xs:
        return {}
    return {"sat": round(sum(s["sat"] for s in xs) / len(xs), 4), "sat_low": round(sum(s["sat"] < .5 for s in xs) / len(xs), 4)}


def batch_stats(batch: dict) -> dict:
    ss = batch["samples"]
    ok = [s for s in ss if s["why"] == "ok"]
    m = lambda k, xs: round(sum(x[k] for x in xs) / len(xs), 4) if xs else None
    by = {t: [s for s in ok if s["task_type"] == t] for t in ("QA", "Summary")}
    return {"samples": len(ss), "reward": m("reward", ss), "ok_rate": round(len(ok) / len(ss), 4), "why": dict(collections.Counter(s["why"] for s in ss)),
            "p_mean": m("p_mean", ok), "p_max": m("p_max", ok), "p_mean_qa": m("p_mean", by["QA"]), "p_mean_summary": m("p_mean", by["Summary"]),
            "answer_flag": round(sum(s["p_max"] >= .5 for s in ok) / len(ok), 4) if ok else None,
            "chars": m("chars", ss), "chars_qa": m("chars", by["QA"]), "chars_summary": m("chars", by["Summary"]), "n_sent": m("n_sent", ss), "copy": m("copy", ss),
            "refusal": m("refusal", ss), "over_limit": m("over_limit", ss), "truncated": sum(s["finish"] == "length" for s in ss),
            **sat_stats([s for s in ok if s.get("sat") is not None]),
            "completion_tokens": round(sum(len(s["ids"]) for s in ss) / len(ss), 1), "t_gen": round(batch["t_gen"], 1), "t_rm": round(batch["t_rm"], 1)}


class TripWire:
    """指标越线就存状态、停（第 45 步起，最近 20 步和第 6–25 步比）。这些都不来自 critic，防的是不答题也能拿高分。"""

    def __init__(self, start=45, window=20, base=(5, 25), len_low=0.75, len_high=1.3, copy_up=0.15, refusal_up=0.05, ok_drop=0.15, kl_max=0.10):
        self.start, self.window, self.base = start, window, base
        self.len_low, self.len_high, self.copy_up, self.refusal_up, self.ok_drop, self.kl_max = len_low, len_high, copy_up, refusal_up, ok_drop, kl_max
        self.hist: list[dict] = []

    def update(self, rec: dict) -> list[str]:
        self.hist.append(rec)
        if len(self.hist) < max(self.start, self.base[1] + self.window):
            return []
        mean = lambda k, rows: sum((r.get(k) or 0.0) for r in rows) / len(rows)
        now, base = self.hist[-self.window:], self.hist[self.base[0]:self.base[1]]
        trips = []
        for k in ("chars_qa", "chars_summary"):
            if mean(k, base) > 0 and not self.len_low <= mean(k, now) / mean(k, base) <= self.len_high:
                trips.append(f"{k} {mean(k, now):.0f} vs base {mean(k, base):.0f}")
        if mean("copy", now) >= mean("copy", base) + self.copy_up:
            trips.append(f"copy {mean('copy', now):.3f} vs base {mean('copy', base):.3f}")
        if mean("refusal", now) >= mean("refusal", base) + self.refusal_up:
            trips.append(f"refusal {mean('refusal', now):.3f} vs base {mean('refusal', base):.3f}")
        if mean("ok_rate", now) <= mean("ok_rate", base) - self.ok_drop:
            trips.append(f"ok_rate {mean('ok_rate', now):.3f} vs base {mean('ok_rate', base):.3f}")
        if mean("kl", now) >= self.kl_max:
            trips.append(f"kl {mean('kl', now):.3f} >= {self.kl_max}")
        return trips


def eval_metrics(rows: list[dict], gens: list[list[dict]], reward: list[list[dict]], shadow: list[list[list[float] | None]]) -> dict:
    """dev 上一个检查点的指标。by_source 留给和第 0 步配对。"""
    flat = [(r, g, s, sh) for r, gs, ss, shs in zip(rows, gens, reward, shadow) for g, s, sh in zip(gs, ss, shs)]
    n = len(flat)
    live = [x for x in flat if x[2].get("ps")]
    m = lambda f, xs: round(sum(f(x) for x in xs) / max(1, len(xs)), 4)
    rec = {"answers": n, "scored": len(live), "ok_rate": m(lambda x: x[2]["why"] == "ok", flat), "truncated": m(lambda x: x[1]["finish"] == "length", flat),
           "refusal": m(lambda x: x[2]["refusal"], flat), "over_limit": m(lambda x: x[2]["over_limit"], flat), "copy": m(lambda x: x[2]["copy"], flat),
           "chars": m(lambda x: x[2]["chars"], flat), "n_sent": m(lambda x: x[2]["n_sent"], flat),
           "reward": m(lambda x: combine(x[2]["ps"]), live), "p_mean": m(lambda x: sum(x[2]["ps"]) / len(x[2]["ps"]), live),
           "p_max": m(lambda x: max(x[2]["ps"]), live), "sent_flag": m(lambda x: sum(p >= .5 for p in x[2]["ps"]) / len(x[2]["ps"]), live), "answer_flag": m(lambda x: max(x[2]["ps"]) >= .5, live),
           **({"shadow_p_mean": m(lambda x: sum(x[3]) / len(x[3]), sh_live), "shadow_answer_flag": m(lambda x: max(x[3]) >= .5, sh_live)} if (sh_live := [x for x in live if x[3]]) else {}),
           **sat_stats([x[2] for x in live if x[2].get("sat") is not None]),
           **({"refusal_p_refuse": m(lambda x: x[2]["p_refuse"], rf)} if (rf := [x for x in flat if x[2].get("p_refuse") is not None]) else {})}
    for t in ("QA", "Summary"):
        xs = [x for x in live if x[0]["task_type"] == t]
        al = [x for x in flat if x[0]["task_type"] == t]
        rec[t] = {"p_mean": m(lambda x: sum(x[2]["ps"]) / len(x[2]["ps"]), xs), "answer_flag": m(lambda x: max(x[2]["ps"]) >= .5, xs), "chars": m(lambda x: x[2]["chars"], al), "refusal": m(lambda x: x[2]["refusal"], al),
                  "over_limit": m(lambda x: x[2]["over_limit"], al), "copy": m(lambda x: x[2]["copy"], al)}
    return rec


def paired_delta(now: dict, first: dict, n: int = 1000, seed: int = 0) -> list[float]:
    """按原文配对 bootstrap：两个 {原文: 值} 的均值差的 95% 区间。"""
    keys = sorted(set(now) & set(first))
    rng = random.Random(seed)
    v = sorted(sum(now[k] - first[k] for k in (rng.choice(keys) for _ in keys)) / len(keys) for _ in range(n))
    return [round(v[int(.025 * n)], 4), round(v[int(.975 * n)], 4)]


def by_source(rows: list[dict], reward: list[list[dict]], shadow: list[list]) -> dict:
    """{指标: {原文: 这个原文几条回答的均值}}，用来和第 0 步配对。"""
    fs = {"p_mean": lambda s, sh: sum(s["ps"]) / len(s["ps"]), "answer_flag": lambda s, sh: float(max(s["ps"]) >= .5), "shadow_p_mean": lambda s, sh: sum(sh) / len(sh)}
    if not any(sh for shs in shadow for sh in shs):   # 训练中没打对照分（--shadow 为空）
        fs.pop("shadow_p_mean")
    by = {}
    for k, f in fs.items():
        by[k] = {}
        for r, ss, shs in zip(rows, reward, shadow):
            v = [f(s, sh) for s, sh in zip(ss, shs) if s.get("ps")]
            if v:
                by[k][r["source_id"]] = sum(v) / len(v)
    return by


def first_from_samples(path: Path, k: int) -> dict:
    """从存盘的第 0 步样本恢复基线，只取每个原文前 k 条：和当前评测的条数同口径。"""
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    return by_source([{"source_id": l["id"]} for l in lines], [l["scores"][:k] for l in lines], [l["shadow"][:k] for l in lines])


class QuickEval:
    """dev × k 次：种子固定，优先级靠后，不挡训练。写 quick_eval.jsonl、quick_samples_s{step}.jsonl。"""

    def __init__(self, args, tok, servers: list, rw: AnswerRewarder, out: Path):
        self.a, self.servers, self.rw, self.out, self.tok = args, servers, rw, out, tok
        self.rows = [{**r, "chat": policy_prompt(tok, r, args.sys_qa)} for r in map(json.loads, open(args.dev, encoding="utf-8"))]
        self.pool = ThreadPoolExecutor(1)
        self.futs, self.first = [], None

    def submit(self, step: int, path: Path | None) -> None:
        self.futs.append(self.pool.submit(self._run, step, path))

    def _sample(self, name: str, r: dict) -> list[dict]:
        srv = self.servers[hash(r["id"]) % len(self.servers)]
        out = srv.client.completions.create(model=name, prompt=self.tok(r["chat"])["input_ids"], n=self.a.eval_k, max_tokens=self.a.max_tokens, temperature=self.a.eval_temperature,
                                            top_p=1.0, seed=0, extra_body={"priority": 1, "top_k": -1})
        return [{"text": c.text, "finish": c.finish_reason} for c in out.choices]

    def _score(self, r: dict, gs: list[dict]) -> tuple[list[dict], list[list[float] | None]]:
        ss, shadow = [], []
        for g in gs:   # 一条接一条：共享前缀，也别和训练抢
            s = self.rw.describe(g["text"], g["finish"], {**r, "ref_chars": None})
            t = g["text"].strip()
            sents = sentences(t)
            ps = self.rw.p_answer(r["source"], sents, priority=1) if sents and g["finish"] == "stop" else None
            sh = self.rw.p_answer(r["source"], sents, model=self.a.shadow, temp=self.a.shadow_temp, priority=1) if ps and self.a.shadow else None
            sat = self.rw.p_sat(r, t, priority=1) if ps else None
            if s["refusal"] and self.rw.refuse_model:   # 拒答落在拒答 RM 判的哪一档：降拒答率时看降的是不是该答的题
                s["p_refuse"] = self.rw.p_refuse(r, priority=1)
            ss.append({**s, **({"ps": [round(p, 4) for p in ps]} if ps else {}), **({"sat": sat} if sat is not None else {})})
            shadow.append(sh)
        return ss, shadow

    def _run(self, step: int, path: Path | None) -> None:
        try:
            t0 = time.time()
            name = "policy_v0"
            if path is not None:
                name = f"{self.a.version_prefix}eval_s{step}"
                for s in self.servers:
                    s.load_lora(name, str(path / "policy" if (path / "policy").exists() else path))
            with ThreadPoolExecutor(self.a.eval_concurrency) as ex:
                gens = list(ex.map(lambda r: self._sample(name, r), self.rows))
                res = list(ex.map(lambda x: self._score(*x), zip(self.rows, gens)))
            if path is not None:
                for s in self.servers:
                    s.unload_lora(name)
            reward, shadow = [x[0] for x in res], [x[1] for x in res]
            rec = {"step": step, **eval_metrics(self.rows, gens, reward, shadow), "t_wall": round(time.time() - t0, 1)}
            by = by_source(self.rows, reward, shadow)
            if self.first is None:
                self.first = by
            else:
                rec["delta_ci"] = {k: paired_delta(by[k], self.first[k]) for k in by if k in self.first}
            with open(self.out / f"quick_samples_{self.a.version_prefix}s{step}.jsonl", "w", encoding="utf-8") as f:
                for r, gs, ss, shs in zip(self.rows, gens, reward, shadow):
                    f.write(json.dumps({"id": r["id"], "task_type": r["task_type"], "outputs": [g["text"] for g in gs], "finish": [g["finish"] for g in gs], "scores": ss, "shadow": shs}, ensure_ascii=False) + "\n")
            with open(self.out / "quick_eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("QUICK_EVAL", json.dumps(rec, ensure_ascii=False), flush=True)
        except Exception as e:   # 评测出错不影响训练
            print("QUICK_EVAL_FAILED", step, repr(e), flush=True)

    def wait(self) -> None:
        for f in self.futs:
            f.result()


def run(args) -> None:
    import torch
    from openai import OpenAI
    random.seed(0)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader", "-l", "2"], stdout=open(out / "gpu.csv", "a"))
    tok, model = load_causal_policy(args) if args.causal else load_policy(args)
    if args.share_impl == "train.prefix_share_llama":
        import train.prefix_share_llama as psl
        psl.USE_CKPT = bool(args.share_ckpt)
    gen_grpo.PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW(gen_grpo.lora_params(model, "policy"), lr=args.lr, weight_decay=0.0)
    servers = [Server(u) for u in (args.policy_url or args.vllm_url).split(",")]   # 策略采样、热加载；逗号分隔 = 两张卡各一个同样的服务，请求分摊
    temp = json.load(open(args.critic_report, encoding="utf-8"))["temperature"]
    sat_temp = json.load(open(args.sat_report, encoding="utf-8"))["temperature"] if args.sat_model else 1.0
    refuse_temp = json.load(open(args.refuse_report, encoding="utf-8"))["temperature"] if args.refuse_model else 1.0
    rw = AnswerRewarder([OpenAI(base_url=u, api_key=vllm_key(), timeout=600, max_retries=3) for u in args.vllm_url.split(",")], args.critic, temp, ThreadPoolExecutor(args.rm_concurrency), args.sat_model, args.sat_kind, sat_temp, args.sat_w,
                        args.refuse_model, refuse_temp, args.refuse_w)
    rw.group_pool = ThreadPoolExecutor(args.rm_concurrency)
    if args.proc_producer:   # 采样 / 打分 / 快评放子进程（训练进程独占 GIL）
        prod = ProcProducer(args, out)
        qe = EvalProxy(prod) if args.eval_k > 0 else None
    else:
        prod = Producer(args, tok, servers, rw)
        qe = QuickEval(args, tok, servers, rw, out) if args.eval_k > 0 else None
    lora_out = Path(args.push_dir)   # 热加载的 LoRA 放内存盘：共享盘写 5 秒 + 首次加载 4 秒，内存盘合计 0.5 秒（2026-11-02 实测）
    version, first = 0, 1
    if args.resume:
        st = load_state(model, opt, rw, prod, Path(args.resume))
        version, first = st["version"], st["step"] + 1
        push_all(model, lora_out, servers, version, args.keep_versions, args.version_prefix)
        prod.version = version
        if qe:   # 接着训：基线从第 0 步样本恢复，补评切换时没评完的检查点
            qe.first = first_from_samples(out / f"quick_samples_{args.version_prefix}s0.jsonl", args.eval_k) if not args.proc_producer else True
            done = {json.loads(l)["step"] for l in open(out / "quick_eval.jsonl", encoding="utf-8")}
            ck = out / "ckpt" / f"step{st['step']}"
            if ck.exists() and st["step"] not in done:
                qe.submit(st["step"], ck)
    elif qe and args.eval_start:
        qe.submit(0, None)
    if args.proc_producer:
        prod.start()
    else:
        threading.Thread(target=prod.run, daemon=True).start()
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    trip = TripWire(kl_max=args.trip_kl) if args.trip else None
    if trip and args.resume:   # 报警基线从已有日志恢复，只认恢复点之前的行
        for l in open(out / "log.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if "t_step" in r and r["step"] < first:
                trip.hist.append(r)
    pusher = AsyncPusher(servers, prod, args.keep_versions, args.version_prefix) if args.async_push else None
    t_start, last_hits = time.time(), cache_hits(args.vllm_url)
    for step in range(first, args.steps + 1):
        t0 = time.time()
        while True:
            batch = prod.get()
            if version - batch["version"] <= args.max_stale:
                break
            log.write(json.dumps({"step": step, "dropped_stale": version - batch["version"]}) + "\n")
        rec = {"step": step, "gen_version": batch["version"], "t_wait": round(time.time() - t0, 1), **batch_stats(batch)}
        save_samples(batch, step, out)
        rec.update(train_step(model, opt, batch, version, args))
        version += 1
        t1 = time.time()
        if pusher:   # 后台推送：t_push 是主线程等的时间，t_push_bg 是上一次后台上传 + 加载用的时间
            pusher.push(model, lora_out, version)
            rec["t_push_bg"] = round(pusher.last_bg, 1)
        else:
            push_all(model, lora_out, servers, version, args.keep_versions, args.version_prefix)
            prod.version = version
        hits = cache_hits(args.vllm_url)
        rec.update({"cache_hit": round((hits[0] - last_hits[0]) / max(1.0, hits[1] - last_hits[1]), 3), "t_push": round(time.time() - t1, 1), "t_step": round(time.time() - t0, 1), "elapsed_min": round((time.time() - t_start) / 60, 1)})
        last_hits = hits
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps({k: rec.get(k) for k in ("step", "reward", "p_mean", "p_max", "answer_flag", "sat", "sat_low", "ok_rate", "chars", "copy", "refusal", "kl", "clip_frac", "grad_norm", "policy_ref_absdiff", "staleness", "cache_hit", "t_gen", "t_rm", "t_train", "t_step")}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ck = out / "ckpt" / f"step{step}"
            model.save_pretrained(str(ck), selected_adapters=["policy"])
            if qe:
                qe.submit(step, ck)
        if step % args.state_every == 0:
            if pusher:
                pusher.wait()   # 存状态时服务上的版本要和 version 一致，续训才对得上
            save_state(model, opt, rw, prod, step, version, out, args.keep_states)
        reasons = trip.update(rec) if trip else []
        if reasons:
            if pusher:
                pusher.wait()
            d = save_state(model, opt, rw, prod, step, version, out, args.keep_states)
            (out / "TRIP.json").write_text(json.dumps({"step": step, "reasons": reasons, "state": str(d)}, ensure_ascii=False, indent=1), encoding="utf-8")
            print("GRPO_TRIPPED", json.dumps({"step": step, "reasons": reasons}, ensure_ascii=False), flush=True)
            break
    if pusher:
        pusher.wait()
    if args.proc_producer:
        prod.shutdown()
    else:
        prod.stop = True
        if qe:
            qe.wait()
    smi.terminate()
    print("GRPO_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--init", default="")
    ap.add_argument("--prompts", default="outputs/rt/ansgrpo/train.jsonl")
    ap.add_argument("--dev", default="outputs/rt/ansgrpo/dev.jsonl")
    ap.add_argument("--vllm_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--critic", default="critic")
    ap.add_argument("--critic_report", default="outputs/rt/rm/rm1__rm1_gold_9b/eval/report.json")
    ap.add_argument("--shadow", default="shadow", help="对照 critic（rm1-9b）：只在评测里打分，不进奖励；空 = 训练中不打（MiniCPM 训完再用 9B 补打）")
    ap.add_argument("--sat_model", default="", help="满意度护栏（vLLM 上的 LoRA 名）；空 = 不用")
    ap.add_argument("--sat_kind", choices=["pair", "abs"], default="pair")
    ap.add_argument("--sat_report", default="outputs/rt/rm/sat_pair/eval/report.json")
    ap.add_argument("--sat_w", type=float, default=0.5)
    ap.add_argument("--refuse_model", default="", help="拒答 RM（vLLM 上的 LoRA 名）；空 = 拒答一律 0 分")
    ap.add_argument("--refuse_report", default="outputs/rt/rm/refuse/eval/report.json")
    ap.add_argument("--refuse_w", type=float, default=0.4, help="拒答的奖励 = refuse_w × P(可以拒答)")
    # 奖励形状 + 分步奖励（2026-11-05）。默认 legacy + sent_beta 0 = 旧行为
    ap.add_argument("--reward_shape", choices=["legacy", "count"], default="legacy", help="legacy：1 − 0.5 平均 P − 0.5 最大 P；count：去同组底噪后计数（ans_reward.count_reward）")
    ap.add_argument("--count_k", type=float, default=3.0, help="count：期望错句数除以 K（多少句错扣满）")
    ap.add_argument("--count_ratio", type=float, default=0.0, help="count：> 0 时 K = max(count_min, ratio × 参考回答句数)，没有参考回答用 count_k")
    ap.add_argument("--count_min", type=float, default=2.0)
    ap.add_argument("--lam_any", type=float, default=0.5, help="count：P(至少一句超出底噪) 的权重")
    ap.add_argument("--lam_cnt", type=float, default=0.5, help="count：期望错句数 / K 的权重")
    ap.add_argument("--sigma_floor", type=float, default=0.1, help="句子层优势除以的同组标准差下限")
    ap.add_argument("--sent_beta", type=float, default=0.0, help="> 0：句子层优势 × beta 加到这句的 token 上（0 = 整段一个优势，旧行为）")
    ap.add_argument("--sent_mode", choices=["neg", "center"], default="neg", help="neg：只罚超出底噪的句子；center：干净句子拿正值")
    ap.add_argument("--sent_gamma", type=float, default=0.0, help="> 0：后一句的负优势按 gamma 传给前一句")
    ap.add_argument("--shadow_temp", type=float, default=1.3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max_stale", type=int, default=2)
    ap.add_argument("--keep_versions", type=int, default=3)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--share", type=int, default=1)
    ap.add_argument("--share_mb", type=int, default=4)
    ap.add_argument("--share_ckpt", type=int, default=1, help="Llama 版 prompt 共享是否逐层梯度检查点（0 = 不重算，快，占显存多）")
    ap.add_argument("--share_impl", default="train.prefix_share", help="prompt 共享的实现：Qwen3.5 用 train.prefix_share，Llama 结构（MiniCPM5-2B）用 train.prefix_share_llama")
    ap.add_argument("--version_prefix", default="")
    ap.add_argument("--push_dir", default="/dev/shm/ans_lora")
    ap.add_argument("--pipeline", type=int, default=0, help="1：每个 prompt 生成完立刻打分（生成和打分重叠）；0：整批生成完再整批打分（原来的做法）")
    ap.add_argument("--proc_producer", type=int, default=0, help="1：采样 / 打分 / 快评放子进程，训练进程独占 GIL（2026-11-05 定位：同进程线程抢锁，t_train 24 秒 vs 单独 10.4 秒）")
    ap.add_argument("--async_push", type=int, default=0, help="1：LoRA 在后台推送（双机模式用；单机默认 0，行为不变）")
    ap.add_argument("--rm_concurrency", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_k", type=int, default=2)
    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--eval_concurrency", type=int, default=8)
    ap.add_argument("--eval_start", type=int, default=1)
    ap.add_argument("--state_every", type=int, default=25)
    ap.add_argument("--keep_states", type=int, default=12)
    ap.add_argument("--resume", default="")
    ap.add_argument("--trip", type=int, default=1)
    # 2026-11-05：三档 critic 200 步时 ref 组第 98 步 KL 窗口均值 0.101 触发（长度 / 复制 / 拒答 / ok_rate 都正常）→ 可从环境变量放宽，不改正在跑的脚本
    ap.add_argument("--trip_kl", type=float, default=float(os.environ.get("TRIP_KL", "0.10")))
    ap.add_argument("--make_init", action="store_true")
    ap.add_argument("--policy_url", default="", help="策略模型单独的 vLLM（MiniCPM 在训练卡上）；空 = 和 RM 同一个服务")
    ap.add_argument("--causal", type=int, default=0, help="1 = 纯文本 CausalLM 策略（MiniCPM5-2B）")
    ap.add_argument("--sys_qa", type=int, default=0, help="1 = QA 加系统提示（ragtruth.ans_data.SYSTEM_QA）")
    ap.add_argument("--template", default="")
    args = ap.parse_args()
    (make_init_causal(args) if args.causal else make_init(args)) if args.make_init else run(args)


if __name__ == "__main__":
    main()
