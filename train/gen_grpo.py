"""生成器 GRPO：两张卡，训练卡只训 LoRA，推理卡一个 vLLM 服务同时放生成器 LoRA 和三个 RM LoRA。

  GPU1：vllm serve <9B 基座> --enable-lora --max-lora-rank 32 --lora-modules rm1=… rm2=… rm3=… policy_v0=<起点 LoRA>
        （VLLM_ALLOW_RUNTIME_LORA_UPDATING=True，训练中热加载新版本生成器 LoRA）
  GPU0：python -m train.gen_grpo --init <起点 LoRA> --out <目录> [--mode async|sync] [--steps N]
        python -m train.gen_grpo --check --init … --seed_adapter … --dpo_adapter …   # 正确性检查

每一步：生产者用当前最新版本的生成器 LoRA 给一批 prompt 各采 n 条（同时拿到采样时每个 token 的 logprob），
在同一服务上用三个 RM 打分（train.gen_reward），算组内优势值（同类型基线）；训练者取一批，算当前策略和参考策略
（起点 LoRA，冻结副本）的 logprob，PPO 截断的重要性比值 × 优势值 − β·KL，更新一次，保存 LoRA 推给推理卡。
async：生产者一直往前生成，队列最多 max_stale 批，落后超过 max_stale 个版本的批丢掉重采（两张卡并行，参数
有一点延迟没关系）；sync：训练者等这一批生成完再训（对照，用来测并行的收益）。
每步记一行 log.jsonl（奖励各项、合格率、类型占比、硬规则拦下的原因、落后版本数、截断比例、KL、耗时）；
每 save_every 步存一个检查点，训练后做阶段性评测。
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import queue
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPTS = "outputs/rt/realgen/grpo/prompts.jsonl"


def vllm_key() -> str:
    """推理服务的 API 密钥。双机模式下推理机的 vLLM 经 AutoDL 对外域名访问，必须带密钥；单机默认 "x"（服务不校验）。"""
    import os
    return os.environ.get("VLLM_API_KEY", "x")


def vllm_headers() -> dict:
    return {"Authorization": f"Bearer {vllm_key()}"}


def lora_sync_url(server_url: str = "") -> str:
    """双机模式：推理机上的 LoRA 同步服务（scripts/lora_sync.py）。设了就不直接让 vLLM 读训练机路径，
    而是让同步服务把共享盘上的 LoRA 拷到推理机内存盘再热加载（每步存共享盘，再加载到内存）。
    多台推理机：RM_URL（或 POLICY_URL）和 LORA_SYNC_URL 都是逗号分隔、顺序一一对应，按这个 vLLM 服务的地址找它自己的同步服务。"""
    import os
    syncs = [s.strip().rstrip("/") for s in os.environ.get("LORA_SYNC_URL", "").split(",") if s.strip()]
    # 策略服务的地址：设了 POLICY_URL（打分在单独的打分机上）就按它对，否则策略和打分同一组服务，按 RM_URL 对
    urls = [u.strip().rstrip("/") for u in (os.environ.get("POLICY_URL") or os.environ.get("RM_URL", "")).split(",") if u.strip()]
    if server_url and urls and server_url.rstrip("/") not in urls:
        return ""   # 本机的策略服务（2B 策略 × 9B 打分：策略在训练机上采样，打分在推理机上）：直接热加载，不走同步服务
    if len(syncs) <= 1:
        return syncs[0] if syncs else ""
    if server_url.rstrip("/") in urls and len(urls) == len(syncs):
        return syncs[urls.index(server_url.rstrip("/"))]
    raise RuntimeError(f"LORA_SYNC_URL has {len(syncs)} entries but server {server_url} not matched in RM_URL")


def strip_visual_inplace(path: str) -> None:
    """Qwen3.5-9B 按视觉语言结构加载时，PEFT 也给视觉塔挂了 LoRA；vLLM 只接受语言模型的模块，带视觉参数的 LoRA 加载会报错。
    推送前就地删掉（同 scripts/strip_visual_lora.py；纯文本任务用不到视觉塔）。MiniCPM5-2B 没有视觉参数，原样不动。"""
    import json
    from safetensors.torch import load_file, save_file
    f = Path(path) / "adapter_model.safetensors"
    if not f.exists():
        return
    w = load_file(str(f))
    keep = {k: v for k, v in w.items() if ".visual." not in k and not k.startswith("visual.")}
    if len(keep) == len(w):
        return
    save_file(keep, str(f))
    c = Path(path) / "adapter_config.json"
    cfg = json.loads(c.read_text())
    cfg["target_modules"] = sorted({k.split(".lora_")[0].split(".")[-1] for k in keep if ".lora_" in k})
    c.write_text(json.dumps(cfg, indent=1))


class Server:
    """推理卡上的 vLLM：生成（completions，拿 token id 和 logprob）+ 热加载 / 卸载 LoRA。"""

    def __init__(self, url: str):
        from openai import OpenAI
        self.url = url.rstrip("/")
        self.client = OpenAI(base_url=self.url, api_key=vllm_key(), timeout=1800, max_retries=3)

    def sample(self, model: str, prompt: str | list[int], n: int, max_tokens: int, temperature: float, priority: int = 0) -> list[dict]:
        # top_p / top_k 显式关掉：vLLM 默认沿用模型 generation_config（MiniCPM5 是 top_p 0.95），截断采样和策略梯度假设的完整分布不一致
        r = self.client.completions.create(model=model, prompt=prompt, n=n, max_tokens=max_tokens, temperature=temperature, top_p=1.0,
                                           logprobs=1, extra_body={"return_tokens_as_token_ids": True, "top_k": -1, **({"priority": priority} if priority else {})})
        out = []
        for c in r.choices:
            lp = c.logprobs
            ids = [int(t.split(":", 1)[1]) for t in lp.tokens]
            out.append({"text": c.text, "ids": ids, "old_lp": [float(x) for x in lp.token_logprobs], "finish": c.finish_reason})
        return out

    def ref_logprobs(self, model: str, prompt_ids: list[int], comp_ids: list[int]) -> list[float]:
        """给定 token 序列，返回 completion 每个 token 在 model 下的 logprob（vLLM prompt_logprobs，只做一次前向）。
        参考策略就是 policy_v0，本来就挂在推理卡上，挪到推理卡算可以给训练卡减负。"""
        import requests
        r = requests.post(self.url + "/completions", headers=vllm_headers(), json={"model": model, "prompt": prompt_ids + comp_ids, "max_tokens": 1,
                                                            "temperature": 0.0, "prompt_logprobs": 0}, timeout=1800)
        r.raise_for_status()
        pl = r.json()["choices"][0]["prompt_logprobs"]
        P = len(prompt_ids)
        return [float(pl[P + k][str(t)]["logprob"]) for k, t in enumerate(comp_ids)]

    def ref_logprobs_batch(self, model: str, prompt_ids: list[int], comps: list[list[int]]) -> list[list[float]]:
        """同一个 prompt 的几条 completion 一次请求（双机模式每个请求经对外域名多约 30 ms，256 个单条请求 35 秒）。
        vLLM completions 的 prompt 可以是多个 token 序列，choices 按 index 对应。"""
        import requests
        r = requests.post(self.url + "/completions", headers=vllm_headers(), json={"model": model, "prompt": [prompt_ids + c for c in comps], "max_tokens": 1,
                                                                                    "temperature": 0.0, "prompt_logprobs": 0}, timeout=1800)
        r.raise_for_status()
        ch = sorted(r.json()["choices"], key=lambda c: c["index"])
        P = len(prompt_ids)
        return [[float(c["prompt_logprobs"][P + k][str(t)]["logprob"]) for k, t in enumerate(comp)] for c, comp in zip(ch, comps)]

    def load_lora(self, name: str, path: str) -> None:
        import requests
        strip_visual_inplace(path)
        sync = lora_sync_url(self.url)
        if sync:   # 双机：LoRA 交给推理机上的同步服务（scripts/lora_sync.py），在推理机内存盘热加载
            import os
            if os.environ.get("LORA_SYNC_MODE", "upload") == "upload":   # 直接上传，不经共享盘
                import io
                import tarfile
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:") as tf:
                    for f in sorted(Path(path).iterdir()):
                        if f.is_file():
                            tf.add(str(f), arcname=f.name)
                r = requests.put(f"{sync}/push/{name}", headers=vllm_headers(), data=buf.getvalue(), timeout=900)
            else:   # 经共享盘（跨区上传慢时用：实测 westc → westd 只有 6 MB/s，192 MB 要 20 秒；共享盘写 + 读约 3 秒）
                fs_dir = os.environ.get("LORA_SYNC_FS_DIR", "")
                if fs_dir and not str(path).startswith(fs_dir):   # 主线程只存本机内存盘，这里（后台线程）再拷到共享盘，不挡训练
                    import shutil
                    dst = Path(fs_dir) / name
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(path, dst)
                    path = str(dst)
                path = str(Path(path).resolve())   # 推理机按绝对路径读（两台机器的共享盘挂在同一位置）
                os.sync()   # 共享盘写完落盘，推理机才读得到完整文件
                r = requests.post(sync + "/sync", headers=vllm_headers(), json={"name": name, "src": str(path)}, timeout=900)
            if r.status_code != 200:
                raise RuntimeError(f"lora sync {name}: {r.status_code} {r.text[:300]}")
            self.last_sync = r.json()
            return
        body = {"lora_name": name, "lora_path": path}
        r = requests.post(self.url + "/load_lora_adapter", headers=vllm_headers(), json=body, timeout=600)
        if r.status_code == 400 and "already been loaded" in r.text:   # 回退 / 接着训时同名版本还在服务上
            # 不能用 load_inplace：这个标记会留在请求上，之后每次生成都重新加载 LoRA，生成慢 4–5 倍（踩过）
            self.unload_lora(name)
            r = requests.post(self.url + "/load_lora_adapter", headers=vllm_headers(), json=body, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter {name}: {r.status_code} {r.text[:300]}")

    def unload_lora(self, name: str) -> None:
        import requests
        sync = lora_sync_url(self.url)
        if sync:
            requests.post(sync + "/unload", headers=vllm_headers(), json={"name": name}, timeout=600)
            import os
            fs_dir = os.environ.get("LORA_SYNC_FS_DIR", "")
            if fs_dir:   # 共享盘上的中转副本一起删（每版 192 MB，300 步不删会堆 57 GB）
                import shutil
                shutil.rmtree(Path(fs_dir) / name, ignore_errors=True)
            return
        requests.post(self.url + "/unload_lora_adapter", headers=vllm_headers(), json={"lora_name": name}, timeout=600)


def chat_prompt(tok, text: str, system: str = "") -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


class Producer:
    def __init__(self, args, tok, server: Server, rewarder):
        self.a, self.tok, self.srv, self.rw = args, tok, server, rewarder
        rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
        self.rows = rows
        self.pos = 0
        self.version = 0
        self.q: queue.Queue = queue.Queue(maxsize=max(1, args.max_stale - 1))   # 排队 1 批 + 正在生成 1 批：最多落后 max_stale 个版本
        self.stop = False
        self.idle = threading.Event()
        self.gen_pool = ThreadPoolExecutor(args.prompts_per_step)
        self.rm_pool = ThreadPoolExecutor(args.rm_concurrency)

    def next_rows(self) -> list[dict]:
        out = []
        for _ in range(self.a.prompts_per_step):
            out.append(self.rows[self.pos % len(self.rows)])
            self.pos += 1
        return out

    def make_batch(self) -> dict:
        from train.gen_reward import group_advantages, place_diversity
        src = getattr(self, "version_src", None)   # 子进程模式：版本号在主进程和子进程共享的整数里
        v = src.value if src is not None else self.version
        rows = self.next_rows()
        t0 = time.time()
        prompts = [chat_prompt(self.tok, r["prompt"]) for r in rows]
        name = "policy_v0" if v == 0 else f"{self.a.version_prefix}policy_v{v}"
        # 发 token id、训练侧用同一份（MiniCPM 的分词器默认加开头 <s>，vLLM 自己分词会加、训练侧原来不加：ans_grpo 第 1 步截断比例 22%）
        ids = [self.tok(p)["input_ids"] for p in prompts]
        if getattr(self.a, "pipeline", 0):   # 每个 prompt 生成完立刻提交打分，生成和打分重叠（采样端优化）
            def one(k):
                g = self.srv.sample(name, ids[k], self.a.n, self.a.max_tokens, self.a.temperature)
                return g, [self.rm_pool.submit(self.rw.score, x["text"], rows[k]) for x in g], time.time()
            res = list(self.gen_pool.map(one, range(len(rows))))
            gens, futs = [r[0] for r in res], [r[1] for r in res]
            t1 = max(r[2] for r in res)
        else:
            gens = list(self.gen_pool.map(lambda p: self.srv.sample(name, p, self.a.n, self.a.max_tokens,
                                                                     self.a.temperature), ids))
            t1 = time.time()
            futs = [[self.rm_pool.submit(self.rw.score, g["text"], r) for g in gs] for r, gs in zip(rows, gens)]
        samples = []
        for r, p, gs, fs in zip(rows, prompts, gens, futs):
            scores = [f.result() for f in fs]
            if self.a.div_mode == "place":
                if hasattr(self.rw, "combine"):
                    from train.gen_reward import place_diversity_target
                    place_diversity_target(scores, self.rw)
                else:
                    place_diversity(scores, self.rw.w)
            if hasattr(self.rw, "group_post"):   # 组内后处理（v2sev rank：组内相对难度）
                self.rw.group_post(scores)
            adv = group_advantages([s["reward"] for s in scores], [s.get("type") for s in scores])
            pids = self.tok(p)["input_ids"]
            for g, s, a in zip(gs, scores, adv):
                samples.append({"prompt_id": r["id"], "prompt_ids": pids, **g, **{k: v_ for k, v_ in s.items()}, "adv": a})
        t2 = time.time()
        if self.a.ref_on_vllm:   # 只给要训练的样本（优势值非 0）算参考 logprob
            need = [x for x in samples if abs(x["adv"]) > 1e-6 and x["ids"]]
            if getattr(self.a, "ref_batch", 0):   # 按 prompt 合并成一次请求
                groups = collections.defaultdict(list)
                for x in need:
                    groups[x["prompt_id"]].append(x)
                gl = list(groups.values())
                outs = list(self.rm_pool.map(lambda g: self.srv.ref_logprobs_batch("policy_v0", g[0]["prompt_ids"], [x["ids"] for x in g]), gl))
                for g, rls in zip(gl, outs):
                    for x, rl in zip(g, rls):
                        x["ref_lp"] = rl
            else:
                refs = list(self.rm_pool.map(lambda x: self.srv.ref_logprobs("policy_v0", x["prompt_ids"], x["ids"]), need))
                for x, rl in zip(need, refs):
                    x["ref_lp"] = rl
        return {"version": v, "samples": samples, "t_gen": t1 - t0, "t_rm": t2 - t1, "t_ref": time.time() - t2, "pos": self.pos}

    def run(self) -> None:   # async：一直往前生成
        while not self.stop:
            b = self.make_batch()
            self.idle.set()   # 这批生成完了，在等训练者取：推理卡空着，快评可以用
            while not self.stop:
                try:
                    self.q.put(b, timeout=5)
                    break
                except queue.Full:
                    continue
            self.idle.clear()
        self.idle.set()


def build_rewarder(args):
    from openai import OpenAI
    from train.gen_reward import Rewarder, TargetRewarder, load_temps
    cl = OpenAI(base_url=args.vllm_url, api_key=vllm_key(), timeout=600, max_retries=3)
    if args.reward == "v2sev":   # 生成器 v2：sev 主奖励，自然度 / critic 门槛（train.gen_reward_v2）
        from train.gen_reward_v2 import V2SevRewarder
        return V2SevRewarder(cl, load_temps(args.temps), nat_gate=args.nat_gate, crit=bool(args.crit_gate), crit_lo=args.crit_lo, crit_hi=args.crit_hi,
                             fail_scale=args.fail_scale, shape=args.v2_shape, err_gate=args.err_gate,
                             err_gate_start=args.err_gate_start, hard_shape=args.hard_shape, hard_base=args.hard_base, warm_samples=args.err_gate_warm * args.n * args.prompts_per_step)
    if args.reward == "ops":   # 按操作清单造错的生成器：三个阶段逐步加难（train.gen_reward_ops）
        from train.gen_reward_ops import OpsRewarder
        return OpsRewarder(cl, load_temps(args.temps), stage=args.ops_stage, err_gate=args.ops_err_gate, nat_gate=args.nat_gate, op_gate=args.op_gate,
                           min_sim=args.ops_min_sim, fail_scale=args.fail_scale, ref=args.ops_ref, w_hard=args.ops_w_hard)
    if args.reward == "target":
        return TargetRewarder(cl, load_temps(args.temps), w_div=args.w_div, w_corr=args.w_corr, w_crit=args.w_crit, gate=args.gate, type_floor=args.type_floor,
                              err_mode=args.err_mode, shape=args.shape, w_real2=args.w_real2, w_hard2=args.w_hard2, crit_max=args.crit_max, crit_floor=args.crit_floor,
                              rm2_ens=[m for m in args.rm2_ens.split(",") if m], gate_lo=args.gate_lo, min_sim=args.min_sim, base_fail=args.base_fail, obvious_pen=args.obvious_pen,
                              blind_bonus=getattr(args, "blind_bonus", 0.0), blind_thr=getattr(args, "blind_thr", 0.5), blind_lo=getattr(args, "blind_lo", 0.0))
    return Rewarder(cl, load_temps(args.temps), args.w_hard, args.w_real, args.w_div, args.type_floor)


def _gen_producer_proc(args, out: str, start_pos: int, rw_state, vval, bq, cmd, ack) -> None:
    """子进程（训练慢，定位到是锁）：采样 / 打分 / 快评都在这里，训练进程独占 GIL。
    奖励的滑动统计在这里，主进程存状态时用 state 命令要。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    server = Server(args.vllm_url)
    rw = build_rewarder(args)
    if rw_state:
        rw.load_state_dict(rw_state)
    prod = Producer(args, tok, server, rw)
    prod.pos, prod.q, prod.version_src = start_pos, bq, vval
    qe = QuickEval(args, tok, server, rw, Path(out), prod.idle) if args.eval_k > 0 else None

    def run_prod():
        try:
            prod.run()
        except BaseException as e:  # noqa: BLE001
            bq.put({"__dead__": repr(e)[:500]})
    threading.Thread(target=run_prod, daemon=True).start()
    while True:
        c = cmd.get()
        if c[0] == "eval" and qe:
            qe.submit(c[1], Path(c[2]) if c[2] else None)
        elif c[0] == "first" and qe:
            qe.first = c[1]
        elif c[0] == "state":
            ack.put(("state", rw.state_dict()))
        elif c[0] == "stop":
            prod.stop = True
            if qe:
                qe.wait()
            ack.put(("done", None))
            return


class GenProcProducer:
    """主进程这边的代理：get / version / pos；rw 代理（state_dict 找子进程要）；快评代理。"""

    def __init__(self, args, out: Path):
        import multiprocessing as mp
        self.ctx, self.a, self.out = mp.get_context("spawn"), args, out
        self._v = self.ctx.Value("i", 0)
        self.pos, self.p, self.rw_state = 0, None, None
        self.q = self.ctx.Queue(maxsize=max(1, args.max_stale - 1))
        self.cmd, self.ack = self.ctx.Queue(), self.ctx.Queue()
        self.target = _gen_producer_proc   # 测试里换成假的子进程

    @property
    def version(self) -> int:
        return self._v.value

    @version.setter
    def version(self, v: int) -> None:
        self._v.value = int(v)

    def start(self) -> None:
        self.p = self.ctx.Process(target=self.target, args=(self.a, str(self.out), self.pos, self.rw_state, self._v, self.q, self.cmd, self.ack), daemon=True)
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

    def ask(self, c, want: str, timeout: float = 600):
        self.cmd.put(c)
        while True:
            k, v = self.ack.get(timeout=timeout)
            if k == want:
                return v

    def shutdown(self) -> None:
        try:
            self.ask(("stop",), "done", timeout=7200)
        except queue.Empty:
            pass
        self.p.join(60)
        if self.p.is_alive():
            self.p.terminate()


class RwProxy:
    def __init__(self, prod: GenProcProducer):
        self.prod = prod

    def state_dict(self) -> dict:
        return self.prod.ask(("state",), "state") if self.prod.p is not None else (self.prod.rw_state or {})

    def load_state_dict(self, st: dict) -> None:   # 续训：启动子进程时带过去
        self.prod.rw_state = st


class EvalProxy:
    def __init__(self, prod: GenProcProducer):
        self.prod = prod

    def submit(self, step: int, path) -> None:
        self.prod.cmd.put(("eval", step, str(path) if path else ""))

    @property
    def first(self):
        return None

    @first.setter
    def first(self, v) -> None:
        self.prod.cmd.put(("first", v))

    def wait(self) -> None:   # 由 shutdown 等
        pass


def lora_params(model, name: str):
    return [p for n, p in model.named_parameters() if f".{name}." in n and "lora_" in n]


PAD_ID = 0


def batch_logprobs(model, prompt_ids: list[int], comps: list[list[int]]):
    """同一个 prompt 的几条 completion 一起前向（右侧补齐；因果注意力下补齐不影响前面的 token）。
    → [B, Cmax] 的 logprob 和 mask。"""
    import torch
    cmax = max(len(c) for c in comps)
    ids = torch.full((len(comps), len(prompt_ids) + cmax), PAD_ID, dtype=torch.long, device="cuda")
    att = torch.zeros_like(ids)
    tgt = torch.full((len(comps), cmax), PAD_ID, dtype=torch.long, device="cuda")
    mask = torch.zeros((len(comps), cmax), dtype=torch.bool, device="cuda")
    for b, c in enumerate(comps):
        n = len(prompt_ids) + len(c)
        ids[b, :n] = torch.tensor(prompt_ids + c, device="cuda")
        att[b, :n] = 1
        tgt[b, :len(c)] = torch.tensor(c, device="cuda")
        mask[b, :len(c)] = True
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=cmax + 1).logits[:, :-1]
    # 逐条转 float 再算 logsumexp：整批一起转 float（B × Cmax × 24.8 万词表）会多出好几 GB，micro_batch 4 时 OOM
    lp = torch.stack([logits[b].float().gather(1, tgt[b][:, None])[:, 0] - torch.logsumexp(logits[b].float(), -1)
                      for b in range(len(comps))])
    return lp, mask


def seq_logprobs(model, prompt_ids: list[int], comp_ids: list[int]):
    import torch
    ids = torch.tensor([prompt_ids + comp_ids], device="cuda")
    C = len(comp_ids)
    logits = model(input_ids=ids, logits_to_keep=C + 1).logits[0, :-1].float()   # 预测 completion 每个 token 的位置
    lp = torch.log_softmax(logits, -1)
    return lp.gather(1, torch.tensor(comp_ids, device="cuda")[:, None])[:, 0]


def load_adapters_strict(base, init: str):
    """挂 policy / ref 两个 LoRA；PEFT 报 missing adapter keys（参数名对不上，等于从零起训）时直接报错，不带着警告往下训。"""
    import warnings
    from peft import PeftModel
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, init, adapter_name="policy", is_trainable=True)
        model.load_adapter(init, adapter_name="ref", is_trainable=False)
    bad = [str(x.message)[:300] for x in w if "missing adapter keys" in str(x.message).lower()]
    if bad:
        raise RuntimeError(f"LoRA init {init} does not match the model (missing adapter keys): {bad[0]}")
    for x in w:
        warnings.warn_explicit(x.message, x.category, x.filename, x.lineno)
    return model


def load_policy(args):
    if getattr(args, "causal", 0):   # 纯文本 CausalLM（MiniCPM5-2B）：同 ans_grpo
        from train.ans_grpo import load_causal_policy
        return load_causal_policy(args)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model = load_adapters_strict(base, args.init)
    model.set_adapter("policy")
    for p in lora_params(model, "ref"):
        p.requires_grad_(False)
    for p in lora_params(model, "policy"):
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()   # from_pretrained 默认 eval：HF 的梯度检查点只在 training 模式生效，不切会 OOM（冒烟测试踩过）
    return tok, model


def _chunk_loss(ch: list[dict], lp, mask, ref, n_tok: int, args, st: collections.Counter):
    """一小批同 prompt 样本的 PPO 截断 + k3 KL 损失（除以整步 token 数），顺便累计统计。"""
    import torch
    W = lp.shape[1]
    pad = lambda xs: torch.tensor([x + [0.0] * (W - len(x)) for x in xs], device="cuda")
    old = pad([s["old_lp"] for s in ch])
    if not torch.is_tensor(ref):
        ref = pad(ref)
    # adv 可以是一条样本一个数（整段一个优势），也可以是每个 token 一个数（分步奖励，ans_grpo --sent_beta）
    adv = pad([s["adv"] if isinstance(s["adv"], list) else [float(s["adv"])] * len(s["old_lp"]) for s in ch])
    ratio = torch.exp(lp - old)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv)
    kl = torch.exp(ref - lp) - (ref - lp) - 1   # k3 估计，非负
    loss = ((-surr + args.beta * kl) * mask).sum() / n_tok
    m = mask.float()
    st["kl"] += (kl.detach() * m).sum().item()
    st["clip"] += (((ratio.detach() - 1).abs() > args.clip) & mask).sum().item()
    st["vllm_hf_absdiff"] += ((lp.detach() - old).abs() * m).sum().item()
    st["ref_hf_absdiff"] += ((lp.detach() - ref).abs() * m).sum().item()
    return loss


def _use_policy(model) -> None:
    model.set_adapter("policy")
    for p in lora_params(model, "ref"):
        p.requires_grad_(False)


def _accum_grads(model, groups: dict, n_tok: int, args, st: collections.Counter) -> None:
    """整段拼接（原做法）：同一 prompt 的样本按 micro_batch 条一起前向 / 反向。"""
    import torch
    chunks = [g[k:k + args.micro_batch] for g in groups.values() for k in range(0, len(g), args.micro_batch)]
    for ch in chunks:
        comps = [s["ids"] for s in ch]
        if all("ref_lp" in s for s in ch):
            ref = [s["ref_lp"] for s in ch]
        else:
            with torch.no_grad():
                model.set_adapter("ref")
                ref, _ = batch_logprobs(model, ch[0]["prompt_ids"], comps)
            _use_policy(model)
        lp, mask = batch_logprobs(model, ch[0]["prompt_ids"], comps)
        _chunk_loss(ch, lp, mask, ref, n_tok, args, st).backward()


def _accum_grads_shared(model, groups: dict, n_tok: int, args, st: collections.Counter) -> None:
    """prompt 共享（train.prefix_share）：每个 prompt 只过一遍，completion 每 share_mb 条一批接在后面。"""
    import importlib
    import torch
    ps = importlib.import_module(getattr(args, "share_impl", "train.prefix_share"))   # Qwen3.5 版；Llama 结构（MiniCPM5-2B）用 train.prefix_share_llama
    backward_prompt, completion_logprobs, leaves, prompt_pass = ps.backward_prompt, ps.completion_logprobs, ps.leaves, ps.prompt_pass
    mb = args.share_mb
    for g in groups.values():
        pids = g[0]["prompt_ids"]
        chunks = [g[k:k + mb] for k in range(0, len(g), mb)]
        refs = {}
        if not all("ref_lp" in s for s in g):
            with torch.no_grad():
                model.set_adapter("ref")
                rp = prompt_pass(model, pids, grad=False)
                for k, ch in enumerate(chunks):
                    refs[k], _ = completion_logprobs(model, rp, [s["ids"] for s in ch], PAD_ID)
                del rp
            _use_policy(model)
        pre = prompt_pass(model, pids, grad=True)
        leaf = leaves(pre)
        for k, ch in enumerate(chunks):
            lp, mask = completion_logprobs(model, leaf, [s["ids"] for s in ch], PAD_ID)
            ref = refs[k] if k in refs else [s["ref_lp"] for s in ch]
            _chunk_loss(ch, lp, mask, ref, n_tok, args, st).backward()
        backward_prompt(pre, leaf)
        del pre, leaf


def has_adv(s: dict) -> bool:
    """这条样本有没有梯度：标量优势非 0，或逐 token 优势里有非 0 的。"""
    a = s["adv"]
    return any(abs(x) > 1e-6 for x in a) if isinstance(a, list) else abs(a) > 1e-6


def train_step(model, opt, batch: dict, cur_version: int, args) -> dict:
    """参考 logprob 优先用推理卡算好的（ref_lp），没有才在训练卡算。share=1 时用 prompt 共享。"""
    import torch
    keep = [s for s in batch["samples"] if has_adv(s) and s["ids"]]
    n_tok = sum(len(s["ids"]) for s in keep) or 1
    st = collections.Counter()
    t0 = time.time()
    groups = collections.defaultdict(list)
    for s in keep:
        groups[s["prompt_id"]].append(s)
    (_accum_grads_shared if getattr(args, "share", 0) else _accum_grads)(model, groups, n_tok, args, st)
    gn = torch.nn.utils.clip_grad_norm_(lora_params(model, "policy"), 1.0).item() if keep else 0.0
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"train_samples": len(keep), "train_tokens": n_tok if keep else 0, "t_train": round(time.time() - t0, 1),
            "grad_norm": gn, "kl": st["kl"] / n_tok, "clip_frac": st["clip"] / n_tok,
            "vllm_hf_absdiff": st["vllm_hf_absdiff"] / n_tok, "policy_ref_absdiff": st["ref_hf_absdiff"] / n_tok,
            "staleness": cur_version - batch["version"]}


def batch_stats(batch: dict) -> dict:
    ss = batch["samples"]
    q = [s for s in ss if s.get("qualified")]
    why = collections.Counter(s["why"] for s in ss)
    groups = collections.defaultdict(set)
    for s in q:
        groups[s["prompt_id"]].add((s.get("op"), s["idx"], s["type"], s["tactic"]))
    m = lambda k, xs: round(sum(x[k] for x in xs) / len(xs), 4) if xs else None
    return {"samples": len(ss), "reward": m("reward", ss), "qualified_rate": round(len(q) / len(ss), 4),
            "conflict_share": round(sum(s["type"] == "conflict" for s in q) / len(q), 4) if q else None,
            "hard": m("hard", q), "real": m("real", q), "div": m("div", q), "p_halluc": m("p_halluc", q), "p_natural": m("p_natural", q),
            "hard_share(rm1<0.5)": round(sum(s["p_halluc"] < 0.5 for s in q) / len(q), 4) if q else None,
            "other_tactic_share": round(sum(s["tactic"] == "other" for s in q) / len(q), 4) if q else None,
            "raw_tactics": len({s["tactic_raw"] for s in q}),
            "orig_flagged_rate": round(sum(bool(s.get("orig_flagged")) for s in ss) / len(ss), 4),
            "top_tactic_share": round(max(collections.Counter((s["type"], s["tactic"]) for s in q).values()) / len(q), 4) if q else None,
            "p_correct": m("p_correct", q),
            # RM3 能看到模型自填的理由和证据：写得越长越有说服力也可能骗过门槛，所以盯长度
            "reason_chars": round(sum(len(s["reason"]) for s in q) / len(q), 1) if q else None,
            "evidence_chars": round(sum(len(s["evidence"]) for s in q) / len(q), 1) if q else None,
            "sentence_chars": round(sum(len(s["sentence"]) for s in q) / len(q), 1) if q else None,
            "place_per_prompt": round(sum(len({(s["op"], s["idx"], s["type"]) for s in q if s["prompt_id"] == p}) for p in {s["prompt_id"] for s in q}) / max(1, len({s["prompt_id"] for s in ss})), 3),
            "distinct_qualified_per_prompt": round(sum(len(v) for v in groups.values()) / max(1, len({s['prompt_id'] for s in ss})), 3),
            "why": dict(why), "completion_tokens": round(sum(len(s["ids"]) for s in ss) / len(ss), 1),
            "truncated": sum(s["finish"] == "length" for s in ss), "t_gen": round(batch["t_gen"], 1), "t_rm": round(batch["t_rm"], 1),
            "t_ref": round(batch.get("t_ref", 0.0), 1)}


def push(model, out: Path, server: Server, version: int, keep_versions: int, prefix: str = "") -> None:
    d = out / "lora" / f"v{version}"
    model.save_pretrained(str(d), selected_adapters=["policy"])
    path = d / "policy" if (d / "policy").exists() else d
    server.load_lora(f"{prefix}policy_v{version}", str(path))
    old = version - keep_versions
    if old >= 1:
        server.unload_lora(f"{prefix}policy_v{old}")
        shutil.rmtree(out / "lora" / f"v{old}", ignore_errors=True)


def save_state(model, opt, rw, prod, step: int, version: int, out: Path, keep: int) -> Path:
    """完整检查点：LoRA + 优化器 + 奖励统计 + prompt 位置 + 随机数（不能每次 reward hack 都重训，要能只回退一小段）。只留最近 keep 个。"""
    import torch
    d = out / "state" / f"step{step}"
    model.save_pretrained(str(d), selected_adapters=["policy"])
    torch.save({"step": step, "version": version, "opt": opt.state_dict(), "rw": rw.state_dict(), "pos": prod.pos,
                "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, d / "state.pt")
    old = sorted((p for p in (out / "state").iterdir() if p.name.startswith("step")), key=lambda p: int(p.name[4:]))
    for p in old[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
    return d


def load_state(model, opt, rw, prod, d: Path, reset_reward: bool = False) -> dict:
    """save_state 的逆操作。reset_reward：奖励口径改过时不恢复奖励统计，其余照常接着训。"""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    ad = d / "policy" if (d / "policy").exists() else d
    res = set_peft_model_state_dict(model, load_file(str(ad / "adapter_model.safetensors")), adapter_name="policy")
    assert not res.unexpected_keys, res.unexpected_keys[:3]
    st = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
    opt.load_state_dict(st["opt"])
    if not reset_reward:
        rw.load_state_dict(st["rw"])
    prod.pos = st["pos"]
    random.setstate(st["py_rng"])
    torch.set_rng_state(st["torch_rng"])
    return st


class TripWire:
    """自动报警（不能等人发现 reward hack）：每步的便宜指标，最近 window 步均值和训练初期的基线比，越线就存状态、停训练。
    只看 RM 分数以外的东西；把冷门的标准改法名硬贴到错误上这种作弊这里抓不到，要看快评的 distinct_qualified 和 distinct_place_type 是否背离。"""

    def __init__(self, start: int = 45, window: int = 20, base=(5, 25), other_max: float = 0.08, raw_ratio: float = 2.0,
                 conflict=(0.15, 0.60), qual_drop: float = 0.12, top_max: float = 0.60, kl_max: float = 0.10, len_ratio: float = 1.5):
        self.len_ratio = len_ratio
        self.start, self.window, self.base = start, window, base
        self.other_max, self.raw_ratio, self.conflict = other_max, raw_ratio, conflict
        self.qual_drop, self.top_max, self.kl_max = qual_drop, top_max, kl_max
        self.hist: list[dict] = []

    def update(self, rec: dict) -> list[str]:
        self.hist.append(rec)
        if len(self.hist) < max(self.start, self.base[1] + self.window):
            return []
        mean = lambda k, rows: sum((r.get(k) or 0.0) for r in rows) / len(rows)
        now, base = self.hist[-self.window:], self.hist[self.base[0]:self.base[1]]
        trips = []
        if mean("other_tactic_share", now) >= self.other_max:
            trips.append(f"other_tactic_share {mean('other_tactic_share', now):.3f} >= {self.other_max}")
        if mean("raw_tactics", now) >= max(self.raw_ratio * mean("raw_tactics", base), mean("raw_tactics", base) + 10):
            trips.append(f"raw_tactics {mean('raw_tactics', now):.1f} vs base {mean('raw_tactics', base):.1f}")
        if not self.conflict[0] <= mean("conflict_share", now) <= self.conflict[1]:
            trips.append(f"conflict_share {mean('conflict_share', now):.3f} outside {self.conflict}")
        if mean("qualified_rate", now) <= mean("qualified_rate", base) - self.qual_drop:
            trips.append(f"qualified_rate {mean('qualified_rate', now):.3f} vs base {mean('qualified_rate', base):.3f}")
        if mean("top_tactic_share", now) >= self.top_max:
            trips.append(f"top_tactic_share {mean('top_tactic_share', now):.3f} >= {self.top_max}")
        if mean("kl", now) >= self.kl_max:
            trips.append(f"kl {mean('kl', now):.3f} >= {self.kl_max}")
        for k in ("reason_chars", "evidence_chars", "sentence_chars"):   # 变长：可能在堆字骗 RM
            if mean(k, base) > 0 and mean(k, now) >= self.len_ratio * mean(k, base):
                trips.append(f"{k} {mean(k, now):.0f} vs base {mean(k, base):.0f}")
        return trips


def save_samples(batch: dict, step: int, out: Path) -> None:
    """每步的原始输出和打分都留着（gzip，一步约 50 KB）：查 reward hack 不用重新采样，也能回看模型什么时候开始变的。"""
    d = out / "samples"
    d.mkdir(exist_ok=True)
    skip = ("prompt_ids", "ids", "old_lp", "ref_lp")
    with gzip.open(d / f"step{step}.jsonl.gz", "wt", encoding="utf-8") as f:
        for s in batch["samples"]:
            f.write(json.dumps({"gen_version": batch["version"], **{k: v for k, v in s.items() if k not in skip}}, ensure_ascii=False))
            f.write(chr(10))


class QuickEval:
    """每个检查点存完就在推理卡上快速评测。
    评测集 = 生成器评测的 198 条种子回答（ragtruth.gen_eval.eval_resps，和训练 prompt 不重叠的原文），每条采 k 次，
    示例和 gen_eval sample 一样固定；只打 RM 原始概率（Rewarder.raw，不动训练的滑动统计）。检查点另起名字加载，
    评完卸载，不受训练里旧版本卸载的影响。后台单线程排队跑，不挡训练。step 0 = 起点 G_dpo2。"""

    def __init__(self, args, tok, server: Server, rewarder, out: Path, gate: threading.Event | None = None):
        self.gate = gate   # 生产者空闲的信号；None = 没有训练在跑，不用让路
        import difflib
        from ragtruth.gen_eval import eval_resps
        from ragtruth.synth import build_prompt_free, example_pool, free_examples
        from common.schema import read_jsonl
        self.difflib = difflib
        self.a, self.srv, self.rw, self.out = args, server, rewarder, out
        pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
        self.rows = []
        if getattr(args, "eval_prompts", ""):   # 定向版：评测 prompt 预先建好（和训练 prompt 同格式、原文不重叠）
            for r in read_jsonl(args.eval_prompts):
                self.rows.append({**r, "chat": tok(chat_prompt(tok, r["prompt"]))["input_ids"]})
        else:
            for s in eval_resps():
                p = build_prompt_free(s, free_examples(pool, s["task_type"], random.Random(s["id"]), exclude_source=s["source"]))
                self.rows.append({**s, "chat": tok(chat_prompt(tok, p))["input_ids"]})
        self.pool = ThreadPoolExecutor(1)
        self.futs = []
        self.first = None

    def submit(self, step: int, path: Path | None) -> None:
        self.futs.append(self.pool.submit(self._run, step, path))

    def _run(self, step: int, path: Path | None) -> None:
        try:
            name = "policy_v0"
            if path is not None:
                name = f"{self.a.version_prefix}eval_s{step}"
                self.srv.load_lora(name, str(path / "policy" if (path / "policy").exists() else path))
            # 只用推理卡的空闲（生成和打分时不做评测）：生产者空着才开始下一条；一直没有空闲时按 eval_minutes 的进度兜底，保证能评完
            gap = self.a.eval_minutes * 60 / len(self.rows)
            spent = collections.Counter()

            t0 = time.time()
            sem = threading.Semaphore(4)
            pr = 1 if self.gate is not None else 0   # vLLM 优先级调度：数字大的靠后，训练要的推理优先

            def one(r: dict) -> tuple[list[dict], list[dict]]:
                t = time.time()
                gs = self.srv.sample(name, r["chat"], self.a.eval_k, self.a.max_tokens, self.a.temperature, priority=pr)
                spent["gen"] += time.time() - t
                t = time.time()
                ss = list(rm.map(lambda g: self.rw.raw(g["text"], r, priority=pr), gs))
                spent["rm"] += time.time() - t
                sem.release()
                return gs, ss

            with ThreadPoolExecutor(8 if gap else self.a.rm_concurrency) as rm, ThreadPoolExecutor(8 if gap else 32) as ex:
                futs = []
                for i, r in enumerate(self.rows):
                    while gap and not ((self.gate is not None and self.gate.is_set()) or time.time() - t0 >= i * gap):
                        time.sleep(0.2)
                    if gap:
                        sem.acquire()   # 同时最多 4 条，空闲结束时能很快让路
                    futs.append(ex.submit(one, r))
                done = [f.result() for f in futs]
            if path is not None:
                self.srv.unload_lora(name)
            gens, scored = [d[0] for d in done], [d[1] for d in done]
            rec = {"step": step, **self.metrics(scored, gens), "t_gen": round(spent["gen"], 1), "t_rm": round(spent["rm"], 1), "t_wall": round(time.time() - t0, 1)}
            with open(self.out / f"quick_samples_{self.a.version_prefix}s{step}.jsonl", "w", encoding="utf-8") as f:   # 换口径时不用重采
                for r, gs, ss in zip(self.rows, gens, scored):
                    f.write(json.dumps({"id": r["id"], "outputs": [g["text"] for g in gs], "scores": ss}, ensure_ascii=False) + "\n")
            with open(self.out / "quick_eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # 按改法去重的多样性涨、不看改法的不涨：很可能在贴标签（把冷门改法名硬贴到错误上）
            kmax = str(max(int(x) for x in rec["distinct_place_type@K"]))
            cur = (rec["distinct_qualified@K"][int(kmax)], rec["distinct_place_type@K"][int(kmax)])
            if self.first is None:
                self.first = cur
            elif cur[0] >= 1.10 * self.first[0] and cur[1] <= 1.02 * self.first[1]:
                print("QUICK_EVAL_WARN label_gaming", json.dumps({"step": step, "first": self.first, "now": cur}), flush=True)
            print("QUICK_EVAL", json.dumps({k: rec[k] for k in ("step", "qualified_rate", "distinct_qualified@K", "distinct_place_type@K", "conflict_share", "other_tactic_share",
                                                                 "hard_share(rm1<0.5)", "natural_share(rm2>0.5)")}), flush=True)
        except Exception as e:   # 评测出错不影响训练
            print("QUICK_EVAL_FAILED", step, repr(e), flush=True)

    def metrics(self, scored: list[list[dict]], gens: list[list[dict]]) -> dict:
        """口径同 ragtruth.gen_eval report：合格 = 过硬规则且 RM3 P(合格) ≥ 0.5；不同 = (op, 句号, 类型, 改法) 去重。"""
        k = self.a.eval_k
        ks = [x for x in (1, 2, 4, 8, 16) if x <= k]
        allq, n_all, why, sims, idx_div, tac_div = [], 0, collections.Counter(), [], [], []
        distinct = {x: [] for x in ks}
        flagged = 0
        place = {x: [] for x in ks}   # 只按（op, 句号, 类型）去重：不看模型自填的改法，这个骗不了
        for ss in scored:
            n_all += len(ss)
            why.update(s["why"] for s in ss)
            g = getattr(self.rw, "gate", 0.5)
            q = [(j, s) for j, s in enumerate(ss) if s["why"] == "ok" and s["p_correct"] >= g and s.get("p_orig_halluc", 0.0) <= self.rw.orig_max]
            flagged += sum(s.get("p_orig_halluc", 0.0) > self.rw.orig_max for s in ss if s["why"] == "ok" and s["p_correct"] >= g)
            allq += [s for _, s in q]
            for x in ks:
                distinct[x].append(len({(s["op"], s["idx"], s["type"], s["tactic"]) for j, s in q if j < x}))
                place[x].append(len({(s["op"], s["idx"], s["type"]) for j, s in q if j < x}))
            idx_div.append(len({s["idx"] for _, s in q}))
            tac_div.append(len({s["tactic"] for _, s in q}))
            sent = [s["sentence"] for _, s in q]
            sims += [self.difflib.SequenceMatcher(None, a, b).ratio() for i, a in enumerate(sent) for b in sent[i + 1:]]
        n = len(scored)
        r = lambda x: round(x, 4)
        nq = max(1, len(allq))
        tactics = collections.Counter(s["tactic"] for s in allq)
        return {"responses": n, "samples": n_all, "rule_ok_rate": r(why["ok"] / max(1, n_all)),
                "qualified_rate": r(len(allq) / max(1, n_all)),
                "distinct_qualified@K": {x: round(sum(v) / max(1, n), 2) for x, v in distinct.items()},
                "distinct_place_type@K": {x: round(sum(v) / max(1, n), 2) for x, v in place.items()},
                "other_tactic_share": r(sum(s["tactic"] == "other" for s in allq) / nq),
                "orig_flagged_rate": r(flagged / max(1, n_all)),
                "raw_tactics_used": len({s["tactic_raw"] for s in allq}),
                "responses_with_any_qualified": r(sum(1 for v in distinct[ks[-1]] if v) / max(1, n)),
                "distinct_sentences_per_resp": round(sum(idx_div) / max(1, n), 2),
                "distinct_tactics_per_resp": round(sum(tac_div) / max(1, n), 2),
                "mean_pairwise_similarity": r(sum(sims) / max(1, len(sims))),
                "tactics_used": len(tactics), "types": dict(collections.Counter(s["type"] for s in allq)),
                "conflict_share": r(sum(s["type"] == "conflict" for s in allq) / nq),
                "insert_share": r(sum(s["op"] == "insert" for s in allq) / nq),
                "p_correct_mean": r(sum(s["p_correct"] for s in allq) / nq),
                "p_halluc_mean": r(sum(s["p_halluc"] for s in allq) / nq),
                "hard_share(rm1<0.5)": r(sum(s["p_halluc"] < 0.5 for s in allq) / nq),
                "p_natural_mean": r(sum(s["p_natural"] for s in allq) / nq),
                "natural_share(rm2>0.5)": r(sum(s["p_natural"] > 0.5 for s in allq) / nq),
                "why": dict(why), "completion_tokens": round(sum(len(g["ids"]) for gs in gens for g in gs) / max(1, n_all), 1),
                "truncated": sum(g["finish"] == "length" for gs in gens for g in gs)}

    def wait(self) -> None:
        for f in self.futs:
            f.result()


def run(args) -> None:
    import torch
    from openai import OpenAI
    from train.gen_reward import Rewarder, TargetRewarder, load_temps
    random.seed(0)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu_log = open(out / "gpu.csv", "a" if args.resume else "w")   # 接着训不能冲掉之前的记录
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader",
                            "-l", "2"], stdout=gpu_log)
    tok, model = load_policy(args)
    global PAD_ID
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW(lora_params(model, "policy"), lr=args.lr, weight_decay=0.0)
    server = Server(args.vllm_url)
    proc = bool(getattr(args, "proc_producer", 0)) and args.mode == "async"
    if proc:   # 采样 / 打分 / 快评放子进程（训练进程独占 GIL）
        prod = GenProcProducer(args, out)
        rw, qe = RwProxy(prod), (EvalProxy(prod) if args.eval_k > 0 else None)
    else:
        rw = build_rewarder(args)
        prod = Producer(args, tok, server, rw)
        qe = QuickEval(args, tok, server, rw, out, prod.idle) if args.eval_k > 0 else None
    version, first = 0, 1
    if args.resume:
        st = load_state(model, opt, rw, prod, Path(args.resume), bool(args.reset_reward))
        version, first = st["version"], st["step"] + 1
        push(model, out, server, version, args.keep_versions, args.version_prefix)
        prod.version = version
        if qe:   # 接着训：贴标签警告的基线、切换时没评完的检查点
            old = [json.loads(l) for l in open(out / "quick_eval.jsonl", encoding="utf-8")] if (out / "quick_eval.jsonl").exists() else []
            if old:
                k = max(old[0]["distinct_place_type@K"], key=int)
                qe.first = (old[0]["distinct_qualified@K"][k], old[0]["distinct_place_type@K"][k])
            ck = out / "ckpt" / f"step{st['step']}"
            if ck.exists() and st["step"] not in {o["step"] for o in old}:
                qe.submit(st["step"], ck)
        print("RESUMED", json.dumps({"step": st["step"], "version": version, "pos": st["pos"], "reset_reward": args.reset_reward}), flush=True)
    elif qe and args.eval_start:
        qe.submit(0, None)
    th = None
    if proc:
        prod.start()
    elif args.mode == "async":
        th = threading.Thread(target=prod.run, daemon=True)
        th.start()
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    trip = TripWire(conflict=tuple(args.conflict_range), other_max=args.other_max, top_max=args.top_max, raw_ratio=args.raw_ratio) if args.trip else None
    if trip and args.resume and (out / "log.jsonl").exists():   # 报警的基线从已有日志恢复，只认恢复点之前的行
        for l in open(out / "log.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if "t_step" in r and r["step"] < first:
                trip.hist.append(r)
    t_start = time.time()
    for step in range(first, args.steps + 1):
        t0 = time.time()
        while True:
            batch = prod.get() if proc else (prod.q.get() if args.mode == "async" else prod.make_batch())
            if version - batch["version"] <= args.max_stale:
                break
            log.write(json.dumps({"step": step, "dropped_stale": version - batch["version"]}) + "\n")
        t_wait = time.time() - t0
        rec = {"step": step, "gen_version": batch["version"], "t_wait": round(t_wait, 1), **batch_stats(batch)}
        save_samples(batch, step, out)
        rec.update(train_step(model, opt, batch, version, args))
        version += 1
        t1 = time.time()
        push(model, out, server, version, args.keep_versions, args.version_prefix)
        prod.version = version
        rec.update({"t_push": round(time.time() - t1, 1), "t_step": round(time.time() - t0, 1),
                    "elapsed_min": round((time.time() - t_start) / 60, 1)})
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps({k: rec[k] for k in ("step", "reward", "qualified_rate", "conflict_share", "kl", "staleness", "t_wait",
                                              "t_train", "t_step")}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ck = out / "ckpt" / f"step{step}"
            model.save_pretrained(str(ck), selected_adapters=["policy"])
            if qe:
                qe.submit(step, ck)
        if step % args.state_every == 0:
            save_state(model, opt, rw, prod, step, version, out, args.keep_states)
        reasons = trip.update(rec) if trip else []
        if reasons:   # 存好状态就停，修完用 --resume 接着训（或回到更早的 state）
            d = save_state(model, opt, rw, prod, step, version, out, args.keep_states)
            (out / "TRIP.json").write_text(json.dumps({"step": step, "reasons": reasons, "state": str(d)}, ensure_ascii=False, indent=1), encoding="utf-8")
            print("GRPO_TRIPPED", json.dumps({"step": step, "reasons": reasons}, ensure_ascii=False), flush=True)
            break
    if proc:
        prod.shutdown()
    else:
        prod.stop = True
        if qe:
            qe.wait()
    smi.terminate()
    print("GRPO_DONE", flush=True)


def eval_ckpts(args) -> None:
    """只评测（不训练）：--eval_ckpts 名字=路径 …，路径写 v0 表示起点 policy_v0。结果写 --out/quick_eval.jsonl，step 字段放名字的序号。"""
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train.gen_reward import Rewarder, load_temps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    server = Server(args.vllm_url)
    cl = OpenAI(base_url=args.vllm_url, api_key=vllm_key(), timeout=600, max_retries=3)
    if args.reward == "target":   # 定向版的评测口径（P(真错) 口径、门槛、2B critic / RM2 分数）和训练时一样（放大评测）
        from train.gen_reward import TargetRewarder
        rw = TargetRewarder(cl, load_temps(args.temps), gate=args.gate, type_floor=args.type_floor, err_mode=args.err_mode, shape=args.shape,
                            rm2_ens=[m for m in args.rm2_ens.split(",") if m], gate_lo=args.gate_lo)
    else:
        rw = Rewarder(cl, load_temps())
    qe = QuickEval(args, tok, server, rw, out)
    for k, item in enumerate(args.eval_ckpts):
        name, path = item.split("=", 1)
        print("EVAL", k, name, path, flush=True)
        qe.submit(k, None if path == "v0" else Path(path))
    qe.wait()


def check(args) -> None:
    """正确性：① 拼接 LoRA（基座 + 一个 r32 LoRA）和「先合并 SFT 再挂 DPO LoRA」逐 token logprob 一致；
    ② 推理卡 vLLM 上 policy_v0 采样时给的 logprob 和训练卡 HF 算的一致（重要性比值的基准）；③ 热加载能用。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")][:3]
    server = Server(args.vllm_url)
    gens = [(chat_prompt(tok, r["prompt"]), server.sample("policy_v0", chat_prompt(tok, r["prompt"]), 2, 256, 1.0)) for r in rows]
    items = [(tok(p, add_special_tokens=False)["input_ids"], g) for p, gs in gens for g in gs]
    res = {}

    def lps(model):
        with torch.no_grad():
            return [seq_logprobs(model, pids, g["ids"]).cpu() for pids, g in items]

    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    m = PeftModel.from_pretrained(base, args.init).eval()
    a = lps(m)
    del m, base
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    m = PeftModel.from_pretrained(base, args.seed_adapter).merge_and_unload()
    m = PeftModel.from_pretrained(m, args.dpo_adapter).eval()
    b = lps(m)
    del m, base
    torch.cuda.empty_cache()
    d_ab = torch.cat([x - y for x, y in zip(a, b)]).abs()
    d_av = torch.cat([x - torch.tensor(g["old_lp"]) for x, (_, g) in zip(a, items)]).abs()
    res["concat_vs_stacked"] = {"tokens": len(d_ab), "mean_abs": round(d_ab.mean().item(), 5), "max_abs": round(d_ab.max().item(), 4)}
    res["hf_vs_vllm"] = {"tokens": len(d_av), "mean_abs": round(d_av.mean().item(), 5), "p99_abs": round(d_av.quantile(0.99).item(), 4),
                         "max_abs": round(d_av.max().item(), 4)}
    server.load_lora("policy_check", args.init)
    res["hot_reload"] = bool(server.sample("policy_check", gens[0][0], 1, 16, 0.0))
    server.unload_lora("policy_check")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("CHECK", json.dumps(res), flush=True)


def check_share(args) -> None:
    """prompt 共享的正确性和速度（单卡，不需要 vLLM）：同一批真实样本（198 条评测回答里前 share_prompts 条的 prompt +
    G_dpo2 之前评测采样的 8 条输出），整段拼接（原做法，micro_batch）和共享各算一遍，比：
    ① 逐 token logprob；② 一步的 LoRA 梯度（余弦相似度、相对误差）；③ 训练卡自己算参考 logprob 时的 KL；④ 耗时、显存峰值。"""
    import torch
    from common.schema import read_jsonl
    from ragtruth.gen_eval import eval_resps
    from ragtruth.synth import build_prompt_free, example_pool, free_examples
    from train.prefix_share import completion_logprobs, prompt_pass
    global PAD_ID
    tok, model = load_policy(args)
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    outs = {r["id"]: r["outputs"] for r in read_jsonl("outputs/rt/realgen/gen_eval/samples_gdpo2.jsonl")}
    pool = example_pool(list(read_jsonl("outputs/rt/annotated/train_sub35.jsonl")))
    rng = random.Random(0)
    groups = {}
    for s in [s for s in eval_resps() if s["id"] in outs][: args.share_prompts]:
        p = build_prompt_free(s, free_examples(pool, s["task_type"], random.Random(s["id"]), exclude_source=s["source"]))
        pids = tok(chat_prompt(tok, p), add_special_tokens=False)["input_ids"]
        groups[s["id"]] = [{"prompt_id": s["id"], "prompt_ids": pids, "adv": rng.gauss(0, 1),
                            "ids": tok(o, add_special_tokens=False)["input_ids"][: args.max_tokens - 1] + [eos]}
                           for o in outs[s["id"]][: args.n]]
    allS = [x for g in groups.values() for x in g]
    n_tok = sum(len(x["ids"]) for x in allS)
    res = {"prompts": len(groups), "samples": len(allS), "prompt_tokens_mean": round(sum(len(g[0]["prompt_ids"]) for g in groups.values()) / len(groups)),
           "completion_tokens_mean": round(n_tok / len(allS), 1)}
    # ① logprob（不带梯度）
    with torch.no_grad():
        a, b = [], []
        for g in groups.values():
            for k in range(0, len(g), args.micro_batch):
                ch = g[k:k + args.micro_batch]
                lp, m = batch_logprobs(model, ch[0]["prompt_ids"], [x["ids"] for x in ch])
                a += [lp[i][m[i]].float().cpu() for i in range(len(ch))]
            rp = prompt_pass(model, g[0]["prompt_ids"], grad=False)
            lp, m = completion_logprobs(model, rp, [x["ids"] for x in g], PAD_ID)
            b += [lp[i][m[i]].float().cpu() for i in range(len(g))]
    d = torch.cat([x - y for x, y in zip(a, b)]).abs()
    for x, la in zip(allS, a):
        x["old_lp"] = la.tolist()
        x["ref_lp"] = (la - 0.01).tolist()   # 给定参考 logprob（推理卡算好的情形），KL 项也有梯度
    res["logprob"] = {"tokens": len(d), "mean_abs": round(d.mean().item(), 5), "p99_abs": round(d.quantile(0.99).item(), 4),
                      "max_abs": round(d.max().item(), 4)}
    params = lora_params(model, "policy")

    def grads(fn, a_):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        st = collections.Counter()
        fn(model, groups, n_tok, a_, st)
        torch.cuda.synchronize()
        dt = time.time() - t
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params]).cpu()
        model.zero_grad(set_to_none=True)
        return g, dt, torch.cuda.max_memory_allocated() / 2**30, st["kl"] / n_tok

    import copy
    A = copy.copy(args)
    grads(_accum_grads, A)   # 预热
    g_old, t_old, m_old, _ = grads(_accum_grads, A)
    out = {"concat_mb%d" % args.micro_batch: {"seconds": round(t_old, 1), "peak_gb": round(m_old, 1)}}
    for mb in args.share_mbs:
        A.share_mb = mb
        grads(_accum_grads_shared, A)
        g_new, t_new, m_new, _ = grads(_accum_grads_shared, A)
        cos = torch.nn.functional.cosine_similarity(g_old, g_new, 0).item()
        rel = ((g_old - g_new).norm() / g_old.norm()).item()
        out["share_mb%d" % mb] = {"seconds": round(t_new, 1), "peak_gb": round(m_new, 1), "grad_cos": round(cos, 6),
                                  "grad_rel_err": round(rel, 5), "speedup": round(t_old / t_new, 2)}
    res["grad_and_speed(ref given)"] = out
    # ③ 参考 logprob 在训练卡算（ref_on_vllm=0 的情形）
    for x in allS:
        x.pop("ref_lp")
    _, t_o, _, kl_o = grads(_accum_grads, A)
    A.share_mb = max(args.share_mbs)
    _, t_s, _, kl_s = grads(_accum_grads_shared, A)
    res["ref_on_trainer"] = {"concat_seconds": round(t_o, 1), "share_seconds": round(t_s, 1), "kl_concat": kl_o, "kl_share": kl_s}
    res["step_estimate_s(16x8, ref given)"] = {k: round(v["seconds"] * 16 / len(groups), 1) for k, v in out.items()}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "check_share.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("CHECK_SHARE", json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3.5-9B")
    ap.add_argument("--init", default="", help="起点生成器 LoRA（G_dpo2 拼成的 r32，scripts/concat_lora.py）")
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--vllm_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["async", "sync"], default="async")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.05, help="KL 系数（相对起点 G_dpo2）")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max_stale", type=int, default=2)
    ap.add_argument("--keep_versions", type=int, default=3)
    ap.add_argument("--micro_batch", type=int, default=4, help="同一 prompt 的几条一起前向 / 反向")
    ap.add_argument("--proc_producer", type=int, default=0, help="1：采样 / 打分 / 快评放子进程（async 模式），训练进程独占 GIL")
    ap.add_argument("--pipeline", type=int, default=0, help="1：每个 prompt 生成完立刻打分（生成和打分重叠）")
    ap.add_argument("--blind_bonus", type=float, default=0.0, help="shape=sev：critic 判不出（P < blind_thr）的合格样本额外加这么多（方案 A）")
    ap.add_argument("--blind_thr", type=float, default=0.5, help="加分区间上沿")
    ap.add_argument("--blind_lo", type=float, default=0.0, help="加分区间下沿（边界样本 0.3–0.7 最好）")
    ap.add_argument("--ref_batch", type=int, default=0, help="1：参考 logprob 按 prompt 合并成一次请求（双机模式省对外域名的往返）")
    ap.add_argument("--ref_on_vllm", type=int, default=1, help="参考 logprob 在推理卡上算（1）还是训练卡（0）")
    ap.add_argument("--version_prefix", default="", help="热加载的 LoRA 名前缀（同一个服务上跑几次时互不冲突；v0 都是 policy_v0）")
    ap.add_argument("--rm_concurrency", type=int, default=96)
    ap.add_argument("--w_hard", type=float, default=1.0)
    ap.add_argument("--w_real", type=float, default=1.0)
    ap.add_argument("--w_div", type=float, default=1.0)
    ap.add_argument("--type_floor", type=float, default=0.3)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_k", type=int, default=4, help="每个检查点快速评测：198 条评测回答各采几次（0 = 不评）")
    ap.add_argument("--eval_start", type=int, default=1, help="开训前先评一次起点（step 0）")
    ap.add_argument("--eval_minutes", type=float, default=60.0, help="快评优先用推理卡空闲，这是一直没空闲时的兜底时长（0 = 全力跑，只评测不训练时用）")
    ap.add_argument("--state_every", type=int, default=25, help="每多少步存一次完整状态（LoRA + 优化器 + 奖励统计 + prompt 位置）")
    ap.add_argument("--keep_states", type=int, default=12, help="一份约 1 GB；12 份 = 300 步全留")
    ap.add_argument("--resume", default="", help="从完整状态接着训：<out>/state/stepN")
    ap.add_argument("--reset_reward", type=int, default=0, help="接着训时不恢复奖励统计（奖励口径改过时用）")
    ap.add_argument("--trip", type=int, default=1, help="自动报警：指标越线就存状态并停（TripWire）")
    ap.add_argument("--div_mode", choices=["tactic", "place"], default="tactic", help="多样性：tactic = 最近窗口里的（类型, 改法）；place = 同一 prompt 这组里的位置（操作, 句号）")
    ap.add_argument("--eval_ckpts", nargs="+", default=[], help="只评测不训练：名字=路径（v0 = 起点）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--share", type=int, default=0, help="1 = prompt 共享（train.prefix_share）：同一 prompt 只算一遍")
    ap.add_argument("--share_mb", type=int, default=8, help="共享时 completion 每批几条")
    ap.add_argument("--check_share", action="store_true", help="单卡验证 prompt 共享的正确性和速度")
    ap.add_argument("--share_prompts", type=int, default=4)
    ap.add_argument("--share_mbs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seed_adapter", default="")
    # 定向造错
    ap.add_argument("--causal", type=int, default=0, help="1 = 纯文本 CausalLM 底座（MiniCPM5-2B）")
    ap.add_argument("--share_impl", default="train.prefix_share", help="prompt 共享实现；Llama 结构用 train.prefix_share_llama")
    ap.add_argument("--nat_gate", type=float, default=0.5, help="v2sev：RM2 v3 P(自然) 门槛")
    ap.add_argument("--crit_gate", type=int, default=0, help="v2sev：1 = 另加 critic P(有幻觉) ∈ [crit_lo, crit_hi) 门槛")
    ap.add_argument("--crit_lo", type=float, default=0.1)
    ap.add_argument("--crit_hi", type=float, default=0.7)
    ap.add_argument("--v2_shape", default="sev", choices=["sev", "hard"], help="v2sev：sev = sev 主奖励；hard = critic 难度主奖励、sev 有错 + 自然度门槛")
    ap.add_argument("--err_gate", type=float, default=0.9, help="v2sev hard：sev P(有错) 门槛（终值）")
    ap.add_argument("--hard_shape", default="tri", choices=["tri", "low", "rank"], help="v2sev hard：难度系数形状（rank = 组内相对难度 + tri）")
    ap.add_argument("--hard_base", type=float, default=0.1, help="v2sev hard：过门槛的保底（≥ 0.9 的明显错拿到的分）")
    ap.add_argument("--err_gate_start", type=float, default=None, help="v2sev hard：动态门槛起点（不设 = 固定 err_gate）")
    ap.add_argument("--err_gate_warm", type=int, default=0, help="v2sev hard：门槛从起点升到终值用多少步")
    ap.add_argument("--fail_scale", type=float, default=0.1, help="v2sev：没过门槛时奖励 = fail_scale × sev")
    ap.add_argument("--reward", choices=["default", "target", "v2sev", "ops"], default="default", help="target = 真实性主力、正确性和 critic 只扣分（gen_reward.TargetRewarder）")
    ap.add_argument("--ops_stage", type=int, default=1, choices=[1, 2, 3], help="ops：1 = 真错 × 自然 × 操作一致；2 = 过门槛后 critic 越低分越高；3 = 只在 critic 拿不准的区间给分")
    ap.add_argument("--ops_err_gate", type=float, default=0.5, help="ops：sev4 P(有错) 门槛")
    ap.add_argument("--op_gate", type=float, default=0.5, help="ops：opm P(操作一致) 门槛")
    ap.add_argument("--ops_min_sim", type=float, default=0.6, help="ops：改写句和原句的词序列相似度下限（只改一处）")
    ap.add_argument("--ops_ref", type=float, default=0.0, help="ops 阶段二 / 三：进入本阶段时阶段一的平均分；给了就自动调难度权重，保证前面几项不下降")
    ap.add_argument("--ops_w_hard", type=float, default=0.5, help="ops 阶段二 / 三：难度项权重的上限")
    ap.add_argument("--temps", default="", help="RM 温度来源：vLLM 名=RM 目录，逗号分隔；空 = 9B 版")
    ap.add_argument("--w_corr", type=float, default=0.5)
    ap.add_argument("--w_crit", type=float, default=0.3)
    ap.add_argument("--gate", type=float, default=0.7, help="RM3 P(合格) 低于这个只给 0.1 × P")
    ap.add_argument("--eval_prompts", default="", help="快评用的预建 prompt 文件（空 = 198 条评测回答 + 自由版 prompt）")
    ap.add_argument("--conflict_range", type=float, nargs=2, default=[0.15, 0.60], help="TripWire：conflict 占比允许范围（定向版以 conflict 为主）")
    ap.add_argument("--err_mode", default="a", choices=["a", "ab"], help="target：P(真错) 只算 RM3 的 A，还是 A + B（解释不对但句子错了）")
    ap.add_argument("--shape", default="v1", choices=["v1", "v2", "filter", "soft", "mult", "boundary", "boundary2", "boundary3", "boundary4", "boundary5", "boundary6", "boundary7", "sev", "gated", "add"], help="target：sev = 严重程度 RM 门槛 + (1 − P(critic)) × 自然度；v2 = 真实性和骗过 critic 都用对数几率绝对分、都加分；filter = 只看真实性，其他过滤；gated = sev门+盲区门→自然度(目标)；add = 盲区(主)+自然度(次)相加")
    ap.add_argument("--crit_max", type=float, default=0.98, help="target filter：critic P(判出) 超过它的过滤掉")
    ap.add_argument("--rm2_ens", default="", help="target：再用这些 RM2（vLLM 里的名字，逗号分隔，温度在 --temps 里给）打 P(自然)，取平均")
    ap.add_argument("--gate_lo", type=float, default=None, help="target：P(真错) 从 gate_lo 到 gate 线性给系数（不设 = 硬门槛）")
    ap.add_argument("--obvious_pen", type=float, default=0.1, help="target boundary7：明显的错（2B P ≥ 0.9）的负分")
    ap.add_argument("--base_fail", type=float, default=0.1, help="target boundary2：P(真错) 没过门槛的基线奖励")
    ap.add_argument("--min_sim", type=float, default=0.0, help="target：改写句和原句相似度低于它的记 0（防整句换掉）")
    ap.add_argument("--crit_floor", type=float, default=0.4, help="target soft：critic P(判出) = 1 时乘在真实性上的系数")
    ap.add_argument("--w_real2", type=float, default=0.45)
    ap.add_argument("--w_hard2", type=float, default=0.45)
    ap.add_argument("--raw_ratio", type=float, default=2.0, help="TripWire：改法原始名字数相对基线的倍数上限")
    ap.add_argument("--top_max", type=float, default=0.60, help="TripWire：最多的（类型, 改法）占比上限")
    ap.add_argument("--other_max", type=float, default=0.08, help="TripWire：改法归 other 的占比上限（定向版多样性只看位置、改法不进奖励，可放宽）")
    ap.add_argument("--dpo_adapter", default="")
    args = ap.parse_args()
    check(args) if args.check else check_share(args) if args.check_share else eval_ckpts(args) if args.eval_ckpts else run(args)


if __name__ == "__main__":
    main()
