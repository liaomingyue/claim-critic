"""训练步性能剖析（训练卡 GPU 利用率只有约 30%，定位 Python 侧调度开销，看能不能优化——优先级最高）。

用合成批次（随机 token，长度按真实分布：每步 P 个 prompt × N 条回答，prompt 约 2000 token、回答约 120 token），
走和 train.ans_grpo 完全一样的 train_step（prompt 共享 + 逐层重算），只是内容是随机的：
  ① 分段计时：prompt 前向 / 回答前向 / 损失反向 / prompt 反向 / 优化器，每段前后 cuda 同步；
  ② torch.profiler：按算子统计 CPU 自身时间和 CUDA 时间，算 GPU 忙的比例（CUDA 时间合计 / 墙钟）；导出 chrome trace。
单卡上跑（训练机空闲时）。

  python -m train.profile_step --base /root/autodl-fs/models/MiniCPM5-2B --init /root/autodl-fs/adapters/mc-init \
      --share_ckpt 1 --share_mb 32 --out /root/autodl-tmp/prof
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace


def synth_batch(P: int, N: int, plen: int, clen: int, vocab: int, seed: int) -> dict:
    rng = random.Random(seed)
    samples = []
    for p in range(P):
        pids = [rng.randrange(1000, vocab - 1) for _ in range(rng.randint(int(plen * 0.6), int(plen * 1.4)))]
        for _ in range(N):
            n = rng.randint(int(clen * 0.5), int(clen * 1.5))
            ids = [rng.randrange(1000, vocab - 1) for _ in range(n)]
            samples.append({"prompt_id": f"p{p}", "prompt_ids": pids, "ids": ids, "adv": rng.choice([-1.0, -0.5, 0.5, 1.0]),
                            "old_lp": [-2.0] * n, "ref_lp": [-2.0] * n})
    return {"samples": samples, "version": 0}


def start_bg_load(n: int, stop) -> None:
    """模拟训练进程里的采样 / 打分线程：反复解析 vLLM 风格的 logprob JSON（约 200 KB）、做点 Python 计算，中间短暂等待（模拟 HTTP 往返）。
    只用来测 GIL 争用对训练线程的影响。"""
    import threading
    payload = json.dumps({"choices": [{"text": "x" * 400, "logprobs": {"tokens": [f"token_id:{i}" for i in range(96)],
                                                                   "token_logprobs": [-1.2345] * 96}} for _ in range(16)]})

    def work():
        while not stop.is_set():
            d = json.loads(payload)
            sum(float(x) for c in d["choices"] for x in c["logprobs"]["token_logprobs"])
            [int(t.split(":", 1)[1]) for c in d["choices"] for t in c["logprobs"]["tokens"]]
            time.sleep(0.02)

    for _ in range(n):
        threading.Thread(target=work, daemon=True).start()


def timed_train_step(model, opt, batch, args, ps) -> dict:
    """同 gen_grpo._accum_grads_shared + train_step，每段前后同步计时。"""
    import collections
    import torch
    from train.gen_grpo import _chunk_loss, lora_params
    sync = torch.cuda.synchronize
    t = collections.Counter()
    keep = [s for s in batch["samples"] if abs(s["adv"]) > 1e-6]
    n_tok = sum(len(s["ids"]) for s in keep)
    st = collections.Counter()
    groups = collections.defaultdict(list)
    for s in keep:
        groups[s["prompt_id"]].append(s)
    sync(); T0 = time.time()
    for g in groups.values():
        pids = g[0]["prompt_ids"]
        chunks = [g[k:k + args.share_mb] for k in range(0, len(g), args.share_mb)]
        sync(); a = time.time()
        pre = ps.prompt_pass(model, pids, grad=True)
        leaf = ps.leaves(pre)
        sync(); t["prompt_fwd"] += time.time() - a
        for ch in chunks:
            a = time.time()
            lp, mask = ps.completion_logprobs(model, leaf, [s["ids"] for s in ch], 0)
            sync(); t["comp_fwd"] += time.time() - a
            a = time.time()
            _chunk_loss(ch, lp, mask, [s["ref_lp"] for s in ch], n_tok, args, st).backward()
            sync(); t["comp_bwd"] += time.time() - a
        a = time.time()
        ps.backward_prompt(pre, leaf)
        sync(); t["prompt_bwd"] += time.time() - a
        del pre, leaf
    a = time.time()
    torch.nn.utils.clip_grad_norm_(lora_params(model, "policy"), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    sync(); t["optim"] += time.time() - a
    t["total"] = time.time() - T0
    return {k: round(v, 2) for k, v in t.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--share_ckpt", type=int, default=1)
    ap.add_argument("--share_mb", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--plen", type=int, default=2000)
    ap.add_argument("--clen", type=int, default=120)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="/root/autodl-tmp/prof")
    ap.add_argument("--bg_threads", default="0", help="逗号分隔：依次在这么多后台负载线程下各测 iters 步（测 GIL 争用），如 0,16,64")
    ap.add_argument("--no_profile", action="store_true")
    a = ap.parse_args()
    import torch
    import train.prefix_share_llama as ps
    from train.ans_grpo import load_causal_policy
    from train.gen_grpo import lora_params
    ps.USE_CKPT = bool(a.share_ckpt)
    args = SimpleNamespace(base=a.base, init=a.init, causal=1, share=1, share_impl="train.prefix_share_llama", share_mb=a.share_mb,
                           beta=0.05, clip=0.2, micro_batch=4)
    tok, model = load_causal_policy(args)
    opt = torch.optim.AdamW(lora_params(model, "policy"), lr=1e-9)
    vocab = len(tok)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"config": vars(a), "steps": []}
    import threading
    for nb in [int(x) for x in a.bg_threads.split(",")]:
        stop = threading.Event()
        start_bg_load(nb, stop)
        for it in range(a.iters):   # 每组第一步含预热（第一组）/ 线程启动，看后面的稳定值
            b = synth_batch(a.prompts, a.n, a.plen, a.clen, vocab, seed=it)
            torch.cuda.reset_peak_memory_stats()
            r = timed_train_step(model, opt, b, args, ps)
            r["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            r["bg_threads"] = nb
            res["steps"].append(r)
            print("STEP", it, json.dumps(r), flush=True)
        stop.set()
        time.sleep(0.5)
    if a.no_profile:
        (out / "profile.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    from torch.profiler import ProfilerActivity, profile
    b = synth_batch(min(4, a.prompts), a.n, a.plen, a.clen, vocab, seed=99)   # 剖析只跑 4 个 prompt，trace 不至于太大
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        t0 = time.time()
        timed_train_step(model, opt, b, args, ps)
        wall = time.time() - t0
    ka = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in ka) / 1e6
    res["profile"] = {"wall_s": round(wall, 2), "cuda_busy_s": round(cuda_total, 2), "gpu_busy_frac": round(cuda_total / wall, 3),
                      "n_cuda_kernels": sum(e.count for e in ka if e.self_device_time_total > 0),
                      "top_cpu": [(e.key, round(e.self_cpu_time_total / 1e6, 3), e.count) for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:25]],
                      "top_cuda": [(e.key, round(e.self_device_time_total / 1e6, 3), e.count) for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:25]]}
    prof.export_chrome_trace(str(out / "trace.json"))
    (out / "profile.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("PROFILE", json.dumps({k: v for k, v in res["profile"].items() if not k.startswith("top")}), flush=True)


if __name__ == "__main__":
    main()
