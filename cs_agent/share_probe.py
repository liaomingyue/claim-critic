"""系统提示共享的显存 / 速度 / 一致性实测（单卡，9B，和 cs_agent.grpo 训练时同一套模型和梯度检查点设置）。

按轨迹总长分档（默认 7k / 10k / 13k / 16k / 20k token，含系统提示，取最接近且不短于该长度的真实轨迹，截到该长度），每档量：
  full    整段：参考 logprob（无梯度）+ 策略 logprob + 反向                     —— 现在训练走的路径
  shared  共享：系统提示带梯度过一遍 + 接在后面算参考 / 策略 + 反向 + 系统提示反向  —— 要打开的路径
  score_full / score_shared  两个奖励模型打分（无梯度）
记录峰值显存（GB）、耗时（秒）、两条路径策略 logprob 的平均 / 最大绝对差。某档 OOM 就记 OOM、接着测下一档。
另外先做一次内核检查：右下角因果注意力能不能走 memory-efficient 内核（不能的话会退回生成整块掩码）。

  python -m cs_agent.share_probe --base <9B> --init <零 LoRA> --critic_step <..> --critic_outcome <..> --eps <轨迹 jsonl> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time


def kernel_check() -> dict:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    q = torch.randn(1, 16, 1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 16, 3072, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = {}
    for name, be in (("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION)):
        try:
            with sdpa_kernel(be):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, k, attn_mask=causal_lower_right(1024, 3072))
                o.sum().backward()
            out[name] = "ok"
        except Exception as e:   # noqa: BLE001
            out[name] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def main() -> None:
    import torch
    from cs_agent import grpo
    from train.prefix_share import backward_prompt, leaves, prompt_pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--critic_step", required=True)
    ap.add_argument("--critic_outcome", default="")
    ap.add_argument("--eps", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[7000, 10000, 13000, 16000, 20000], help="总长（含 5,066 token 的系统提示）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"kernel": kernel_check()}
    print("KERNEL", json.dumps(res["kernel"]), flush=True)
    eps = []
    for l in open(a.eps, encoding="utf-8"):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        if e.get("prompt_ids") and e.get("ids") and e.get("lmask"):
            eps.append(e)
    pids = eps[0]["prompt_ids"]
    eps = [e for e in eps if e["prompt_ids"] == pids]
    tok, model, heads = grpo.load_models(a)
    cache = grpo.PrefixCache(pids)
    P = len(pids)
    gb = lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2)   # noqa: E731
    res["static_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    res["rows"] = []

    def run(fn):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        base = round(torch.cuda.memory_allocated() / 2**30, 2)
        t = time.time()
        try:
            v = fn()
            torch.cuda.synchronize()
            print("RUN", getattr(fn, "__name__", "?"), "base_gb", base, "peak_gb", gb(), flush=True)
            return v, gb(), round(time.time() - t, 2)
        except torch.OutOfMemoryError as ex:
            print("RUN", getattr(fn, "__name__", "?"), "base_gb", base, "OOM", str(ex)[:300].replace("\n", " "), flush=True)
            return None, "OOM", round(time.time() - t, 2)
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    for L in a.lens:
        cand = sorted((e for e in eps if P + len(e["ids"]) >= L), key=lambda e: len(e["ids"]))
        if not cand:
            res["rows"].append({"len": L, "skip": "no episode this long"})
            continue
        e = cand[0]
        ids = (e["prompt_ids"] + e["ids"])[:L]
        pos = [P + k for k, m in enumerate(e["lmask"]) if m and P + k < L]
        if not pos:
            res["rows"].append({"len": L, "skip": "no policy tokens within this length"})
            continue

        def full():
            with torch.no_grad(), model.disable_adapter():
                grpo.token_logprobs(model, ids, pos, grad=False)
            grpo.use_policy(model)
            lp = grpo.token_logprobs(model, ids, pos, grad=True)
            lp.sum().backward()
            return lp.detach()

        def shared():
            grpo.use_policy(model)
            pre = prompt_pass(model, pids, grad=True)
            leaf = leaves(pre)
            with torch.no_grad(), model.disable_adapter():
                grpo.shared_logprobs(model, cache.frozen_prefix(model, "ref"), ids, P, pos, grad=False)
            grpo.use_policy(model)
            lp = grpo.shared_logprobs(model, leaf, ids, P, pos, grad=True)
            lp.sum().backward()
            backward_prompt(pre, leaf)
            return lp.detach()

        ep = dict(e, ids=e["ids"][: L - P], turns=[t for t in e["turns"] if t["b"] <= L - P])
        lp_f, m_f, t_f = run(full)
        lp_s, m_s, t_s = run(shared)
        _, m_sf, t_sf = run(lambda: grpo.score_episode(model, heads, ep, None))
        _, m_ss, t_ss = run(lambda: grpo.score_episode(model, heads, ep, cache))
        row = {"len": L, "episode": e.get("episode_id"), "n_pos": len(pos), "full_gb": m_f, "full_s": t_f, "shared_gb": m_s, "shared_s": t_s,
               "score_full_gb": m_sf, "score_full_s": t_sf, "score_shared_gb": m_ss, "score_shared_s": t_ss}
        if lp_f is not None and lp_s is not None:
            d = (lp_f - lp_s).abs()
            row.update(lp_mean_diff=round(d.mean().item(), 5), lp_max_diff=round(d.max().item(), 4))
        res["rows"].append(row)
        print("PROBE", json.dumps(row), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("PROBE_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
