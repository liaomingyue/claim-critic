"""GRPO 训练卡的 prompt 共享，Llama 结构版（2026-11-02，用户：MiniCPM5-2B 也要有，速度才能更快）。接口和 train.prefix_share（Qwen3.5 版）相同，
train.gen_grpo._accum_grads_shared 按 --share_impl 选用。

做法同 Qwen3.5 版：prompt 逐层梯度检查点过一遍，拿到每层的 K / V 和末位隐状态 → detach 成叶子 → completion 分批接在后面算 logprob、反向
（梯度一部分进 LoRA，一部分累加到叶子上）→ 最后对 prompt 阶段的 K / V 做一次反向。数学上和整段拼接完全一样。
Llama 只有全注意力，前缀状态就是每层的 K / V，不需要 Qwen3.5 那样处理卷积和循环状态。
依赖 transformers 5.20 里 LlamaDecoderLayer / LlamaAttention 对缓存的调用方式（past_key_values.update），升级后要重跑 tests/test_prefix_share_llama.py。
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

USE_CKPT = True   # 逐层梯度检查点：省显存、多一次前向。2B 显存够时关掉更快（train.ans_grpo --share_ckpt 0）
LM_CHUNK = 2048   # 输出层每块 token 数：fp32 logits 约 LM_CHUNK × 词表 × 4 字节


class _Shim:
    """单层的缓存替身。prefix：这一层 prompt 的 K / V（completion 阶段），None = prompt 阶段。out：这一层算出的 K / V（含前缀）。"""

    def __init__(self, prefix: dict | None):
        self.prefix, self.out = prefix or {}, {}

    def update(self, k, v, layer_idx, *a, **kw):
        if "k" in self.prefix:
            k, v = torch.cat([self.prefix["k"], k], -2), torch.cat([self.prefix["v"], v], -2)
        self.out["k"], self.out["v"] = k, v
        return k, v

    def get_seq_length(self, layer_idx=0) -> int:
        return self.prefix["k"].shape[-2] if "k" in self.prefix else 0


def parts(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    text = next(m for m in core.modules() if type(m).__name__ == "LlamaModel")
    assert text.config._attn_implementation == "sdpa", text.config._attn_implementation   # completion 阶段用布尔掩码
    return text, core.lm_head


def _rope(text, h, start: int):
    pos = torch.arange(start, start + h.shape[1], device=h.device)[None]
    return text.rotary_emb(h, pos), pos


def prompt_pass(model, prompt_ids: list[int], grad: bool = True) -> list:
    """→ [h_last, 第 0 层 K, 第 0 层 V, 第 1 层 K, …]。grad=True 时逐层梯度检查点，图留着给最后一次反向。"""
    text, _ = parts(model)
    ids = torch.tensor([prompt_ids], device=text.embed_tokens.weight.device)
    with torch.set_grad_enabled(grad):
        h = text.embed_tokens(ids)
        pe, pos = _rope(text, h, 0)
        states = []
        for layer in text.layers[: text.config.num_hidden_layers]:

            def f(h_, layer=layer):
                sh = _Shim(None)
                o = layer.forward(h_, position_embeddings=pe, attention_mask=None, position_ids=pos, past_key_values=sh)   # 无掩码：sdpa 走因果
                return o, sh.out["k"], sh.out["v"]

            h, k, v = checkpoint(f, h, use_reentrant=False) if grad and USE_CKPT else f(h)
            states += [k, v]
        return [h[:, -1:]] + states


def completion_logprobs(model, prefix: list, comps: list[list[int]], pad_id: int):
    """prefix：prompt_pass 的输出（可以是 detach 后的叶子）。→ ([B, Cmax] logprob, mask)，和 gen_grpo.batch_logprobs 一样。"""
    text, lm_head = parts(model)
    dev = text.embed_tokens.weight.device
    B = len(comps)
    cmax = max(len(c) for c in comps)
    ids = torch.full((B, cmax), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((B, cmax), dtype=torch.bool, device=dev)
    for b, c in enumerate(comps):
        ids[b, :len(c)] = torch.tensor(c, device=dev)
        mask[b, :len(c)] = True
    h_last, states = prefix[0], prefix[1:]
    P = states[0].shape[-2]
    # 右侧补齐只影响补齐位置自己（因果）；completion 第 t 个位置看 prompt 全部 + 自己及之前
    attn = (torch.arange(P + cmax, device=dev)[None, :] <= P + torch.arange(cmax, device=dev)[:, None])[None, None]
    h = text.embed_tokens(ids)
    pe, pos = _rope(text, h, P)
    grad = torch.is_grad_enabled()
    for i, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
        k_, v_ = states[2 * i], states[2 * i + 1]

        def f(h_, k_, v_, layer=layer):
            return layer.forward(h_, position_embeddings=pe, attention_mask=attn, position_ids=pos,
                                 past_key_values=_Shim({"k": k_.expand(B, -1, -1, -1), "v": v_.expand(B, -1, -1, -1)}))

        h = checkpoint(f, h, k_, v_, use_reentrant=False) if grad and USE_CKPT else f(h, k_, v_)
    hs = torch.cat([h_last.expand(B, -1, -1), h[:, :-1]], 1)   # 预测 completion 第 t 个 token 的是位置 t-1（第 0 个看 prompt 末尾）
    hs = text.norm(hs)

    def head(h_, t_):   # 带检查点：不存 [N, 13 万词表] 的 logits 给反向
        lg = lm_head(h_).float()
        return lg.gather(1, t_[:, None])[:, 0] - torch.logsumexp(lg, -1)

    # 只算有效 token、拉平后按 LM_CHUNK 分块（原来一条回答一块：块多且含补齐位置，2B 上 GPU0 利用率只有 30%，2026-11-02）
    hv, tv = hs[mask], ids[mask]
    out = [checkpoint(head, hv[k:k + LM_CHUNK], tv[k:k + LM_CHUNK], use_reentrant=False) if grad else head(hv[k:k + LM_CHUNK], tv[k:k + LM_CHUNK])
           for k in range(0, hv.shape[0], LM_CHUNK)]
    lp = torch.zeros((B, cmax), dtype=torch.float32, device=dev)
    lp = lp.masked_scatter(mask, torch.cat(out)) if out else lp
    return lp, mask


def leaves(prefix: list) -> list:
    return [t.detach().requires_grad_(t.requires_grad) for t in prefix]


def backward_prompt(prefix: list, leaf: list) -> None:
    """completion 都反向完以后：用叶子上累加的梯度对 prompt 阶段的状态做一次反向。"""
    pairs = [(t, l.grad) for t, l in zip(prefix, leaf) if t.requires_grad and l.grad is not None]
    if pairs:
        torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
