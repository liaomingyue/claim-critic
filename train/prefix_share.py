"""GRPO 训练卡的 prompt 共享（2026-10-31，用户：先验证）：同一 prompt 的 n 条 completion 只算一遍 prompt。

现在 train.gen_grpo.batch_logprobs 每条样本都把 prompt（~3,300 token）+ completion（~240）整段过前向 / 反向，
prompt 占 93% 的算力且同一 prompt 重复 n 次。这里：
  1. prompt 过一遍（带梯度，逐层梯度检查点），拿到每层的「读完 prompt 后的状态」：全注意力层的 K / V，
     线性注意力层（Gated DeltaNet）的卷积状态（最后 4 个输入）和循环状态，以及最后一个位置的隐状态；
  2. 把这些状态 detach 成叶子张量，completion 分批接在后面算 logprob、loss、反向——梯度一部分直接进 LoRA，
     一部分累加到叶子张量上（n 条的和）；
  3. 最后用累加的梯度对第 1 步的状态做一次反向，梯度沿 prompt 传回 LoRA。
数学上和整段拼接完全一样（不是截断梯度的近似）。

为什么自己逐层写：HF 开梯度检查点时强制丢掉缓存（GradientCheckpointingLayer 把 past_key_values 置 None），
而 prompt 不开检查点显存放不下；拼成一条序列用掩码隔开的做法对线性注意力层无效（状态按 token 顺序往后传）。
所以直接调每个 decoder layer 的 forward，用一个只实现 Qwen3.5 层会调用的那几个方法的缓存替身（_Shim）收发状态。
依赖 transformers 5.19 里 Qwen3.5 的内部实现（Qwen3_5DecoderLayer / GatedDeltaNet / Attention 对缓存的调用方式），
升级 transformers 后要重新跑 train.gen_grpo --check_share。
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


class _LayerState:
    def __init__(self, rec=None):
        self.record_past = True            # 让 GatedDeltaNet 走整段（非单 token 原地更新）的分支
        self.recurrent_states = {0: rec}


class _Shim:
    """单层的缓存替身。prefix：这一层 prompt 的状态（completion 阶段），None = prompt 阶段。out：这一层算出的新状态。"""

    def __init__(self, idx: int, prefix: dict | None):
        self.prefix, self.out = prefix or {}, {}
        self.layers = {idx: _LayerState(self.prefix.get("rec"))}

    def has_previous_state(self, layer_idx=None, state_idx=None) -> bool:
        return "conv" in self.prefix

    def update_conv_state(self, x, layer_idx, conv_kernel_size=None, **kw):
        full = torch.cat([self.prefix["conv"], x], -1) if "conv" in self.prefix else x
        k = conv_kernel_size
        self.out["conv"] = full[..., -k:] if full.shape[-1] >= k else torch.nn.functional.pad(full, (k - full.shape[-1], 0))
        return full

    def update_recurrent_state(self, s, layer_idx, **kw):
        self.out["rec"] = s
        return s

    def update(self, k, v, layer_idx, *a, **kw):
        if "k" in self.prefix:
            k, v = torch.cat([self.prefix["k"], k], -2), torch.cat([self.prefix["v"], v], -2)
        self.out["k"], self.out["v"] = k, v
        return k, v

    def get_seq_length(self, layer_idx=0) -> int:
        return self.prefix["k"].shape[-2] if "k" in self.prefix else 0


KEYS = {"full_attention": ("k", "v"), "linear_attention": ("conv", "rec")}


def parts(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    text = next(m for m in core.modules() if type(m).__name__ == "Qwen3_5TextModel")
    assert text.config._attn_implementation == "sdpa", text.config._attn_implementation   # completion 阶段用布尔掩码
    return text, core.lm_head


def _rope(text, h, start: int):
    B, L = h.shape[:2]
    pos = torch.arange(start, start + L, device=h.device)
    pos4 = pos.view(1, 1, -1).expand(4, B, -1)   # 同 Qwen3_5TextModel.forward：第 0 行是文本位置，后 3 行给 mrope
    return text.rotary_emb(h, pos4[1:]), pos4[0]


def prompt_pass(model, prompt_ids: list[int], grad: bool = True) -> list:
    """→ [h_last, 每层状态…]（按层、按 KEYS 顺序展平）。grad=True 时逐层梯度检查点，图留着给最后一次反向。"""
    text, _ = parts(model)
    ids = torch.tensor([prompt_ids], device=text.embed_tokens.weight.device)
    with torch.set_grad_enabled(grad):
        h = text.embed_tokens(ids)
        pe, tpos = _rope(text, h, 0)
        states = []
        for i, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
            kind = text.config.layer_types[i]

            def f(h_, i=i, layer=layer, kind=kind):
                sh = _Shim(i, None)
                o = layer.forward(h_, position_embeddings=pe, attention_mask=None, position_ids=tpos, past_key_values=sh)
                return (o, *[sh.out[k] for k in KEYS[kind]])

            out = checkpoint(f, h, use_reentrant=False) if grad else f(h)
            h, st = out[0], out[1:]
            states += list(st)
        return [h[:, -1:]] + states


def completion_logprobs(model, prefix: list, comps: list[list[int]], pad_id: int):
    """prefix：prompt_pass 的输出（可以是 detach 后的叶子）。→ ([B, Cmax] logprob, mask)，和 batch_logprobs 一样。"""
    text, lm_head = parts(model)
    dev = text.embed_tokens.weight.device
    B = len(comps)
    cmax = max(2, max(len(c) for c in comps))   # 至少 2：长度 1 会走 GatedDeltaNet 单 token 原地更新的分支
    ids = torch.full((B, cmax), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((B, cmax), dtype=torch.bool, device=dev)
    for b, c in enumerate(comps):
        ids[b, :len(c)] = torch.tensor(c, device=dev)
        mask[b, :len(c)] = True
    h_last, states = prefix[0], prefix[1:]
    P = states[2 * text.config.layer_types.index("full_attention")].shape[-2]   # 第一层全注意力的 K：[1, 头, P, d]
    # 右侧补齐只影响补齐位置自己（因果），不用 padding 掩码；全注意力层：completion 第 t 个位置看 prompt 全部 + 自己及之前
    attn = (torch.arange(P + cmax, device=dev)[None, :] <= P + torch.arange(cmax, device=dev)[:, None])[None, None]
    h = text.embed_tokens(ids)
    pe, tpos = _rope(text, h, P)
    grad = torch.is_grad_enabled()
    s = 0
    for i, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
        kind = text.config.layer_types[i]
        a_, b_ = states[s], states[s + 1]
        s += 2

        def f(h_, a_, b_, i=i, layer=layer, kind=kind):
            if kind == "full_attention":
                pre = {"k": a_.expand(B, -1, -1, -1), "v": b_.expand(B, -1, -1, -1)}
                m = attn
            else:
                pre = {"conv": a_.expand(B, -1, -1), "rec": b_.expand(B, -1, -1, -1).contiguous()}
                m = None
            return layer.forward(h_, position_embeddings=pe, attention_mask=m, position_ids=tpos, past_key_values=_Shim(i, pre))

        h = checkpoint(f, h, a_, b_, use_reentrant=False) if grad else f(h, a_, b_)
    hs = torch.cat([h_last.expand(B, -1, -1), h[:, :-1]], 1)   # 预测 completion 第 t 个 token 的是位置 t-1（第 0 个看 prompt 末尾）
    hs = text.norm(hs)

    def row(h_, t_):   # 一条一条算、带检查点：不存 [C, 24.8 万词表] 的 logits 给反向（每批 4 条时 OOM 过）
        lg = lm_head(h_).float()
        return lg.gather(1, t_[:, None])[:, 0] - torch.logsumexp(lg, -1)

    lp = torch.stack([checkpoint(row, hs[b], ids[b], use_reentrant=False) if grad else row(hs[b], ids[b]) for b in range(B)])
    return lp, mask


def leaves(prefix: list) -> list:
    return [t.detach().requires_grad_(t.requires_grad) for t in prefix]


def backward_prompt(prefix: list, leaf: list) -> None:
    """completion 都反向完以后：用叶子上累加的梯度对 prompt 阶段的状态做一次反向。"""
    pairs = [(t, l.grad) for t, l in zip(prefix, leaf) if t.requires_grad and l.grad is not None]
    if pairs:
        torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
