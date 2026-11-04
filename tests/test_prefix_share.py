"""train.prefix_share：随机初始化的迷你 Qwen3.5（fp32、CPU、torch 参考实现）上，prompt 共享和整段拼接的
逐 token logprob、全部参数的梯度一致（梯度要沿 prompt 传回去，不是截断近似）。"""
import torch

from train import prefix_share as ps

DEV = "cuda" if torch.cuda.is_available() else "cpu"   # 装了 fla 的机器上线性注意力走 Triton 核，只能在 GPU 上跑
TOL = dict(atol=2e-3, rtol=2e-2) if DEV == "cuda" else dict(atol=1e-4, rtol=1e-3)   # fla 核内部用 tf32 / bf16 累加


def tiny():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, max_position_embeddings=512, attn_implementation="sdpa")
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg).float().to(DEV)
    with torch.no_grad():   # A_log / dt_bias 等默认初始化太平，放大一点让线性注意力的状态真正起作用
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return m


def concat_lp(m, prompt, comps):
    out = []
    for c in comps:
        ids = torch.tensor([prompt + c], device=DEV)
        logits = m(input_ids=ids).logits[0, len(prompt) - 1:-1].float()
        out.append(torch.log_softmax(logits, -1).gather(1, torch.tensor(c, device=DEV)[:, None])[:, 0])
    return out


def test_prefix_share_matches_concat():
    m = tiny()
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 97, (37,), generator=g).tolist()
    comps = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (5, 1, 9, 3)]
    adv = torch.tensor([0.7, -1.2, 0.4, 1.5], device=DEV)
    # 整段拼接
    m.zero_grad()
    a = concat_lp(m, prompt, comps)
    sum(w * x.sum() for w, x in zip(adv, a)).backward()
    ga = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    # 共享：两批 completion
    m.zero_grad()
    pre = ps.prompt_pass(m, prompt, grad=True)
    leaf = ps.leaves(pre)
    b = []
    for k in (0, 2):
        lp, mask = ps.completion_logprobs(m, leaf, comps[k:k + 2], pad_id=0)
        rows = [lp[i][mask[i]] for i in range(lp.shape[0])]
        b += rows
        sum(w * x.sum() for w, x in zip(adv[k:k + 2], rows)).backward()
    ps.backward_prompt(pre, leaf)
    gb = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=TOL["atol"]), (x, y)
    assert ga.keys() == gb.keys()
    for n in ga:
        rel = ((ga[n] - gb[n]).norm() / ga[n].norm().clamp_min(1e-12)).item()
        assert torch.allclose(ga[n], gb[n], **TOL) or rel < 1e-2, (n, rel, (ga[n] - gb[n]).abs().max())
