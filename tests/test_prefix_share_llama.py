"""train.prefix_share_llama：随机初始化的迷你 Llama（fp32、CPU）上，prompt 共享和整段拼接的逐 token logprob、全部参数的梯度一致
（梯度要沿 prompt 传回去，不是截断近似）；不反传 prompt 时梯度应该对不上（测试本身有效）。"""
import torch

from train import prefix_share_llama as ps


def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=512, attn_implementation="sdpa", tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).float()


def concat_lp(m, prompt, comps):
    out = []
    for c in comps:
        logits = m(input_ids=torch.tensor([prompt + c])).logits[0, len(prompt) - 1:-1].float()
        out.append(torch.log_softmax(logits, -1).gather(1, torch.tensor(c)[:, None])[:, 0])
    return out


def _run(m, prompt, comps, adv, back_prompt=True):
    m.zero_grad()
    pre = ps.prompt_pass(m, prompt, grad=True)
    leaf = ps.leaves(pre)
    b = []
    for k in (0, 2):
        lp, mask = ps.completion_logprobs(m, leaf, comps[k:k + 2], pad_id=0)
        rows = [lp[i][mask[i]] for i in range(lp.shape[0])]
        b += rows
        sum(w * x.sum() for w, x in zip(adv[k:k + 2], rows)).backward()
    if back_prompt:
        ps.backward_prompt(pre, leaf)
    return b, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_prefix_share_llama_matches_concat():
    m = tiny()
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 97, (37,), generator=g).tolist()
    comps = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (5, 1, 9, 3)]
    adv = torch.tensor([0.7, -1.2, 0.4, 1.5])
    m.zero_grad()
    a = concat_lp(m, prompt, comps)
    sum(w * x.sum() for w, x in zip(adv, a)).backward()
    ga = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    b, gb = _run(m, prompt, comps, adv)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5), (x, y)
    assert ga.keys() == gb.keys()
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-5, rtol=1e-4), n
    _, gc = _run(m, prompt, comps, adv, back_prompt=False)   # 不把梯度传回 prompt：前面层的梯度必须对不上
    assert not all(torch.allclose(ga[n], gc.get(n, torch.zeros_like(ga[n])), atol=1e-5) for n in ga)


def test_prefix_share_llama_no_grad_matches():
    m = tiny().eval()
    prompt, comps = list(range(10, 30)), [[1, 2, 3], [4, 5]]
    with torch.no_grad():
        a = concat_lp(m, prompt, comps)
        lp, mask = ps.completion_logprobs(m, ps.prompt_pass(m, prompt, grad=False), comps, pad_id=0)
    for i, x in enumerate(a):
        assert torch.allclose(x, lp[i][mask[i]], atol=1e-5)


def test_prefix_share_llama_small_lm_chunk_matches():   # 输出层分块跨回答边界
    old, ps.LM_CHUNK = ps.LM_CHUNK, 3
    try:
        test_prefix_share_llama_matches_concat()
    finally:
        ps.LM_CHUNK = old


def test_prefix_share_llama_without_checkpoint_matches():
    ps.USE_CKPT = False
    try:
        test_prefix_share_llama_matches_concat()
    finally:
        ps.USE_CKPT = True
