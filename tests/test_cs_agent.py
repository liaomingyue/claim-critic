"""客服 Agent 的多轮格式和只对策略 token 算损失（rollout 整体的自检见 scripts/analysis/cs_rollout_selftest.py，要装 τ²-bench）。"""
import collections
import types

import pytest

from cs_agent.qwen_fmt import parse_turn

TOOLS = [{"type": "function", "function": {"name": "exchange_delivered_order_items", "parameters": {"properties": {
    "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}}, "payment_method_id": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "calculate", "parameters": {"properties": {"expression": {"type": "string"}}}}}]


def test_parse_tool_call_restores_types():
    t = ("I'll exchange it now.\n\n<tool_call>\n<function=exchange_delivered_order_items>\n<parameter=order_id>\n#W2378156\n</parameter>\n"
         "<parameter=item_ids>\n[\"1151293680\", \"4983901480\"]\n</parameter>\n<parameter=payment_method_id>\ncredit_card_9513926\n</parameter>\n"
         "</function>\n</tool_call><|im_end|>")
    p = parse_turn(t, TOOLS)
    assert p["error"] is None and p["content"] == "I'll exchange it now."
    assert p["calls"] == [{"name": "exchange_delivered_order_items", "arguments": {
        "order_id": "#W2378156", "item_ids": ["1151293680", "4983901480"], "payment_method_id": "credit_card_9513926"}}]


def test_string_param_that_looks_like_a_number_stays_a_string():
    p = parse_turn("<tool_call>\n<function=calculate>\n<parameter=expression>\n12\n</parameter>\n</function>\n</tool_call>", TOOLS)
    assert p["calls"][0]["arguments"] == {"expression": "12"}


def test_plain_reply_and_malformed_call():
    assert parse_turn("Could you confirm your zip code?", TOOLS) == {"content": "Could you confirm your zip code?", "calls": [], "error": None}
    assert parse_turn("<tool_call>\n<function=calculate>", TOOLS)["error"] == "malformed tool call"
    assert parse_turn("<tool_call>\n<function=nope>\n</function>\n</tool_call>", TOOLS)["error"] == "unknown tool nope"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="损失计算在 GPU 上")
def test_masked_tokens_do_not_change_the_loss():
    import torch
    from train.gen_grpo import _chunk_loss
    args = types.SimpleNamespace(clip=0.2, beta=0.04)
    lp = torch.tensor([[-1.0, -2.0, -0.5, -0.3]], device="cuda")
    mask = torch.ones_like(lp, dtype=torch.bool)
    base = {"old_lp": [-1.1, 0.0, -0.4, 0.0], "lmask": [1, 0, 1, 0]}
    a = _chunk_loss([{**base, "adv": [1.0, 0.0, 1.0, 0.0]}], lp, mask, [[-1.0, -9.0, -0.5, -9.0]], 2, args, collections.Counter())
    b = _chunk_loss([{**base, "adv": [1.0, 50.0, 1.0, -50.0]}], lp, mask, [[-1.0, 3.0, -0.5, 3.0]], 2, args, collections.Counter())
    assert torch.allclose(a, b)
