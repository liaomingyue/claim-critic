"""Qwen3.5 聊天格式的多轮 token 拼接，和工具调用的解析。

做法是「token 进、token 出」：对话不在每轮重新套模板，而是逐段往一个 token 列表后面接。
  - 开头（系统提示 + 工具定义）用官方模板渲染一次；
  - 策略每一轮：先接生成提示（<|im_start|>assistant + 空的思考块，和模板关闭思考时一致），再原样接 vLLM 返回的生成 token；
  - 工具返回、用户的话：按模板的写法渲染成文本再分词接上。
这样训练时看到的 token 和采样时完全一样。如果每轮重新套模板，旧的助手轮会被模板去掉思考块、重新分词也可能切法不同，
训练和采样就对不上了。每段带一个标记：策略生成的 token 算 loss（lmask = 1），其余都不算。
"""
from __future__ import annotations

import json
import re

GEN_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
IM_END = "<|im_end|>"


class Conv:
    """一条对话的 token 序列和每个 token 的来源。"""

    def __init__(self, tok, system: str, tools: list[dict]):
        self.tok = tok
        self.im_end = tok.convert_tokens_to_ids(IM_END)
        # 模板要求至少有一条用户消息：拿一条占位消息渲染，再截到占位消息之前，就是系统提示 + 工具定义那一段
        txt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": "@@PLACEHOLDER@@"}],
                                      tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        self.prefix_text = txt[: txt.index("<|im_start|>user\n@@PLACEHOLDER@@")]
        self.prefix_ids = self.enc(self.prefix_text)
        self.ids: list[int] = []     # 开头之后的所有 token
        self.lmask: list[int] = []   # 1 = 策略生成，算 loss
        self.old_lp: list[float] = []
        self.turns: list[dict] = []  # 每个策略轮在 ids 里的区间 [a, b)，以及解析结果，供逐步打分和步骤级惩罚定位

    def enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _add(self, ids: list[int], policy: bool, lps: list[float] | None = None) -> None:
        self.ids += ids
        self.lmask += [1 if policy else 0] * len(ids)
        self.old_lp += lps if lps is not None else [0.0] * len(ids)

    def add_text(self, text: str) -> None:
        self._add(self.enc(text), False)

    def add_user(self, text: str) -> None:
        self.add_text(f"<|im_start|>user\n{text.strip()}{IM_END}\n")

    def add_tool_results(self, contents: list[str]) -> None:
        # 模板：连续的工具结果放在同一个 user 块里，每条包一层 <tool_response>
        self.add_text("<|im_start|>user" + "".join(f"\n<tool_response>\n{c}\n</tool_response>" for c in contents) + f"{IM_END}\n")

    def add_assistant_text(self, text: str) -> None:
        """不是策略生成的助手消息（τ² 固定的开场白）：按生成提示的格式接上，不算 loss。"""
        self.add_text(GEN_PROMPT + text + f"{IM_END}\n")

    def gen_prompt(self) -> list[int]:
        """采样用的完整 prompt：开头 + 目前的对话 + 生成提示（生成提示在 add_policy 里正式接上）。"""
        return self.prefix_ids + self.ids + self.enc(GEN_PROMPT)

    def add_policy(self, gen_ids: list[int], gen_lps: list[float], finish: str) -> tuple[int, int]:
        """接上一轮策略输出，返回它在 ids 里的区间（不含生成提示和结尾的换行）。
        vLLM 在 <|im_end|> 处停下时，结尾符一般在返回的 token 里（算 loss，模型要学会停）；不在就补一个不算 loss 的。"""
        self.add_text(GEN_PROMPT)
        a = len(self.ids)
        self._add(list(gen_ids), True, list(gen_lps))
        if not gen_ids or gen_ids[-1] != self.im_end:
            self._add([self.im_end], False)
        b = len(self.ids)
        self.add_text("\n")
        return a, b

    def n_tokens(self) -> int:
        return len(self.prefix_ids) + len(self.ids)


_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _cast(v: str, schema: dict | None):
    """模板把非字符串参数写成 JSON（tojson），字符串原样写；按工具定义里的类型还原。"""
    if (schema or {}).get("type") == "string":
        return v
    try:   # 数组、对象、数字，或定义里没写类型：能当 JSON 解析就解析，否则当字符串
        return json.loads(v)
    except ValueError:
        return v


def parse_turn(text: str, tools: list[dict]) -> dict:
    """解析一轮策略输出：{"content": 工具调用之前的文字, "calls": [{"name", "arguments"}], "error": 格式问题或 None}。"""
    if IM_END in text:
        text = text.split(IM_END)[0]
    schemas = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    calls, err = [], None
    for m in _CALL.finditer(text):
        name, body = m.group(1), m.group(2)
        if name not in schemas:
            err = f"unknown tool {name}"
        args = {k: _cast(v, schemas.get(name, {}).get(k)) for k, v in _PARAM.findall(body)}
        calls.append({"name": name, "arguments": args})
    if "<tool_call>" in text and not calls:
        err = "malformed tool call"
    content = text.split("<tool_call>")[0].strip()
    return {"content": content, "calls": calls, "error": err}
