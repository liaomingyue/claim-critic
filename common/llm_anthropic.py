"""Claude（Anthropic Messages API）后端，由 common.llm.LLM 在 provider: anthropic 时调用。

与 OpenAI 兼容接口的差别在这里抹平：
- system 消息提到顶层 system 参数；
- 不支持 n>1，循环调用；
- 不再接受 temperature（Messages API 对所有模型返回 400），调用方传的 temperature 直接忽略，
  多样性靠 prompt；
- 思考不能关（Opus 5.5），用 output_config.effort 控制深浅，max_tokens 要留出思考的余量；
- 开启服务端 refusal fallback（fallbacks="default"），被安全分类器拒答时由 API 自动换模型重试；
  整条链都拒答时返回空串并打日志，调用方按「无结果」处理。
"""
from __future__ import annotations

import os

from common import trace

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeChat:
    def __init__(self, model: str, api_key_env: str, max_tokens: int, effort: str, max_retries: int):
        import anthropic  # 延迟导入，单测和只用 vLLM 的环境不依赖

        # api_key_env 未设置时交给 SDK 自己找凭据（ANTHROPIC_API_KEY / ant auth login 的 profile）
        key = os.environ.get(api_key_env) if api_key_env else None
        self.client = anthropic.Anthropic(api_key=key, max_retries=max_retries) if key \
            else anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def chat(self, messages: list[dict], n: int = 1, max_tokens: int | None = None, **_ignored) -> list[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        params = dict(model=self.model, max_tokens=max_tokens or self.max_tokens, messages=turns,
                      output_config={"effort": self.effort},
                      betas=[FALLBACK_BETA], fallbacks="default")
        if system:
            params["system"] = system
        outs = []
        for _ in range(n):
            resp = self.client.beta.messages.create(**params)
            u = resp.usage
            trace.add_usage({"input_tokens": u.input_tokens or 0,
                             "cache_read_tokens": u.cache_read_input_tokens or 0,
                             "cache_write_tokens": u.cache_creation_input_tokens or 0,
                             "output_tokens": u.output_tokens or 0})
            if resp.stop_reason == "refusal":
                category = resp.stop_details.category if resp.stop_details else None
                print(f"[claude] refusal ({category}); request {resp._request_id}")
                outs.append("")
                continue
            if resp.stop_reason == "max_tokens":
                print(f"[claude] hit max_tokens={params['max_tokens']}; request {resp._request_id}")
            outs.append("".join(b.text for b in resp.content if b.type == "text"))
        return outs
