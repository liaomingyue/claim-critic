"""模型调用封装。provider=openai：OpenAI 兼容接口（本地 vLLM、各家兼容 API）；provider=anthropic：Claude API（common/llm_anthropic.py）；provider=claude_cli：本机 Claude Code 的 claude -p（common/llm_claude_cli.py）。

批量调用支持断点续跑：结果逐条追加到输出文件，重启时跳过已完成的 id。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from common import trace
from common.schema import append_jsonl, read_jsonl


@dataclass
class LLMConfig:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    n: int = 1
    concurrency: int = 16
    max_retries: int = 5
    timeout: float = 120.0       # openai 后端单次请求超时（秒），超时后按重试处理
    provider: str = "openai"     # openai | anthropic | claude_cli
    effort: str = "medium"       # anthropic / claude_cli：low | medium | high | xhigh | max

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if cfg.provider == "anthropic":
            from common.llm_anthropic import ClaudeChat

            self.claude = ClaudeChat(cfg.model, cfg.api_key_env, cfg.max_tokens, cfg.effort, cfg.max_retries)
            return
        if cfg.provider == "claude_cli":
            from common.llm_claude_cli import ClaudeCLIChat

            self.claude = ClaudeCLIChat(cfg.model, cfg.effort)
            return
        from openai import OpenAI  # 延迟导入，单测不依赖

        self.claude = None
        # 单次请求超时：默认 600 s 太长，WSL 端口转发断连时请求会挂住不报错，整批卡 10 分钟
        self.client = OpenAI(base_url=cfg.base_url, timeout=cfg.timeout,
                             api_key=os.environ.get(cfg.api_key_env, "EMPTY"))

    def chat(self, messages: list[dict], **overrides) -> list[str]:
        """返回 n 个补全，并写调用记录（common.trace）。"""
        t0 = time.time()
        trace.reset_usage()  # 上一次调用失败时残留的用量不算到这次
        outs = self._chat(messages, **overrides)
        trace.record(self.cfg.model, self.cfg.provider, messages, outs, time.time() - t0)
        return outs

    def _chat(self, messages: list[dict], **overrides) -> list[str]:
        """失败时指数退避重试（anthropic 由 SDK 自己重试 429/5xx，4xx 直接抛出）。"""
        if self.claude is not None:
            return self.claude.chat(messages, n=overrides.pop("n", self.cfg.n), **overrides)
        params = dict(model=self.cfg.model, messages=messages,
                      temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
                      n=self.cfg.n)
        params.update(overrides)
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(**params)
                return [c.message.content or "" for c in resp.choices]
            except Exception as e:  # noqa: BLE001 网络、限流等都重试
                if attempt == self.cfg.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[llm] {type(e).__name__}: {e}; retry in {wait}s")
                time.sleep(wait)
        return []

    def chat_one(self, messages: list[dict], **overrides) -> str:
        return self.chat(messages, n=1, **overrides)[0]


def map_resumable(items: Iterable[dict], fn: Callable[[dict], dict | list[dict] | None],
                  out_path: str | Path, key: str = "id", concurrency: int = 16) -> None:
    """对每个 item 调用 fn 并把结果追加写入 out_path。已在 out_path 中出现过 key 的 item 跳过。

    fn 返回 None 表示该条丢弃（同样记一条 {key, _dropped: true}，避免重跑）。
    """
    out_path = Path(out_path)
    done = {r[key] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it[key] not in done]
    print(f"[map] {len(done)} done, {len(todo)} to go -> {out_path}")

    def run(it: dict) -> list[dict]:
        try:
            with trace.item(out_path.stem.split(".")[0], str(it[key])):
                res = fn(it)
        except Exception as e:  # noqa: BLE001 单条失败不影响整批，记录后跳过
            print(f"[map] {it[key]} failed: {type(e).__name__}: {e}")
            return []
        if res is None:
            return [{key: it[key], "_dropped": True}]
        rows = res if isinstance(res, list) else [res]
        for r in rows:
            r.setdefault(key, it[key])
        return rows

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run, it) for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                append_jsonl(out_path, row)
            if i % 100 == 0:
                print(f"[map] {i}/{len(todo)}")


def kept(path: str | Path) -> list[dict]:
    """读取 map_resumable 的输出，去掉被丢弃的占位行。"""
    return [r for r in read_jsonl(path) if not r.get("_dropped")]
