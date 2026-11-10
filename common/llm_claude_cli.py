"""通过 Claude Code 命令行（claude -p）调用 Claude，由 common.llm.LLM 在 provider: claude_cli 时使用。

用本机已登录的 Claude Code（claude.ai 订阅登录时占订阅额度，不另外计费），适合小规模试跑；
正式规模受额度限制，换 provider: anthropic。

要点（在 Claude Code 2.1.309 上实测）：
- 总是传 --system-prompt（调用方没给就用一句最短的），否则套用 Claude Code 默认 system prompt，每次多约 2,200 token。
- 关掉 MCP 连接器（--strict-mcp-config + 空 mcp 配置）、用户级设置（--setting-sources project）和 skills，
  并在空目录里运行；否则每次调用额外带约 8 万 token 的工具定义和上下文，关掉后约 600 token。
- --tools "" 禁用全部工具，--no-session-persistence 不落盘会话。
- prompt 走 stdin、以 UTF-8 字节传入，输出按 UTF-8 解码；Windows 下经 shell 管道中文会乱码。
- 不支持 temperature 与 n>1：忽略 temperature，n 次循环调用。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from common import trace

EMPTY_MCP = '{"mcpServers":{}}'
MINIMAL_SYSTEM = "Follow the user's instructions exactly."


class ClaudeCLIChat:
    def __init__(self, model: str, effort: str, workdir: str | None = None, timeout: int = 600):
        self.model = model
        self.effort = effort
        self.timeout = timeout
        # 空目录：不让 claude -p 读到项目里的 CLAUDE.md、记忆等上下文
        self.workdir = Path(workdir or os.environ.get("CLAUDE_CLI_WORKDIR", Path.home() / ".claude-cli-empty"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def command(self, system: str) -> list[str]:
        # 必须总是传 --system-prompt：不传时 claude -p 套用 Claude Code 自己的默认 system prompt，
        # 每次多约 2,200 token（实测 2,760 vs 540）；剩下约 540 token 是 claude -p 自带的最小开销。
        return ["claude", "-p", "--model", self.model, "--effort", self.effort,
                "--tools", "", "--no-session-persistence", "--output-format", "json",
                "--strict-mcp-config", "--mcp-config", EMPTY_MCP, "--setting-sources", "project",
                "--disable-slash-commands", "--system-prompt", system or MINIMAL_SYSTEM]

    def run_once(self, system: str, prompt: str) -> str:
        r = subprocess.run(self.command(system), input=prompt.encode("utf-8"), capture_output=True,
                           cwd=self.workdir, timeout=self.timeout)
        out = r.stdout.decode("utf-8", errors="replace")
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p failed (exit {r.returncode}): "
                               f"{r.stderr.decode('utf-8', errors='replace')[:500]} {out[:200]}")
        if d.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(d.get('result'))[:500]}")
        u = d.get("usage") or {}
        trace.add_usage({"input_tokens": u.get("input_tokens", 0),
                         "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                         "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
                         "output_tokens": u.get("output_tokens", 0),
                         "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                         "cost_usd_list": d.get("total_cost_usd", 0.0)})  # 订阅登录时不实际计费，是按标价折算
        if d.get("stop_reason") == "refusal":
            print(f"[claude_cli] refusal; session {d.get('session_id')}")
            return ""
        return d.get("result") or ""

    def chat(self, messages: list[dict], n: int = 1, **_ignored) -> list[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if len(turns) != 1 or turns[0]["role"] != "user":
            raise ValueError("claude_cli backend supports a single user turn (plus optional system)")
        return [self.run_once(system, turns[0]["content"]) for _ in range(n)]
