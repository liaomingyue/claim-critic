from types import SimpleNamespace

from common.llm_anthropic import FALLBACK_BETA, ClaudeChat


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None, _request_id="req_x",
                               usage=SimpleNamespace(input_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=2),
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text="<verdict>faithful</verdict>")])


def make(stop_reason="end_turn"):
    chat = ClaudeChat.__new__(ClaudeChat)
    chat.model, chat.max_tokens, chat.effort = "claude-opus-5-5", 16000, "medium"
    chat.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(stop_reason)))
    return chat


def test_system_lifted_temperature_dropped_and_n_looped():
    chat = make()
    outs = chat.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], n=2, temperature=0.7)
    calls = chat.client.beta.messages.calls
    assert outs == ["<verdict>faithful</verdict>"] * 2 and len(calls) == 2
    p = calls[0]
    assert p["system"] == "S" and p["messages"] == [{"role": "user", "content": "U"}]
    assert "temperature" not in p and p["output_config"] == {"effort": "medium"}
    assert p["betas"] == [FALLBACK_BETA] and p["fallbacks"] == "default"


def test_refusal_returns_empty_string():
    assert make("refusal").chat([{"role": "user", "content": "U"}]) == [""]


def test_claude_cli_command_disables_mcp_settings_and_tools(tmp_path):
    from common.llm_claude_cli import ClaudeCLIChat

    cmd = ClaudeCLIChat("claude-opus-5-5", "high", workdir=str(tmp_path)).command("S")
    joined = " ".join(cmd)
    assert "--strict-mcp-config" in cmd and '{"mcpServers":{}}' in cmd
    assert "--setting-sources project" in joined and cmd[cmd.index("--tools") + 1] == ""
    assert "--effort high" in joined and cmd[-2:] == ["--system-prompt", "S"]
    assert ClaudeCLIChat("m", "low", workdir=str(tmp_path)).command("")[-1]  # 没给 system 也要传，避免默认 system prompt


def test_trace_records_prompt_output_and_item(tmp_path, monkeypatch):
    from common import trace
    from common.llm import LLM, LLMConfig, map_resumable

    monkeypatch.setenv("CRITIC_TRACE_DIR", str(tmp_path / "tr"))
    llm = LLM.__new__(LLM)
    llm.cfg = LLMConfig(model="m", provider="claude_cli")
    llm.claude = SimpleNamespace(chat=lambda messages, n=1, **kw: ["OUT"] * n)
    map_resumable([{"id": "x1"}], lambda it: {"o": llm.chat_one([{"role": "system", "content": "S"},
                                                                  {"role": "user", "content": "U"}])},
                  tmp_path / "reviews.jsonl")
    rows = trace.load(tmp_path / "tr" / "reviews.jsonl")["x1"]
    assert rows[0]["system"] == "S" and rows[0]["user"] == "U" and rows[0]["outputs"] == ["OUT"]
    assert rows[0]["stage"] == "reviews" and rows[0]["model"] == "m"
