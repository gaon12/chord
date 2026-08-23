"""Tests for the MCP client - config parsing and tool adapters.

Real MCP sessions (subprocesses / HTTP) are never started here; the
manager's connector path is exercised through fakes, while config
parsing and the Skill adapter are tested directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chord.config import Settings
from chord.mcp_client import (
    McpManager,
    McpTool,
    extract_text,
    load_server_specs,
    sanitize_tool_name,
)


def _settings(tmp_path, **keys) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        mcp_config_path=tmp_path / "mcp.json",
        **keys,
    )


# -- Config parsing -----------------------------------------------------------------


def test_missing_config_file_means_no_servers(tmp_path):
    assert load_server_specs(tmp_path / "missing.json") == {}


def test_valid_config_is_parsed(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"fetch": {"command": "uvx", "args": ["x"]}, '
        '"remote": {"url": "https://example.com/mcp"}}}',
        encoding="utf-8",
    )
    specs = load_server_specs(config)
    assert set(specs) == {"fetch", "remote"}
    assert specs["fetch"]["command"] == "uvx"
    assert specs["remote"]["url"] == "https://example.com/mcp"


def test_invalid_entries_are_skipped(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"broken": {"foo": 1}, "not_json_ok": "x"}}',
        encoding="utf-8",
    )
    assert load_server_specs(config) == {}


def test_broken_json_returns_empty(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text("{ not json", encoding="utf-8")
    assert load_server_specs(config) == {}


# -- Tool adapter ---------------------------------------------------------------------


def _fake_tool(name="fetch", description="Fetch a URL."):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )


class FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"result of {name}: {arguments}")],
            isError=False,
        )


def test_mcp_tool_adapter_name_and_schema():
    session = FakeSession()
    tool = McpTool("myserver", session, _fake_tool())

    assert tool.name == "myserver_fetch"
    assert tool.description == "Fetch a URL."
    definition = tool.to_openai_tool()["function"]
    assert definition["parameters"]["required"] == ["url"]


async def test_mcp_tool_run_calls_remote_and_returns_text():
    session = FakeSession()
    tool = McpTool("srv", session, _fake_tool())

    result = await tool.run(url="https://example.com")

    assert session.calls == [("fetch", {"url": "https://example.com"})]
    assert result == "result of fetch: {'url': 'https://example.com'}"


def test_extract_text_handles_empty_content():
    empty = SimpleNamespace(content=[], isError=False)
    assert extract_text(empty) == ""
    errored = SimpleNamespace(content=[], isError=True)
    assert "Error" in extract_text(errored)


def test_sanitize_tool_name():
    # Hyphens are legal in tool names, so only spaces/punctuation change.
    assert sanitize_tool_name("My Server!", "do-thing") == "My_Server__do-thing"


# -- Manager lifecycle ------------------------------------------------------------------


async def test_manager_disabled_registers_nothing(tmp_path):
    settings = _settings(tmp_path, mcp_enabled=False)
    registered: list = []
    manager = McpManager()
    assert await manager.start(settings, registered.append) == 0
    assert registered == []


@pytest.mark.asyncio
async def test_manager_skips_failing_servers(monkeypatch, tmp_path):
    """A server that fails to start is skipped without raising."""
    config = tmp_path / "mcp.json"
    config.write_text('{"mcpServers": {"boom": {"command": "nope"}}}', encoding="utf-8")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        mcp_config_path=config,
    )

    import chord.mcp_client as mcp_client

    class BrokenTransport:
        def __aenter__(self):
            raise RuntimeError("cannot spawn")

    def broken_stdio(params):
        return BrokenTransport()

    monkeypatch.setattr(mcp_client, "stdio_client", broken_stdio)

    manager = McpManager()
    registered: list = []
    count = await manager.start(settings, registered.append)

    assert count == 0
    assert registered == []
