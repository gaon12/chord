"""Tests for the MCP client - config parsing and tool adapters.

Real MCP sessions (subprocesses / HTTP) are never started here; the
manager's connector path is exercised through fakes, while config
parsing and the Skill adapter are tested directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import chord.mcp_client as mcp_client
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


# -- ${VAR} expansion -----------------------------------------------------------------


def test_env_placeholders_expanded_from_map(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"keenable": {"url": "https://api.test/mcp", '
        '"headers": {"X-API-Key": "${KEENABLE_API_KEY}"}}}}',
        encoding="utf-8",
    )
    specs = load_server_specs(config, env={"KEENABLE_API_KEY": "keen_secret"})
    assert specs["keenable"]["headers"]["X-API-Key"] == "keen_secret"


def test_unset_placeholder_left_as_is(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"s": {"url": "https://x/${MISSING_VAR}"}}}',
        encoding="utf-8",
    )
    specs = load_server_specs(config, env={})
    assert specs["s"]["url"] == "https://x/${MISSING_VAR}"


def test_real_environment_used_as_fallback(tmp_path, monkeypatch):
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"s": {"command": "run", "args": ["${MY_TOKEN}"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_TOKEN", "from-env")
    specs = load_server_specs(config, env={})
    assert specs["s"]["args"] == ["from-env"]


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


def _write_config(path, servers_json: str):
    path.write_text('{"mcpServers": ' + servers_json + "}", encoding="utf-8")


async def fake_serve(self, server_name, spec, ready, stop_event):
    """Lifecycle stand-in: one tool per server, parks until stopped."""
    from types import SimpleNamespace

    session = SimpleNamespace(exited=False)

    async def fake_exit(*exc):
        session.exited = True

    session.__aexit__ = fake_exit
    from types import SimpleNamespace as NS

    tool = NS(
        name=f"{server_name}_tool",
        description="fake",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )
    ready.set_result((session, [tool]))
    await stop_event.wait()


async def test_reload_detects_no_change_without_edit(tmp_path, monkeypatch):
    config = tmp_path / "mcp.json"
    _write_config(config, '{"a": {"command": "x"}}')
    settings = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        mcp_config_path=config,
    )

    monkeypatch.setattr(mcp_client.McpManager, "_serve", fake_serve)
    manager = McpManager()

    registered: list = []
    await manager.start(settings, registered.append)
    assert len(registered) == 1

    changed = await manager.reload_if_changed(settings, {}, registered.append)

    assert changed is False
    assert len(registered) == 1  # nothing re-registered
    await manager.stop()


async def test_reload_swaps_tools_when_config_changes(tmp_path, monkeypatch):
    config = tmp_path / "mcp.json"
    _write_config(config, '{"a": {"command": "x"}}')
    settings = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        mcp_config_path=config,
    )

    monkeypatch.setattr(mcp_client.McpManager, "_serve", fake_serve)

    # Config source controlled by the test.
    servers = {"a": {"command": "x"}}
    monkeypatch.setattr(
        mcp_client,
        "load_server_specs",
        lambda path, env=None: {name: dict(spec) for name, spec in servers.items()},
    )

    manager = McpManager()

    class RegistryLike(dict):
        def unregister(self, name):
            return self.pop(name, None) is not None

    registry_like = RegistryLike()
    registered: list = []

    def register(skill):
        registered.append(skill)
        registry_like[skill.name] = skill

    count = await manager.start(settings, register)
    assert count == 1
    assert [s.name for s in registered] == ["a_a_tool"]

    # Simulate an edit adding a second server.
    servers["b"] = {"url": "https://x.test/mcp"}
    _write_config(config, '{"a": {"command": "x"}, "b": {"url": "https://x.test/mcp"}}')

    changed = await manager.reload_if_changed(settings, registry_like, register)

    assert changed is True
    # Old 'a' instance was unregistered from the registry, then both
    # servers registered fresh instances.
    assert sorted(registry_like) == ["a_a_tool", "b_b_tool"]
    assert registry_like["a_a_tool"] is not registered[0]  # fresh instance


async def test_registry_unregister_roundtrip():
    from chord.skills.base import Skill
    from chord.skills.registry import SkillRegistry

    class Dummy(Skill):
        name = "dummy"

        async def run(self):
            return ""

    registry = SkillRegistry()
    dummy = Dummy()
    registry.register(dummy)
    assert registry.unregister("dummy") is True
    assert registry.unregister("dummy") is False
    assert "dummy" not in registry


# -- Cross-platform command resolution ------------------------------------------------


def test_resolve_command_wraps_node_launchers_on_windows():
    from chord.mcp_client import resolve_command

    assert resolve_command("npx", ["-y", "pkg"], windows=True) == (
        "cmd",
        ["/c", "npx", "-y", "pkg"],
    )
    # Real executables are never wrapped.
    assert resolve_command("tools/x.exe", [], windows=True) == ("tools/x.exe", [])


def test_resolve_command_passthrough_on_posix():
    from chord.mcp_client import resolve_command

    assert resolve_command("npx", ["-y", "pkg"], windows=False) == ("npx", ["-y", "pkg"])


def test_python_placeholder_expands_to_running_interpreter(tmp_path):
    import sys as _sys

    from chord.mcp_client import load_server_specs

    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers": {"s": {"command": "${PYTHON}", "args": ["server.py"]}}}',
        encoding="utf-8",
    )
    specs = load_server_specs(config)
    assert specs["s"]["command"] == _sys.executable
