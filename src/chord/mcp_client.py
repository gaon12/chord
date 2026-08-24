"""MCP (Model Context Protocol) client - external tool servers.

Reads an ``mcp.json`` file (see ``mcp.json.sample``) describing servers
in the same shape most MCP clients use:

    {
      "mcpServers": {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "remote": {"url": "https://example.com/mcp"}
      }
    }

``stdio`` servers (``command``) are spawned as subprocesses; ``http``
servers (``url``) connect over streamable HTTP. Every tool a server
exposes is adapted into a normal :class:`chord.skills.base.Skill` and
registered into the shared registry, so the LLM sees built-in skills
and MCP tools side by side with no special handling anywhere else.

A broken or unreachable server is skipped with a warning - it must
never take the whole bot down.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:  # mcp >= 2.0 renamed the HTTP transport helper.
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )
except ImportError:  # pragma: no cover - older SDKs
    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

    def create_mcp_http_client(headers=None, **kwargs):  # type: ignore[misc]
        import httpx

        return httpx.AsyncClient(headers=headers)


from chord.config import Settings
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

#: MCP tool names may collide across servers; prefix keeps them unique.
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(server_name: str, tool_name: str) -> str:
    """Build a unique, LLM-safe tool name like ``fetch_fetch``."""
    clean_server = _INVALID_NAME_CHARS.sub("_", server_name)
    clean_tool = _INVALID_NAME_CHARS.sub("_", tool_name)
    return f"{clean_server}_{clean_tool}"[:64]


class McpTool(Skill):
    """One remote MCP tool wrapped as a local skill."""

    def __init__(self, server_name: str, session: Any, tool: Any) -> None:
        self._server_name = server_name
        self._session = session
        self._tool_name = tool.name
        self.name = sanitize_tool_name(server_name, tool.name)
        self.description = (getattr(tool, "description", "") or "").strip()
        schema = getattr(tool, "inputSchema", None)
        self.parameters: ClassVar[dict] = (
            schema
            if isinstance(schema, dict) and schema
            else {"type": "object", "properties": {}, "required": []}
        )

    async def run(self, **kwargs: Any) -> str:
        result = await self._session.call_tool(self._tool_name, arguments=kwargs)
        return extract_text(result)

    @property
    def server(self) -> str:
        return self._server_name


def extract_text(call_result: Any) -> str:
    """Pull readable text out of an MCP CallToolResult."""
    parts = []
    for item in getattr(call_result, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    if getattr(call_result, "isError", False):
        return "Error: the MCP tool failed."
    return ""


def load_server_specs(
    config_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Read and validate the mcp.json file; missing file means no servers.

    String values may reference environment variables with
    ``${VAR_NAME}`` (e.g. an API key kept out of the config file).
    Resolution order: explicit ``env`` map first, then ``os.environ``;
    unresolved placeholders are left as-is and logged.
    """
    if not config_path.exists():
        logger.info("MCP config %s not found - skipping MCP setup.", config_path)
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read MCP config %s: %s", config_path, exc)
        return {}

    merged_env = {**os.environ, **(env or {})}
    servers = data.get("mcpServers") or {}
    valid: dict[str, dict] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict) or not ("command" in spec or "url" in spec):
            logger.warning("MCP server %r ignored: needs either 'command' or 'url'.", name)
            continue
        valid[name] = _expand_env_placeholders(spec, merged_env)
    return valid


_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_placeholders(value: Any, env_map: dict[str, str]) -> Any:
    """Recursively replace ${VAR} in strings; unknown vars stay untouched."""

    def substitute(text: str) -> str:
        def replace(match: re.Match) -> str:
            var = match.group(1)
            if var in env_map:
                return env_map[var]
            logger.warning("MCP config references unset variable $%s.", var)
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace, text)

    if isinstance(value, str):
        return substitute(value)
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v, env_map) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item, env_map) for item in value]
    return value


class McpManager:
    """Starts MCP sessions at startup and adapts their tools as skills.

    Each server runs inside its own long-lived asyncio task that owns
    the full context-manager lifecycle (transport -> session -> close).
    This matters because MCP transports are anyio cancel scopes: they
    must be entered and exited from the same task, and Discord's
    ``setup_hook``/``close`` run in different tasks. ``stop()`` simply
    signals the per-server stop event and joins its task.

    ``reload_if_changed()`` lets callers (a periodic Discord task) pick
    up mcp.json edits at runtime without restarting the bot.
    """

    def __init__(self) -> None:
        self._servers: list[dict[str, Any]] = []
        self._tool_names: list[str] = []
        self._signature: str | None = None

    def _config_signature(self, settings: Settings) -> str:
        """Stable hash of the config content plus the enabled flag."""
        path = Path(settings.mcp_config_path)
        try:
            content = path.read_bytes()
        except OSError:
            content = b""
        return f"{settings.mcp_enabled}:{hashlib.sha256(content).hexdigest()}"

    def _env_map(self, settings: Settings) -> dict[str, str]:
        """Variables usable as ${VAR} inside mcp.json.

        String-valued settings (API keys loaded from .env) are exposed
        under both their field name ('keenable_api_key') and their
        conventional upper-case form ('KEENABLE_API_KEY').
        """
        env_map: dict[str, str] = {}
        for key, value in settings.model_dump().items():
            if isinstance(value, str) and value:
                env_map[key] = value
                env_map[key.upper()] = value
        return env_map

    async def start(self, settings: Settings, register) -> int:
        """Connect to all servers and register their tools.

        Args:
            settings: Application settings (config path, enabled flag).
            register: Callable accepting one Skill per MCP tool.

        Returns:
            How many tools were registered.
        """
        if not settings.mcp_enabled:
            logger.info("MCP disabled via settings.")
            return 0

        specs = load_server_specs(Path(settings.mcp_config_path), env=self._env_map(settings))
        if not specs:
            return 0

        self._signature = self._config_signature(settings)

        registered = 0
        for server_name, spec in specs.items():
            try:
                record = await self._launch(server_name, spec)
            except Exception as exc:  # noqa: BLE001 - one bad server != outage
                logger.warning("MCP server %s failed to start (%s) - skipped.", server_name, exc)
                continue

            session = record["session"]
            for tool in record["tools"]:
                adapter = McpTool(server_name, session, tool)
                register(adapter)
                self._tool_names.append(adapter.name)
                registered += 1
            self._servers.append(record)
            logger.info(
                "MCP server %s connected with %d tool(s).",
                server_name,
                len(record["tools"]),
            )
        return registered

    async def _launch(self, server_name: str, spec: dict) -> dict:
        """Start one server in a dedicated task; returns when it is ready."""
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            self._serve(server_name, spec, ready, stop_event),
            name=f"mcp-{server_name}",
        )

        try:
            session, tools = await ready
        except BaseException:
            task.cancel()
            raise

        return {
            "name": server_name,
            "task": task,
            "stop": stop_event,
            "session": session,
            "tools": tools or [],
        }

    async def _serve(
        self, server_name: str, spec: dict, ready: asyncio.Future, stop_event: asyncio.Event
    ) -> None:
        """Own one server's whole lifecycle inside this single task.

        Everything uses plain ``async with`` blocks so the transports
        (anyio cancel scopes) are entered AND exited here; the task then
        parks on ``stop_event.wait()`` until shutdown is requested.
        """
        try:
            if "url" in spec:
                # Custom headers (e.g. X-API-Key) ride on a dedicated
                # HTTP client scoped to this connection.
                headers = spec.get("headers") or None
                http_client = create_mcp_http_client(headers=headers)
                cm = streamable_http_client(spec["url"], http_client=http_client)
            else:
                params = StdioServerParameters(
                    command=spec["command"],
                    args=spec.get("args", []),
                    env=spec.get("env"),
                )
                cm = stdio_client(params)

            async with cm as opened:
                # mcp 2.x yields (read, write); 1.x also had get_session_id.
                read, write = opened[0], opened[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()

                    ready.set_result((session, list(tools_response.tools or [])))
                    await stop_event.wait()
                    # Leaving the async-with blocks shuts the transport
                    # down cleanly inside this very task.
        except BaseException as exc:  # noqa: BLE001 - reported via `ready`
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("MCP server %s crashed: %s", server_name, exc)

    async def stop(self) -> None:
        """Signal every server task to shut down and wait for it."""
        for record in self._servers:
            record["stop"].set()
            try:
                await asyncio.wait_for(asyncio.shield(record["task"]), timeout=10)
            except Exception:  # noqa: BLE001 - shutdown must never hang the bot
                record["task"].cancel()
                logger.warning("MCP server %s did not shut down cleanly.", record["name"])
        self._servers.clear()
        self._tool_names.clear()

    async def reload_if_changed(self, settings: Settings, registry, register) -> bool:
        """Restart MCP servers when mcp.json changed since the last load.

        Old tools are unregistered from ``registry`` first so stale tools
        never linger. Returns True when a reload actually happened.
        """
        signature = self._config_signature(settings)
        if signature == self._signature:
            return False

        logger.info("mcp.json changed - reloading MCP servers.")
        for name in self._tool_names:
            registry.unregister(name)
        await self.stop()
        await self.start(settings, register)
        return True
