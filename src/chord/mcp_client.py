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
    """Starts MCP sessions at startup and adapts their tools as skills."""

    def __init__(self) -> None:
        self._sessions: list[Any] = []
        self._http_clients: list[Any] = []

    def _env_map(self, settings: Settings) -> dict[str, str]:
        """Variables usable as ${VAR} inside mcp.json.

        Real environment variables win; string-valued settings (API keys
        loaded from .env) are exposed under their field names too.
        """
        env_map = {
            key: value
            for key, value in settings.model_dump().items()
            if isinstance(value, str) and value
        }
        return {**env_map}

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

        registered = 0
        for server_name, spec in specs.items():
            try:
                if "url" in spec:
                    # Custom headers (e.g. X-API-Key) ride on a dedicated
                    # http client that we close again in stop().
                    headers = spec.get("headers") or None
                    http_client = create_mcp_http_client(headers=headers)
                    streams = await streamable_http_client(
                        spec["url"], http_client=http_client
                    ).__aenter__()
                    read, write, _ = streams
                    self._http_clients.append(http_client)
                else:
                    params = StdioServerParameters(
                        command=spec["command"],
                        args=spec.get("args", []),
                        env=spec.get("env"),
                    )
                    read, write = await stdio_client(params).__aenter__()

                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()

                tools_response = await session.list_tools()
                for tool in tools_response.tools or []:
                    register(McpTool(server_name, session, tool))
                    registered += 1
                self._sessions.append(session)
                logger.info(
                    "MCP server %s connected with %d tool(s).",
                    server_name,
                    len(tools_response.tools or []),
                )
            except Exception as exc:  # noqa: BLE001 - one bad server != outage
                logger.warning(
                    "MCP server %s failed to start (%s) - skipped.",
                    server_name,
                    exc,
                )
        return registered

    async def stop(self) -> None:
        """Close every live session, ignoring shutdown errors."""
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.warning("Error while closing an MCP session.", exc_info=True)
        self._sessions.clear()

        for http_client in self._http_clients:
            try:
                await http_client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("Error while closing an MCP HTTP client.", exc_info=True)
        self._http_clients.clear()
