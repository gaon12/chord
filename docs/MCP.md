# MCP servers (mcp.json)

External tools come from [Model Context Protocol](https://modelcontextprotocol.io)
servers listed in `mcp.json` (copy `mcp.json.sample`). The bot connects to
every server on startup, adapts each tool into the shared registry as
`<server>_<tool>`, and hot-reloads changes automatically.

## Config format

```jsonc
{
  "mcpServers": {
    // Remote HTTP server with an authenticated header.
    "keenable": {
      "url": "https://api.keenable.ai/mcp",
      "headers": { "X-API-Key": "${KEENABLE_API_KEY}" }
    },

    // Local stdio server: any executable plus args.
    "sqlite": {
      "command": "${PYTHON}",
      "args": ["tools/sqlite_mcp_server.py", "--db-path", "chord.db"]
    }
  }
}
```

Rules:

* A server needs either `"command"` (stdio) or `"url"` (streamable HTTP).
* `${VAR}` placeholders are replaced from real environment variables and
  from `.env` settings - keep secrets out of this file. Unresolved
  placeholders stay verbatim and are logged.
* `"${PYTHON}"` expands to the bot's own interpreter (`sys.executable`) -
  portable across Windows/Linux/macOS venv layouts.
* On Windows, bare node launchers (`npx`, `npm`) are auto-wrapped as
  `cmd /c npx ...`; write just `"npx"` everywhere.
* A relative path command (`"tools/rhwp/rhwp.exe"`) is resolved against
  the working directory before spawning. Windows' `CreateProcess`
  rejects relative forward-slash paths and the server would otherwise
  die with a confusing `[WinError 2] file not found`. Bare program names
  are left alone so `PATH` lookup still applies.

## Bundled servers

| Server | What it adds | Setup notes |
|---|---|---|
| `keenable` | live web search + page fetch | needs `KEENABLE_API_KEY` |
| `korean-law` | 12 legal tools over 법제처 APIs (statutes, precedents, citation verification) | public server is rate-shared; append `?oc=<key>` from open.law.go.kr for an own credential |
| `playwright` | browser automation: navigate / snapshot / click / screenshot | first run downloads the package; browsers via `npx playwright install chromium` |
| `sqlite` | persistent memory for the LLM (`db_tables` / `db_query` / `db_execute` on `chord.db`) | runs inside the project venv, zero extra installs |
| `rhwp` | HWP/HWPX reading, searching, form filling, PDF/text/SVG export | download a release binary from <https://github.com/edwardkim/rhwp/releases>; Windows path expected at `tools/rhwp/rhwp/rhwp.exe`, macOS/Linux adjust `command` accordingly |

## Lifecycle

* Servers start in `ChordBot.setup_hook()` before the gateway connects.
* Each server lives in its own asyncio task that owns its whole
  context-manager lifetime (anyio cancel scopes require enter/exit in the
  same task).
* One failing server is skipped with a warning - it never blocks startup
  or the other servers.
* A background loop re-reads `mcp.json` every **30 minutes**; edits are
  applied automatically (old tools unregistered, new ones registered).
  Disable everything with `MCP_ENABLED=false`.
