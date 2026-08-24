# Development guide

## Setup (any OS)

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -e ".[dev]"

# macOS / Linux
.venv/bin/pip install -e ".[dev]"
```

Copy `.env.sample` to `.env` and fill in `DISCORD_TOKEN` +
`OPENAI_API_KEY`. Everything else is optional.

## Run

```bash
python -m chord      # or: chord
```

`!usage` inside Discord shows remaining provider quotas at any time.

## Quality gates

```bash
ruff format src tests   # formatting
ruff check src tests    # linting
pytest                  # full suite, offline
```

The suite never touches the network:

* HTTP is mocked with `respx`
* the LLM is a small fake (`tests/fakes.py`)
* MCP servers use lifecycle fakes (`tests/test_mcp_client.py`)
* each test gets an isolated quota store via the autouse fixture in
  `tests/conftest.py` (sets `QUOTA_STORE_PATH` to a temp file)

Keep it that way: probe a live endpoint once in a scratch script, then
encode the observed response shape as a respx fixture.

## Project layout

```
src/chord/
  config.py         Settings (pydantic-settings, .env -> typed fields)
  bot.py            Discord events, commands, composition root, MCP loop
  engine.py         one turn = LLM <-> tools loop (max 6 rounds)
  conversation.py   per-channel history (RAM only)
  llm.py            AsyncOpenAI facade (base_url configurable)
  mcp_client.py     mcp.json -> sessions -> Skill adapters (+ hot reload)
  skills/
    base.py         Skill -> OpenAI tool definition
    registry.py     collection + total (never-raising) execution
    _http/_geo      shared request + geocoding helpers
    _quota          usage counters / limits / usage.json persistence
    <topic>.py      one module per skill (auto-discovered)
tools/
  sqlite_mcp_server.py   in-repo SQLite MCP stdio server
docs/                SKILLS.md / MCP.md / this file
tests/               mirror of src, fully mocked
```

## Design rules

1. The engine knows nothing about Discord or MCP - it sees an LLM facade
   and a registry.
2. Tool execution is *total*: unknown tools, bad arguments, exhausted
   quotas and upstream failures become readable text for the model.
3. Multi-source skills prefer official/Korean providers and fall back to
   key-less services on any failure (quota exhaustion included).
4. Adding a skill or MCP server must not require touching existing files:
   skills are drop-in modules, MCP servers are drop-in JSON entries.

## Cross-platform notes

* Node launchers in `mcp.json` (`npx`, `npm`) are auto-wrapped as
  `cmd /c ...` on Windows; write bare names everywhere.
* `"${PYTHON}"` in a command expands to the running interpreter -
  portable across `.venv\Scripts` vs `.venv/bin` layouts.
* The tz database ships via the `tzdata` dependency (Windows has none).

CI runs the same gates on Ubuntu, Windows and macOS via
`.github/workflows/tests.yml`.
