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
`OPENAI_API_KEY`, or use `LLM_PROVIDER=groq` and `GROQ_API_KEY` instead.
See [Groq setup](GROQ.md) for model fallback and reasoning settings.

## Run

```bash
python -m chord      # or: chord
```

`/usage` inside Discord shows remaining provider quotas at any time.

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
  bot.py            Discord events, composition root, background loops
  discord_commands.py  slash commands, authorization, runtime settings
  discord_text.py   incoming message context and reply cleanup
  message_splitter.py  message limits and balanced code fences
  channel_turns.py  per-channel turn serialization
  engine.py         one turn = LLM <-> tools loop (max 6 rounds)
  tool_catalog.py   per-turn schema selection and offered-tool checks
  conversation.py   per-channel history (RAM only)
  llm.py            shared completion facade and client lifecycle
  llm_options.py    optional request fields for compatible providers
  groq_provider.py  ordered free-model fallback and reasoning mapping
  groq_limits.py    rate headers, cooldowns, and usage descriptions
  mcp_client.py     mcp.json -> sessions -> Skill adapters (+ hot reload)
  skills/
    base.py         Skill -> OpenAI tool definition
    registry.py     collection + total (never-raising) execution
    _http/_geo      shared request + geocoding helpers
    _quota          usage counters / limits / usage.json persistence
    <topic>.py      one module per skill (auto-discovered)
src/chord/sqlite_server.py   optional, bounded read-only SQLite MCP server
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


## Discord state and permissions

Each channel processes one complete turn at a time, including history updates
and compaction. Different channels can proceed independently. `/reset` uses
the same boundary so a pending reply cannot restore cleared history.
Idle channel locks are released, and conversation storage keeps at most 1000
recently updated channels. Reading an unknown channel creates no stored entry.

Model replies and reminder deliveries cannot ping users, roles, or everyone.
Incoming messages keep code indentation and line breaks. Long outgoing code
blocks are closed and reopened in each message with their language labels.

Only the Discord application owner (or application team owner) can view or
reload the persona or change the global reasoning setting, including in DMs.
Other users may view the reasoning setting. Resetting a guild conversation
requires Manage Messages permission or application ownership; DM resets affect
only that DM conversation. These checks are enforced by command callbacks.


## Tool schemas and parallel calls

`TOOL_CATALOG_MODE=auto` offers schemas on demand for Groq free mode; other
setups keep the full catalog. Selection is local to each turn. A model cannot
execute a guessed tool that was absent from the current request, even if it
selects that tool in the same completion.

A skill may set `parallel_safe = True` only after review confirms that calls
are independent reads. Adjacent safe calls run with a concurrency limit of
four. Every other call is an ordering barrier. MCP tools remain sequential,
regardless of remote read-only hints. Error results retain their positions
without cancelling successful sibling lookups.


## Sensitive errors and network destinations

Application error logs record operation names and exception types without
private request text, signed URLs, or raw provider error bodies. Only errors
that identify an invalid assistant/tool history exchange clear stored history;
a bad model name or tool schema does not erase a valid conversation.

Untrusted URL connections reject private, reserved, multicast, site-local,
and private addresses embedded in IPv6 transition forms. DNS checks run off
the event loop, and connections use checked numeric IPs while retaining the
original TLS hostname and certificate verification. Regression tests cover
both the destination socket and the HTTPS Host/SNI behavior.

Runtime database files and their SQLite journals are ignored by Git. The
legacy local SQLite MCP script is unsupported; remove it from existing
configuration and enable the packaged read-only server only for a dedicated,
shareable database. Existing reminder databases do not need a migration.
