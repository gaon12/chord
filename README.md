# chord

A Discord bot powered by **any OpenAI-compatible LLM** (OpenAI, OpenRouter,
Ollama, vLLM, ...), with pluggable built-in **skills**, support for external
**MCP** tool servers, and clean, chat-friendly answers.

Mention the bot in a server or DM it - it chats back and automatically calls
tools when a question needs real-world data.

---

## Features

| Area | Tool name | Provider (preferred) | Fallback without key |
|---|---|---|---|
| Weather | `get_weather` | 기상청 KMA (Korea) · WeatherAPI.com | Open-Meteo |
| Fine dust | `get_air_quality` | 에어코리아 AirKorea (Korea) | Open-Meteo CAMS |
| Exchange rates | `get_exchange_rate` | Frankfurter (ECB) | - (key-less) |
| Stock prices | `get_stock_price` | Yahoo Finance | - (key-less) |
| Parcel tracking | `track_parcel` | SweetTracker 스마트택배 | CJ/우체국 site scraping |
| Flight info | `get_flight_info` | Aviationstack | OpenSky radar + adsbdb |
| Web search | `web_search` | DuckDuckGo lite | - (key-less) |
| Places | `find_places` | Kakao Local (Korea) | OpenStreetMap Nominatim |
| Directions | `get_directions` | Kakao Navi (Korea) | OSRM demo server |
| Date/time & timezones | `get_current_datetime`, `convert_timezone` | pure Python | - |
| Unit conversion | `convert_units` | pure Python (incl. 평) | - |
| URL shortener | `shorten_url`, `expand_short_url` | lrl.kr API | needs `LRL_API_KEY` |
| URL safety | `check_url_safety` | Google Safe Browsing cache (lrl.kr) + Cloudflare 1.1.1.2 + Radar scan | Cloudflare DNS works key-less |
| Summarize | `summarize_text` | your configured LLM | - |
| Translate | `translate_text` | your configured LLM | - |
| ELI5 explainer | `explain_eli5` | your configured LLM (audience-calibrated) | - |
| Crypto prices | `get_crypto_price` | Upbit public ticker | - (key-less) |
| Wikipedia | `get_wiki_summary` | Korean Wikipedia API | - (key-less) |
| News headlines | `get_news` | 연합뉴스 RSS · Google News RSS | - (key-less) |
| Random utilities | `random_pick` | dice / coin / number / pick / shuffle | - |
| Reminders | `set_reminder`, `list_reminders` | pure Python (SQLite) | - |

Everything works **without any API keys** except the URL shortener; optional
keys unlock the official/premium providers listed above.

External **MCP** servers add more tools on top - see [MCP tools](#mcp-tools).

---

## Setup

### 1. Create the Discord application

1. Go to the [Discord developer portal](https://discord.com/developers/applications)
   and create an application, then a bot under it.
2. Copy the bot token.
3. On the *Bot* page enable **Message Content Intent** (required to read messages).

### 2. Install and configure

```powershell
git clone <this repo>
cd chord
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

Copy-Item .env.sample .env   # then fill in the values
```

Required in `.env`:

```ini
DISCORD_TOKEN=your-discord-bot-token
OPENAI_API_KEY=your-api-key
```

Optional but recommended:

```ini
OPENAI_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible server
OPENAI_MODEL=gpt-4o-mini                    # model understood by that server
REASONING_LEVEL=none                        # auto|none|light|medium|heavy
LLM_TIMEOUT_SECONDS=120                     # give up on a stalled provider
LOG_LEVEL=INFO                              # DEBUG explains every reply decision
KMA_API_KEY=...                             # 기상청 weather for Korea
AIRKOREA_API_KEY=...                        # 에어코리아 fine dust
SWEETTRACKER_API_KEY=...                    # aggregated parcel tracking
LRL_API_KEY=...                             # URL shortener
AVIATIONSTACK_API_KEY=...                   # scheduled flight data
WEATHERAPI_API_KEY=...                      # worldwide weather
KAKAO_REST_API_KEY=...                     # Korean places/navigation
KEENABLE_API_KEY=...                       # Keenable MCP live search
QUOTA_STORE_PATH=usage.json                # provider usage counters
MCP_CONFIG_PATH=mcp.json                    # external MCP servers
```

See `.env.sample` for the full list with signup links.

### 3. Run

```powershell
python -m chord        # or just `chord`, both start the bot
```

---

## Using the bot

* **Chat**: mention the bot anywhere (`@chord 서울 날씨 어때?`) or DM it.
  The LLM decides which skills to call; you can also ask naturally:
  *"5 km를 마일로 바꿔줘"*, *"지금 뉴욕 시간 몇 시야?"*, *"KE801 항공편 어디까지 왔어?"*
* `/help` - show usage.
* `/usage` - show remaining API quotas per provider.
* `/reminders` - list pending reminders in this channel.
* `/reset` - clear this channel's conversation memory.
* `/persona` - view or reload the character definition.
* `/reasoning` - view or change how hard the bot thinks before answering.

**Reminders**: ask naturally — *"30분 후 라면 끓어라고 알려줘"* or *"8월 25일 오후 2시에 회의"*
— and the bot posts the message back into the same channel at the right time.
The character is defined in `persona.md`; edit it and changes apply on the
very next message (no restart needed).

Conversations are kept per channel **in memory only** - restarting the bot
clears them, and nothing is persisted.

### Reasoning level

How much the model deliberates before replying is a setting, not a fixed
provider behaviour:

| Level    | `reasoning_effort` sent | Use it when |
|----------|-------------------------|-------------|
| `auto`   | *(nothing)*             | you want the provider's own default back |
| `none`   | `minimal`               | **default** - chat, quick answers |
| `light`  | `low`                   | occasional hard questions |
| `medium` | `medium`                | balanced |
| `heavy`  | `high`                  | analysis worth waiting for |

Set `REASONING_LEVEL` in `.env` for the startup value, or `/reasoning
<level>` to retune a running bot without dropping its MCP connections
and conversation history.

Models that have no notion of reasoning effort (`gpt-4o-mini`, most Gemma
and local models) reject the parameter outright. The bot notices on the
first request, logs a warning, and answers without it - so the setting is
always safe to leave on. `/reasoning` reports when this has happened
rather than showing a level that does nothing.

Chain-of-thought that a model prints into its answer (`<thought>...`,
`<think>...`) is stripped before sending, so the reasoning never reaches
the channel. Text inside fenced code blocks is left alone.

---

## Troubleshooting

**The bot ignores `@chord ...` in a server.** In order, check:

1. The bot is online in the member list. If the process exits at
   startup, the log says why.
2. **Message Content Intent** is enabled in the Developer Portal
   (*Bot → Privileged Gateway Intents*). Without it, mentions arrive
   with empty text; the bot warns about this on startup and answers the
   mention by telling you so.
3. The bot can actually write in that channel. Missing *Send Messages*
   used to make it look dead - it now logs
   `Not allowed to send in channel <id>` instead.
4. Set `LOG_LEVEL=DEBUG` in `.env` and watch the decision: every
   answered message logs its trigger (`reason=user-mention`,
   `role-mention`, `mention-token`, `reply-to-bot`, `dm`). Chatty
   libraries stay at INFO so your own lines remain readable.

Mentioning the bot's *role* instead of the bot user works too - that is
what Discord's autocomplete usually inserts. `@everyone` and `@here` are
deliberately ignored.

**Replies take 20-30 seconds, or the bot says it hit a usage limit.**
Almost always the input-token rate limit, not a slow model. Every tool
definition is re-sent with **every** request, and a turn that calls tools
sends the whole catalog again each round. Measured on this project with
`usage.prompt_tokens`:

| Tools offered | Schema size | Prompt tokens per request |
|---|---|---|
| 24 (built-in only) | 11 341 chars | 3 490 |
| 39 (default `mcp.json`) | 16 888 chars | 5 428 |
| 63 (+ playwright) | 22 427 chars | 6 356 |
| 145 (+ an 82-tool server) | 47 677 chars | 15 303 |

The Gemini free tier allows **16 000 input tokens per minute**, so at 145
tools a *single* greeting used 96% of the minute's budget and the next
request got:

```
429 - Quota exceeded for metric: generate_content_free_tier_input_token_count,
limit: 16000 ... Please retry in 18.4s
```

The SDK retries that with backoff, which is where the 20-30 second replies
come from. Fix it by trimming `mcp.json` rather than by waiting: the bot
warns at startup when its catalog is big enough to matter, and
`LOG_LEVEL=DEBUG` prints the actual token count of every request. Requests
are also capped by `LLM_TIMEOUT_SECONDS`, after which the bot says so
instead of staying silent.

**An MCP server is missing from the tool list.** A server that fails to
start is skipped with a warning so it cannot take the bot down - check
the startup log for `MCP server <name> failed to start`.

---

## MCP tools

Copy `mcp.json.sample` to `mcp.json` and list servers. String values may
reference environment variables as `${VAR_NAME}` - resolved from real
environment variables and from `.env` settings (e.g. API keys) - so secrets
never live in the config file:

```json
{
  "mcpServers": {
    "keenable": {
      "url": "https://api.keenable.ai/mcp",
      "headers": {"X-API-Key": "${KEENABLE_API_KEY}"}
    },
    "korean-law": {
      "url": "https://mcp.gomdori.app/law"
    },
    "sqlite": {
      "command": ".venv/Scripts/python.exe",
      "args": ["tools/sqlite_mcp_server.py", "--db-path", "chord.db"]
    }
  }
}
```

These three expose 15 tools, bringing a request to 5 428 prompt tokens
including the built-in skills - about three messages per minute on the
Gemini free tier. Every server you add is charged on every message, so
weigh new ones against your provider's input-token rate limit - see
[Troubleshooting](#troubleshooting).

With the `keenable` server configured (`KEENABLE_API_KEY=...` in `.env`) the
bot gains **live web search** (`keenable_search_web_pages`,
`keenable_fetch_page_content`) and answers questions about recent releases,
prices and docs from fresh, sourced results instead of model memory.

The other bundled servers:

* **korean-law** - 12 tools over 법제처 APIs: statute/precedent search,
  law text, citation verification (환각 방지), 조례 정비 레이더 등. Public
  server is quota-shared; append `?oc=<your key>` from
  <https://open.law.go.kr> for an own credential.
* **sqlite** (`tools/sqlite_mcp_server.py`, in-repo) - `db_tables` /
  `db_query` / `db_execute` against `chord.db`, giving the LLM a small
  persistent memory it can create tables in.
Not enabled by default:

* **playwright** (`@playwright/mcp`) - browser automation: navigate,
  snapshot, click, fill forms, screenshot. 24 tools, ~1 600 prompt tokens
  on every message, and `keenable` already covers reading a page. Worth
  turning on for JS-rendered pages you must actually drive - see
  [docs/MCP.md](docs/MCP.md#playwright-opt-in).
* **rhwp** - HWP/HWPX reading, searching, form filling and PDF/text/SVG
  export. Excellent tools, but **82 of them**: adding it takes a request
  from ~6 400 to ~15 300 prompt tokens, which alone exceeds the Gemini
  free tier's 16 000-per-minute input budget after a single message. Turn
  it on when you actually work with HWP files - see
  [docs/MCP.md](docs/MCP.md#rhwp-opt-in) for the snippet and the binary.

Every tool those servers expose is registered next to the built-in skills
(namespaced as `<server>_<tool>`), and the same tool-calling loop drives them.
Set `MCP_ENABLED=false` to skip MCP entirely. A failing server is skipped at
startup - it never blocks the bot.

**Hot reload**: a background task re-reads `mcp.json` every 30 minutes;
add, remove or edit servers and they are swapped in automatically -
no restart needed.

---

## URL safety checks

`check_url_safety` combines independent verdicts so one source being wrong or
unavailable cannot flip the answer:

1. **lrl.kr v5** - Google Safe Browsing cache (`LRL_API_KEY`; the key must
   have the URL-check service enabled at <https://api.lrl.kr>).
2. **Cloudflare 1.1.1.2 for Families** - key-less DNS blocklist; malicious
   domains resolve to `0.0.0.0`.
3. **Cloudflare Radar URL Scanner** (optional) - a real live scan when
   `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` are set.

Verdict logic: any unsafe source means **UNSAFE**; otherwise **SAFE** when at
least one source actively cleared it; **UNKNOWN** if nothing could decide.
Sources without keys are reported as skipped instead of failing the check.

---

## API quota management

Every metered call is counted in `usage.json` (git-ignored runtime state,
path configurable via `QUOTA_STORE_PATH`) and enforced **before** the request
goes out. Counters reset automatically with the calendar month/day, and when
a budget is spent the skill degrades to its key-less fallback instead of
erroring - e.g. an exhausted WeatherAPI key silently answers via Open-Meteo.

| Bucket | Limit | On exhaustion |
|---|---|---|
| SweetTracker | 100/month + same waybill 10/day | falls back to CJ/Post scraping · repeats served from cache |
| Kakao Map | 300,000/month | OSM Nominatim / OSRM |
| WeatherAPI.com | 100,000/month | Open-Meteo |
| Aviationstack | 100/month | OpenSky radar + adsbdb |
| KMA 기상청 (data.go.kr) | ~1,000/day (service default) | Open-Meteo |
| AirKorea 에어코리아 (data.go.kr) | ~500/day (service default) | Open-Meteo CAMS |
| Open-Meteo (key-less) | ~10,000/day tracked defensively | readable error |
| OpenSky (key-less) | ~100 calls/day (400 anonymous credits) | readable error |

Notes: data.go.kr services share one credential string but each service
meters separately, so KMA and AirKorea keep their own daily buckets.
Nominatim is rate-limited by policy (1 req/s), enforced with a client-side
throttle rather than a counter. Frankfurter, Yahoo Finance, OSRM demo,
adsbdb and DuckDuckGo lite have no published hard limits and are not metered.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/SKILLS.md](docs/SKILLS.md) | how to write a skill (drop-in plugin, injection, quota integration, testing) |
| [docs/MCP.md](docs/MCP.md) | mcp.json format, `${VAR}`/`${PYTHON}` placeholders, bundled servers, hot reload |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | per-OS setup, quality gates, project layout, design rules, cross-platform notes |

---

## Adding a skill (one file)

Create `src/chord/skills/<name>.py` - that is the whole installation:

```python
from typing import ClassVar

from chord.skills.base import Skill


class MySkill(Skill):
    name = "my_tool"
    description = "Tell the model WHEN to use this - that is the quality lever."
    parameters: ClassVar[dict] = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    async def run(self, x: str) -> str:
        return f"you said {x}"
```

The registry auto-discovers every `*Skill` class in this package and
turns it into an OpenAI tool definition; declare `settings` or `llm`
constructor parameters and they are injected automatically. HTTP
helpers live in `skills/_http.py`, geocoding in `skills/_geo.py`,
quota guards in `skills/_quota.py` - full guide:
[docs/SKILLS.md](docs/SKILLS.md).

---

## Architecture

```
Discord events          bot.py            mentions, commands, message splitting
     |                  engine.py         one turn = LLM <-> tools loop (max 6 rounds)
     v                  conversation.py   per-channel history (RAM only)
LLM provider            llm.py            AsyncOpenAI facade (base_url configurable)
     |
Tools                   skills/
                          base.py         Skill -> OpenAI tool definition
                          registry.py     collection + safe execution
                          _http/_geo      shared request/geocoding helpers
                          *.py            one module per skill
External tools          mcp_client.py     mcp.json -> sessions -> Skill adapters
Config                  config.py         pydantic-settings (.env -> typed fields)
```

Design rules worth keeping:

* The engine knows nothing about Discord or MCP - it only sees an LLM facade
  and a registry.
* Every tool call is total: unknown tools, bad arguments or provider failures
  become readable error text for the model instead of exceptions.
* Providers degrade gracefully: official Korean sources first, free key-less
  services as fallback.

---

## Development

```powershell
.venv\Scripts\ruff format src tests    # formatting
.venv\Scripts\ruff check src tests     # linting
.venv\Scripts\pytest                   # tests (no network needed - all mocked)
```

Tests never touch the network: HTTP is mocked with `respx`, the LLM with small
fakes, so the whole suite runs offline in seconds.

---

## License

MIT
