# Groq setup and model fallback

Groq uses the existing OpenAI-compatible client. No separate Groq SDK is
required. This integration does not change the provider for existing setups.

## Minimal setup

Add these values to `.env`:

```ini
DISCORD_TOKEN=your-discord-bot-token
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key
GROQ_FREE_TIER=true
REASONING_LEVEL=none
GROQ_MAX_COMPLETION_TOKENS=1024
```

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are not needed for
this setup. Explicit Groq mode uses `https://api.groq.com/openai/v1` and never
forwards an unrelated OpenAI key. Existing setups that deliberately set
`OPENAI_BASE_URL` to the official Groq endpoint can keep their Groq key in
`OPENAI_API_KEY` for compatibility.

The account's actual billing plan is configured in Groq, not in chord.
`GROQ_FREE_TIER=true` selects the free-model fallback policy; it does not
change a paid account into a free account or impose a monetary spending cap.

## Model order

Free mode tries these models in this exact order:

1. `qwen/qwen3.8-27b`
2. `openai/gpt-oss-120b`
3. `openai/gpt-oss-20b`

A response with HTTP 429 moves to the next available model immediately.
Headers can also mark a model as exhausted before the next request. Models
in a known cooldown are skipped, and the preferred model is tried again
after its cooldown or exhausted quota resets. Each model is attempted at
most once per completion. There are no hidden SDK retries in Groq mode.

An explicit `model_not_found` or `model_decommissioned` error also allows
fallback, with a five-minute cooldown. Authentication errors, unrelated
invalid requests, timeouts, and server errors propagate without cycling
through models. A malformed tool schema needs a fix, not another model.
When all models are limited, the bot reports a usage limit. It does not
switch providers, enable a paid tier, or retry completed tool actions.

To select one model explicitly:

```ini
GROQ_FREE_TIER=false
GROQ_MODEL=openai/gpt-oss-120b
```

## Reasoning levels

`auto` is an application setting. The literal string `auto` is never sent
as `reasoning_effort`; the parameter is omitted and the model uses its
documented default. It does not estimate task difficulty locally.

| Application level | Qwen 3.8 27B | GPT-OSS 120B / 20B |
|---|---|---|
| `auto` | omitted; model default | omitted; model default |
| `none` | `none`; thinking disabled | `low`; minimum supported effort |
| `light` | `low` | `low` |
| `medium` | `medium` | `medium` |
| `heavy` | `high` | `high` |

GPT-OSS cannot turn reasoning off. The `none` application level therefore
uses its lowest supported effort. `/reasoning` displays the model that last
succeeded and its effective setting. Fallback recalculates the effort for
the selected model. A custom model outside this list uses its own default
rather than receiving guessed reasoning parameters.

Known models use `reasoning_format=hidden`, so reasoning text is not placed
in the visible answer. The output cap includes reasoning tokens. Higher
effort can use the entire cap before producing an answer; increase
`GROQ_MAX_COMPLETION_TOKENS` when needed, leaving room for the prompt and tool
schemas in the token rate limit. Truncated tool calls are never executed.

## Usage headers

`/usage` shows the last observed headers for each configured Groq model:

| Header family | Meaning |
|---|---|
| `x-ratelimit-limit-requests`, `remaining-requests`, `reset-requests` | requests per day (RPD) |
| `x-ratelimit-limit-tokens`, `remaining-tokens`, `reset-tokens` | tokens per minute (TPM) |
| `retry-after` | cooldown after HTTP 429 |

Reset durations such as `2m59.56s` are parsed as durations. Missing or invalid
values remain unknown. Expired observations are shown as unknown, not as a
fresh full quota. RPM and TPD remaining are not provided by these headers.
No request is made just to populate `/usage`.

Limits are organization-wide. Other applications can consume capacity after
chord receives a header, so a remaining count is an observation, not a
reservation. Chat, history compaction, and LLM-backed text tools share one
client and one set of observations. Their Groq requests are serialized to
avoid races inside this process.

If a 429 has neither a useful reset header nor `retry-after`, the model gets
a sixty-second cooldown. The code does not guess whether that hidden quota
was RPM or TPD.

## On-demand tool schemas

`TOOL_CATALOG_MODE=auto` loads tool schemas on demand in Groq free mode.
The first request offers only `chord_select_tools` and the short tool-name
index in the system prompt. A greeting therefore does not carry the full
schema catalog. When a lookup is needed, the model selects exact tool names,
receives their schemas on the next request, and then calls those tools.
Selections last for one turn and are isolated between channels.

This saves schema tokens but adds a selection round-trip for lookups.
`TOOL_CATALOG_MODE=full` restores the full catalog on each request;
`on_demand` enables selection for any provider. `/tools` distinguishes full
catalog size from the schemas actually offered on demand. The short index
and conversation history still use tokens.

Reviewed independent lookups run with at most four concurrent calls. Writes,
reminders, unknown plugins, and MCP tools stay sequential. A state-changing
call is an ordering barrier between lookup groups, and results retain the
model's original call order.

## Request configuration and sources

`LLM_EXTRA_BODY` can supply additional provider options, but cannot override
Groq model selection, messages, tools, streaming, reasoning fields, or output
limits. Those fields have dedicated settings or are controlled by chord.

Provider behavior was checked against the official documentation:

- [Qwen 3.8 27B](https://console.groq.com/docs/model/qwen/qwen3.8-27b)
- [Reasoning](https://console.groq.com/docs/reasoning)
- [Rate limits and response headers](https://console.groq.com/docs/rate-limits)
- [OpenAI compatibility](https://console.groq.com/docs/openai)
- [API error codes](https://console.groq.com/docs/errors)


## Application moderation and provider policies

The default `LLM_SAFETY_FILTERS=off` value is a compatibility preference.
chord does not add a separate moderation call, keyword blacklist, or response
censor. Its shipped and fallback personas contain no topic blacklist or
scripted refusal catchphrase.

Groq's documented Chat API has no general "disable all safety" parameter.
The code therefore sends no invented safety override. A model's own training
and provider-enforced policies may still affect responses. This setting does
not disable authentication, tool authorization, private-network guards, or
Discord age-restricted channel checks.
