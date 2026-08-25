"""Application settings loaded from environment variables / a `.env` file.

Every value the bot needs lives here in one place. Values are read in this
order (later wins):

1. Defaults declared on each field below.
2. The `.env` file in the working directory (see `.env.sample`).
3. Real environment variables (highest priority, standard 12-factor style).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default base URL points at the real OpenAI API, but any OpenAI-compatible
#: server works (OpenRouter, Ollama, vLLM, ...) by overriding OPENAI_BASE_URL.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

#: How hard the model should think before answering. Chat replies want
#: speed far more than deliberation, so the scale starts at "none".
ReasoningLevel = Literal["auto", "none", "light", "medium", "heavy"]

#: Level -> OpenAI ``reasoning_effort`` value. ``None`` means "send no
#: parameter at all and let the provider pick its own default".
REASONING_EFFORT_BY_LEVEL: dict[str, str | None] = {
    "auto": None,
    "none": "minimal",
    "light": "low",
    "medium": "medium",
    "heavy": "high",
}

#: The same levels as an ordered tuple, cheapest thinking first. Used to
#: build the /reasoning choices so code and command never drift apart.
REASONING_LEVELS: tuple[str, ...] = tuple(REASONING_EFFORT_BY_LEVEL)

#: How much provider-side content filtering to ask for. "off" only means
#: anything where the provider actually exposes a threshold (today: the
#: Gemini API); it never disables the model's own training.
SafetyFilters = Literal["default", "off"]

#: Standard logging levels, accepted in any casing.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed container for all configuration values."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown keys in .env are ignored, not errors
    )

    # -- Required -----------------------------------------------------------

    #: Discord bot token from the developer portal (Bot -> Reset Token).
    discord_token: str

    #: API key for the OpenAI-compatible provider.
    openai_api_key: str

    # -- LLM ----------------------------------------------------------------

    #: Root URL of an OpenAI-compatible chat API.
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL

    #: Model name, interpreted by whichever provider the base URL points to.
    openai_model: str = "gpt-4o-mini"

    #: How much the model may deliberate before replying. Chat wants
    #: fast answers, so the default asks for as little thinking as the
    #: provider allows; "auto" sends nothing and keeps its default.
    #: Providers that reject the parameter are detected at runtime and
    #: fall back transparently (see chord.llm.LLMService).
    reasoning_level: ReasoningLevel = "none"

    #: Provider-side content filtering. Set LLM_SAFETY_FILTERS=off when
    #: the provider's own filter - not the model - is refusing things it
    #: shouldn't. Only the Gemini API exposes such a threshold; see
    #: chord.llm.build_extra_body for what it does elsewhere (nothing).
    llm_safety_filters: SafetyFilters = "default"

    #: Raw JSON object merged into every chat request's ``extra_body``,
    #: for provider extras this project has no setting of its own. The
    #: escape hatch that keeps provider-specific knobs out of the code.
    llm_extra_body: str = ""

    #: Seconds to wait for one LLM response before giving up. The SDK
    #: default (10 minutes) is an eternity in a chat window: a stalled
    #: provider just looks like a bot that ignores you.
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    #: Estimated prompt tokens of stored channel history above which the
    #: oldest turns are folded into a summary (see chord.compaction).
    #: Set well below the model's real context window: it is the *next*
    #: turn - history plus tool catalog plus the answer - that has to
    #: fit. 0 disables compaction and leaves only the message-count cap.
    history_token_budget: int = Field(default=6000, ge=0)

    #: Hard cap on messages remembered per channel. The backstop for
    #: histories compaction cannot help with (one enormous turn), and
    #: what bounds memory when compaction is off.
    history_max_messages: int = Field(default=40, gt=0)

    #: System prompt prepended to every conversation.
    system_prompt: str = (
        "You are chord, a friendly and concise Discord bot. "
        "Answer in the language the user writes in. "
        "Use the provided tools whenever they help you give a better answer, "
        "and keep replies short enough to be readable in a chat window."
    )

    # -- Diagnostics ---------------------------------------------------------

    #: Root logging level. DEBUG additionally logs why each message was
    #: answered (or, for MCP, what each server exposed), which is the
    #: fastest way to find out why the bot stayed quiet.
    log_level: LogLevel = "INFO"

    # -- MCP (external tool servers) -----------------------------------------

    #: JSON file describing MCP servers to connect to. See mcp.json.sample.
    mcp_config_path: Path = Path("mcp.json")

    #: Set MCP_ENABLED=false to skip MCP entirely.
    mcp_enabled: bool = True

    # -- Optional third-party API keys ---------------------------------------
    # Skills fall back to key-less free sources; these keys unlock the
    # preferred Korean/official providers.

    #: KMA (기상청 단기예보) key for official Korean weather.
    kma_api_key: str = ""

    #: WeatherAPI.com key for worldwide weather.
    weatherapi_api_key: str = ""

    #: OpenWeather key, used by the weather skill when present.
    openweather_api_key: str = ""

    #: AirKorea (에어코리아) key for Korean station-level air quality.
    airkorea_api_key: str = ""

    #: Aviationstack key for flight lookup (real-time flights).
    aviationstack_api_key: str = ""

    #: SweetTracker (스마트택배) key for aggregated parcel tracking.
    sweettracker_api_key: str = ""

    #: lrl.kr URL shortener API key (UUID, sent as x-api-key).
    lrl_api_key: str = ""

    #: Kakao REST API key for Korean maps/places/navigation.
    kakao_rest_api_key: str = ""

    #: Keenable live-web-search MCP API key (used via mcp.json header).
    keenable_api_key: str = ""

    #: Cloudflare API token for Radar URL Scanner (optional URL checks).
    cloudflare_api_key: str = ""

    #: Cloudflare account ID that owns the Radar scanner token.
    cloudflare_account_id: str = ""

    #: JSON file where provider usage counters are persisted.
    quota_store_path: Path = Path("usage.json")

    #: Markdown file defining the bot's character (system prompt body).
    persona_path: Path = Path("persona.md")

    #: Font used for chart labels. Leave empty to search the usual
    #: system locations; set it when charts come out with Korean text
    #: missing, which means no Hangul-capable font was found.
    chart_font_path: Path | None = None

    #: SQLite database for reminders (also exposed via the sqlite MCP).
    reminder_db_path: Path = Path("chord.db")

    @field_validator("reasoning_level", mode="before")
    @classmethod
    def _normalize_reasoning_level(cls, value: object) -> object:
        """Accept 'NONE', ' Light ' and friends from hand-edited .env files."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("llm_safety_filters", mode="before")
    @classmethod
    def _normalize_safety_filters(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("llm_extra_body")
    @classmethod
    def _validate_extra_body(cls, value: str) -> str:
        """Reject malformed JSON at startup, not on the first message."""
        if not value.strip():
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError('LLM_EXTRA_BODY must be a JSON object, e.g. {"key": 1}')
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Accept 'debug' as readily as 'DEBUG'."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @property
    def extra_body(self) -> dict:
        """``LLM_EXTRA_BODY`` parsed; ``{}`` when unset."""
        return json.loads(self.llm_extra_body) if self.llm_extra_body.strip() else {}

    @property
    def reasoning_effort(self) -> str | None:
        """The ``reasoning_effort`` value to send, or None to send none."""
        return REASONING_EFFORT_BY_LEVEL[self.reasoning_level]


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings, raising an error that actually helps fix the problem.

    Args:
        env_file: Optional path to an env file to read instead of `.env`.
            Pass an explicit value to make tests deterministic.

    Returns:
        A validated :class:`Settings` instance.

    Raises:
        SystemExit: With a friendly message when required values are missing.
    """
    try:
        # When env_file is None, DON'T pass _env_file at all so
        # pydantic-settings reads the default .env from model_config.
        # Passing _env_file=None would explicitly DISABLE .env reading.
        kwargs: dict = {}
        if env_file is not None:
            kwargs["_env_file"] = str(env_file)
        settings = Settings(**kwargs)  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = [str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"]
        raise SystemExit(
            "Missing required settings: "
            + (", ".join(missing) if missing else str(exc))
            + "\nCopy .env.sample to .env and fill in the values."
        ) from exc

    _validate_api_keys(settings)
    return settings


def _validate_api_keys(settings: Settings) -> None:
    """Reject empty/whitespace-only credentials with actionable hints."""
    checks = [
        (
            "DISCORD_TOKEN",
            settings.discord_token,
            "Discord bot token from https://discord.com/developers/applications",
        ),
        (
            "OPENAI_API_KEY",
            settings.openai_api_key,
            "API key from your LLM provider. Any format works:\n"
            "  OpenAI: sk-... | Gemini: AIzaSy... | OpenRouter: sk-or-...\n"
            "Just make sure OPENAI_BASE_URL matches the provider.",
        ),
    ]
    for var_name, value, hint in checks:
        if not value or not value.strip():
            raise SystemExit(
                f"{var_name} is empty or missing.\n{hint}\nSet it in .env (see .env.sample)."
            )
