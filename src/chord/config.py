"""Application settings loaded from environment variables / a `.env` file.

Every value the bot needs lives here in one place. Values are read in this
order (later wins):

1. Defaults declared on each field below.
2. The `.env` file in the working directory (see `.env.sample`).
3. Real environment variables (highest priority, standard 12-factor style).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default base URL points at the real OpenAI API, but any OpenAI-compatible
#: server works (OpenRouter, Ollama, vLLM, ...) by overriding OPENAI_BASE_URL.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


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

    #: System prompt prepended to every conversation.
    system_prompt: str = (
        "You are chord, a friendly and concise Discord bot. "
        "Answer in the language the user writes in. "
        "Use the provided tools whenever they help you give a better answer, "
        "and keep replies short enough to be readable in a chat window."
    )

    # -- MCP (external tool servers) -----------------------------------------

    #: JSON file describing MCP servers to connect to. See mcp.json.sample.
    mcp_config_path: Path = Path("mcp.json")

    #: Set MCP_ENABLED=false to skip MCP entirely.
    mcp_enabled: bool = True

    # -- Optional third-party API keys ---------------------------------------
    # All skills work without keys (free data sources are the default);
    # these keys simply unlock better/different providers.

    #: OpenWeather key, used by the weather skill when present.
    openweather_api_key: str = ""

    #: AirKorea (에어코리아) key for Korean station-level air quality.
    airkorea_api_key: str = ""


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
        if env_file is None:
            return Settings(_env_file=None)  # type: ignore[call-arg]
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = [str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"]
        raise SystemExit(
            "Missing required settings: "
            + (", ".join(missing) if missing else str(exc))
            + "\nCopy .env.sample to .env and fill in the values."
        ) from exc
