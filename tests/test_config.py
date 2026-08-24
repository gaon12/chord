"""Tests for chord.config - settings loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chord.config import DEFAULT_OPENAI_BASE_URL, Settings, load_settings


def _write_env(tmp_path, content: str):
    """Helper: write an env file and return its path."""
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def test_settings_require_required_values():
    """Creating Settings with nothing set must fail loudly."""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            _env_file_encoding=None,  # type: ignore[call-arg]
        )


def test_settings_defaults_applied():
    """Optional values fall back to sensible defaults."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="discord-token",
        openai_api_key="api-key",
    )
    assert settings.openai_base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.mcp_enabled is True
    assert settings.openweather_api_key == ""


def test_settings_read_from_env_file(tmp_path):
    """All values can come from a .env file."""
    env_file = _write_env(
        tmp_path,
        "\n".join(
            [
                "DISCORD_TOKEN=token-from-file",
                "OPENAI_API_KEY=key-from-file",
                "OPENAI_BASE_URL=http://localhost:11434/v1",
                "OPENAI_MODEL=llama3",
                "MCP_ENABLED=false",
            ]
        ),
    )
    settings = load_settings(env_file)
    assert settings.discord_token == "token-from-file"
    assert settings.openai_api_key == "key-from-file"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_model == "llama3"
    assert settings.mcp_enabled is False


def test_load_settings_missing_value_prints_hint(tmp_path, monkeypatch):
    """A missing token exits with a friendly hint instead of a traceback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        load_settings()
    message = str(excinfo.value)
    assert ".env.sample" in message
    assert "discord_token" in message


def test_unknown_env_keys_are_ignored(tmp_path):
    """Extra keys in the env file must not crash the app."""
    env_file = _write_env(
        tmp_path,
        "\n".join(
            [
                "DISCORD_TOKEN=t",
                "OPENAI_API_KEY=k",
                "SOMETHING_ELSE=whatever",
            ]
        ),
    )
    settings = load_settings(env_file)
    assert settings.discord_token == "t"


# -- API key validation ---------------------------------------------------------------


def test_empty_api_key_raises_with_provider_hint(tmp_path):
    from chord.config import load_settings

    env_file = _write_env(tmp_path, "DISCORD_TOKEN=t\nOPENAI_API_KEY=\n")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is empty"):
        load_settings(env_file)


def test_whitespace_api_key_raises_with_hint(tmp_path):
    from chord.config import load_settings

    env_file = _write_env(tmp_path, "DISCORD_TOKEN=t\nOPENAI_API_KEY=   \n")
    with pytest.raises(SystemExit, match="Any format works"):
        load_settings(env_file)


def test_non_sk_key_format_passes_validation(tmp_path):
    """Gemini/OpenRouter keys must not be rejected for their format."""
    from chord.config import load_settings

    env_file = _write_env(tmp_path, "DISCORD_TOKEN=t\nOPENAI_API_KEY=AIzaSyTest123abc\n")
    settings = load_settings(env_file)
    assert settings.openai_api_key == "AIzaSyTest123abc"


def test_discord_token_empty_also_validated(tmp_path):
    from chord.config import load_settings

    env_file = _write_env(tmp_path, "DISCORD_TOKEN=\nOPENAI_API_KEY=k\n")
    with pytest.raises(SystemExit, match="DISCORD_TOKEN is empty"):
        load_settings(env_file)


def test_load_settings_without_arg_reads_default_dotenv(tmp_path, monkeypatch):
    """Regression: load_settings() with no args must read .env file.

    Previously, passing _env_file=None to Settings explicitly DISABLED
    .env reading, so values in .env were silently ignored.
    """

    _write_env(
        tmp_path,
        "DISCORD_TOKEN=from-dotenv\nOPENAI_API_KEY=from-dotenv\n",
    )
    # Clear real environment variables so only .env provides values.
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()  # no explicit env_file -> must use .env
    assert settings.discord_token == "from-dotenv"
    assert settings.openai_api_key == "from-dotenv"


# -- Reasoning level ------------------------------------------------------------------


def _minimal_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="discord-token",
        openai_api_key="api-key",
        **overrides,
    )


def test_reasoning_defaults_to_none_and_asks_for_minimal_effort():
    """Chat replies want speed, so the bot asks for as little thinking as possible."""
    settings = _minimal_settings()
    assert settings.reasoning_level == "none"
    assert settings.reasoning_effort == "minimal"


@pytest.mark.parametrize(
    ("level", "effort"),
    [
        ("auto", None),
        ("none", "minimal"),
        ("light", "low"),
        ("medium", "medium"),
        ("heavy", "high"),
    ],
)
def test_reasoning_levels_map_to_openai_effort_values(level, effort):
    assert _minimal_settings(reasoning_level=level).reasoning_effort == effort


def test_reasoning_auto_sends_no_parameter():
    """'auto' means: don't touch the provider's own default."""
    assert _minimal_settings(reasoning_level="auto").reasoning_effort is None


def test_reasoning_level_is_case_and_space_insensitive():
    """Hand-edited .env files are forgiving."""
    assert _minimal_settings(reasoning_level="  LIGHT ").reasoning_level == "light"


def test_unknown_reasoning_level_is_rejected():
    with pytest.raises(ValidationError):
        _minimal_settings(reasoning_level="galaxy-brain")


def test_reasoning_level_read_from_env_file(tmp_path):
    env_file = _write_env(
        tmp_path,
        "DISCORD_TOKEN=t\nOPENAI_API_KEY=k\nREASONING_LEVEL=heavy\n",
    )
    assert load_settings(env_file).reasoning_level == "heavy"


# -- LLM timeout ----------------------------------------------------------------------


def test_llm_timeout_has_a_chat_sized_default():
    """The SDK's own default is 10 minutes - far too long for a chat reply."""
    assert _minimal_settings().llm_timeout_seconds == 120.0


def test_llm_timeout_is_configurable():
    assert _minimal_settings(llm_timeout_seconds=30).llm_timeout_seconds == 30.0


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_llm_timeout_is_rejected(bad):
    with pytest.raises(ValidationError):
        _minimal_settings(llm_timeout_seconds=bad)
