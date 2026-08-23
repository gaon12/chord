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


def test_load_settings_missing_value_prints_hint():
    """A missing token exits with a friendly hint instead of a traceback."""
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env_file=None)
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
