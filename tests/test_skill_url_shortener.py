"""Tests for the URL shortener skills (lrl.kr API, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.url_shortener import (
    ExpandUrlSkill,
    ShortenUrlSkill,
    extract_hash,
)

API_BASE = "https://api.lrl.kr/v6"


def _settings(**keys) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


def _shorten_settings() -> Settings:
    return _settings(lrl_api_key="uuid-key")


# -- Missing key -------------------------------------------------------------------


async def test_missing_key_fails_fast_without_request():
    with respx.mock:
        route = respx.post(f"{API_BASE}/short").respond(json={"result": "x"})
        with pytest.raises(SkillHTTPError, match="LRL_API_KEY"):
            await ShortenUrlSkill(_settings()).run(url="https://example.com")
        assert not route.called


# -- shorten_url ---------------------------------------------------------------------


@respx.mock
async def test_shorten_url_happy_path():
    route = respx.post(f"{API_BASE}/short").respond(json={"result": "https://lrl.kr/bPy"})

    result = await ShortenUrlSkill(_shorten_settings()).run(
        url="https://example.com/very/long/path?q=1"
    )

    assert "https://lrl.kr/bPy" in result
    assert route.called
    # The API key must travel in the header, never in the URL/body.
    request = route.calls[0].request
    assert request.headers.get("x-api-key") == "uuid-key"


async def test_shorten_url_rejects_non_urls_locally():
    with pytest.raises(SkillHTTPError, match="does not look like"):
        await ShortenUrlSkill(_shorten_settings()).run(url="not a url")


@respx.mock
async def test_shorten_url_error_message_passthrough():
    respx.post(f"{API_BASE}/short").respond(status_code=400, json={"message": "ERR_INVALID_URL"})

    with pytest.raises(SkillHTTPError, match="ERR_INVALID_URL"):
        await ShortenUrlSkill(_shorten_settings()).run(url="https://bad.example")


# -- expand_short_url ------------------------------------------------------------------


@respx.mock
async def test_expand_short_url_with_full_link():
    route = respx.get(f"{API_BASE}/short/bPy").respond(
        json={"url": "https://google.com", "hits": 3469}
    )

    result = await ExpandUrlSkill(_shorten_settings()).run(short_url="https://lrl.kr/bPy")

    assert route.called
    assert "https://google.com" in result
    assert "3,469 clicks" in result


@respx.mock
async def test_expand_short_url_accepts_bare_hash():
    respx.get(f"{API_BASE}/short/bPy").respond(json={"url": "https://google.com", "hits": 1})

    result = await ExpandUrlSkill(_shorten_settings()).run(short_url="bPy")

    assert "https://google.com" in result


@respx.mock
async def test_expand_unknown_hash_reports_error():
    respx.get(f"{API_BASE}/short/zzz").respond(status_code=404, json={"message": "ERR_NO_DATA"})

    with pytest.raises(SkillHTTPError, match="ERR_NO_DATA"):
        await ExpandUrlSkill(_shorten_settings()).run(short_url="https://lrl.kr/zzz")


# -- Helpers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://lrl.kr/bPy", "bPy"),
        ("http://www.lrl.kr/bPy/", "bPy"),
        ("look at lrl.kr/bPy please", "bPy"),
        ("bPy", "bPy"),
    ],
)
def test_extract_hash(text, expected):
    assert extract_hash(text) == expected


def test_extract_hash_rejects_garbage():
    with pytest.raises(SkillHTTPError, match="Could not find"):
        extract_hash("https://example.com/nope")
