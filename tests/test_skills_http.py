"""Tests for chord.skills._http - the shared request plumbing."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from chord.skills._http import (
    SkillHTTPError,
    close_shared_client,
    get_json,
    get_text,
    shared_client,
)

URL = "https://api.test/thing"


@respx.mock
async def test_json_and_text_come_back_parsed():
    respx.get(URL).respond(json={"a": 1})
    assert await get_json(URL) == {"a": 1}

    respx.get(URL).respond(text="plain")
    assert await get_text(URL) == "plain"


@respx.mock
async def test_a_bad_status_becomes_a_readable_error():
    respx.get(URL).respond(status_code=503)

    with pytest.raises(SkillHTTPError, match="HTTP 503"):
        await get_json(URL)


@respx.mock
async def test_an_error_message_does_not_leak_the_query_string():
    """API keys ride in params - NL_API_KEY does - and errors get logged."""
    respx.get(URL).respond(status_code=401)

    with pytest.raises(SkillHTTPError) as raised:
        await get_json(URL, params={"key": "super-secret"})

    assert "super-secret" not in str(raised.value)


@respx.mock
async def test_a_network_failure_becomes_a_readable_error():
    respx.get(URL).mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(SkillHTTPError, match="Could not reach"):
        await get_json(URL)


@respx.mock
async def test_invalid_json_is_reported_as_such():
    respx.get(URL).respond(text="not json")

    with pytest.raises(SkillHTTPError, match="invalid JSON"):
        await get_json(URL)


# -- The shared client ------------------------------------------------------------


async def test_requests_reuse_one_client():
    """A client per request is a TLS handshake per request."""
    assert shared_client() is shared_client()


async def test_a_closed_client_is_replaced():
    first = shared_client()
    await close_shared_client()

    assert shared_client() is not first


def test_the_client_does_not_outlive_its_event_loop():
    """A client is bound to the loop that built its connection pool, and
    every test - and every bot run - brings its own loop."""
    seen: list = []

    async def grab():
        seen.append(shared_client())

    asyncio.run(grab())  # one loop
    asyncio.run(grab())  # and a different one

    assert seen[0] is not seen[1]


@respx.mock
async def test_several_requests_run_concurrently_on_the_one_client():
    respx.get(URL).respond(json={"ok": True})

    results = await asyncio.gather(*(get_json(URL) for _ in range(5)))

    assert results == [{"ok": True}] * 5


async def test_closing_twice_is_harmless():
    shared_client()
    await close_shared_client()
    await close_shared_client()
