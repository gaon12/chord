"""Tiny async-HTTP helpers shared by all data skills.

Keeping request plumbing here means each skill reads like its API:
geocode, fetch, format. Every helper raises on transport errors and
returns parsed Python objects, so skills only handle business logic.

Requests go through one shared client. Opening a fresh
``httpx.AsyncClient`` per call - which is what this module used to do -
means a fresh TCP connection and TLS handshake for every request, and
these skills call the same handful of hosts over and over. Measured
against openlibrary.org it is about 2.5x, and a turn that fans out
across several skills pays it several times.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Generous timeout: some public APIs are slow at peak times.
TIMEOUT_SECONDS = 15.0

#: Some APIs reject requests without a browser-ish User-Agent.
DEFAULT_HEADERS = {"User-Agent": "chord-discord-bot/0.1 (+https://github.com/)"}


class SkillHTTPError(RuntimeError):
    """Raised when an upstream API call fails; text goes back to the LLM."""


#: The shared client, and the loop it belongs to. A client is bound to
#: the event loop that created its connection pool, so it cannot be
#: carried across loops - which happens constantly under pytest, where
#: each test gets a fresh one. Remembering the loop makes that a
#: transparent rebuild instead of a confusing failure.
_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def shared_client() -> httpx.AsyncClient:
    """The client every skill request goes through."""
    global _client, _client_loop

    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
        _client_loop = loop
    return _client


async def close_shared_client() -> None:
    """Close the shared client; the next request opens a new one.

    Called on shutdown so the process does not exit with sockets still
    open, and by tests that want a clean slate.
    """
    global _client, _client_loop

    client, _client, _client_loop = _client, None, None
    if client is not None and not client.is_closed:
        await client.aclose()


async def get_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
) -> Any:
    """GET ``url`` and return the parsed JSON body."""
    response = await _get(url, params, headers)
    try:
        return response.json()
    except ValueError as exc:
        raise SkillHTTPError(f"{url} returned invalid JSON.") from exc


async def get_text(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
) -> str:
    """GET ``url`` and return the raw body text."""
    response = await _get(url, params, headers)
    return response.text


async def _get(url, params, headers) -> httpx.Response:
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    try:
        response = await shared_client().get(url, params=params, headers=merged_headers)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s from %s", exc.response.status_code, url)
        raise SkillHTTPError(f"{url} answered HTTP {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        logger.warning("Network error calling %s: %s", url, exc)
        raise SkillHTTPError(f"Could not reach {url}.") from exc
