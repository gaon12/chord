"""Tiny async-HTTP helpers shared by all data skills.

Keeping request plumbing here means each skill reads like its API:
geocode, fetch, format. Every helper raises on transport errors and
returns parsed Python objects, so skills only handle business logic.
"""

from __future__ import annotations

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
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(url, params=params, headers=merged_headers)
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s from %s", exc.response.status_code, url)
        raise SkillHTTPError(f"{url} answered HTTP {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        logger.warning("Network error calling %s: %s", url, exc)
        raise SkillHTTPError(f"Could not reach {url}.") from exc
