"""URL shortener skills for the lrl.kr API (requires LRL_API_KEY).

The service offers two operations this skill exposes as separate tools:

* ``shorten_url``      - POST /v6/short  {"url": ...} -> {"result": short}
* ``expand_short_url`` - GET  /v6/short/{hash} -> {"url": ..., "hits": n}

Authentication is a UUID API key sent in the ``x-api-key`` header;
errors arrive as HTTP status plus {"message": "ERR_..."} which we pass
through as readable text.
"""

from __future__ import annotations

import re
from typing import ClassVar

import httpx

from chord.config import Settings
from chord.skills._http import DEFAULT_HEADERS, TIMEOUT_SECONDS, SkillHTTPError
from chord.skills.base import Skill

API_BASE = "https://api.lrl.kr/v6"

#: Matches an already-shortened link and captures its hash.
SHORT_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?lrl\.kr/([A-Za-z0-9]+)/?", re.I)

#: Sanity check before sending a URL upstream.
URL_RE = re.compile(r"^https?://\S+$", re.I)


def extract_hash(short_url_or_hash: str) -> str:
    """Accept either a full lrl.kr link or a bare hash."""
    text = short_url_or_hash.strip()
    match = SHORT_URL_RE.search(text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        return text
    raise SkillHTTPError(f"Could not find a short-URL hash in '{short_url_or_hash}'.")


class UrlShortenerBase(Skill):
    """Shared plumbing: settings access and authenticated requests."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _require_key(self) -> str:
        key = self._settings.lrl_api_key
        if not key:
            raise SkillHTTPError(
                "This tool needs an lrl.kr API key. Set LRL_API_KEY in .env "
                "(issue one at https://api.lrl.kr)."
            )
        return key

    async def _request(self, method: str, path: str, json_body: dict | None = None) -> dict:
        headers = {**DEFAULT_HEADERS, "x-api-key": self._require_key()}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.request(
                    method,
                    f"{API_BASE}{path}",
                    json=json_body,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            message = ""
            try:
                message = exc.response.json().get("message", "")
            except ValueError:
                pass
            detail = f" ({message})" if message else ""
            raise SkillHTTPError(
                f"lrl.kr answered HTTP {exc.response.status_code}{detail}."
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise SkillHTTPError("Could not reach lrl.kr.") from exc


class ShortenUrlSkill(UrlShortenerBase):
    name = "shorten_url"
    description = "Create a short link (lrl.kr) for a long URL using the lrl.kr shortener API."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The long URL to shorten, including https:// .",
            }
        },
        "required": ["url"],
    }

    async def run(self, url: str) -> str:
        url = url.strip()
        if not URL_RE.match(url):
            raise SkillHTTPError(f"'{url}' does not look like a valid http(s) URL.")
        data = await self._request("POST", "/short", json_body={"url": url})
        short = data.get("result")
        if not short:
            raise SkillHTTPError("lrl.kr did not return a shortened URL.")
        return f"Shortened: {short}"


class ExpandUrlSkill(UrlShortenerBase):
    name = "expand_short_url"
    description = (
        "Resolve an lrl.kr short link back to its original URL and show "
        "its click count. Accepts the full link or just the hash."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "short_url": {
                "type": "string",
                "description": "Short link such as 'https://lrl.kr/bPy' or hash 'bPy'.",
            }
        },
        "required": ["short_url"],
    }

    async def run(self, short_url: str) -> str:
        url_hash = extract_hash(short_url)
        data = await self._request("GET", f"/short/{url_hash}")
        original = data.get("url")
        if not original:
            raise SkillHTTPError(f"No original URL found for '{short_url}'.")
        hits = data.get("hits")
        hits_part = f", {int(hits):,} clicks" if hits is not None else ""
        return f"{short_url.strip()} -> {original}{hits_part}"
