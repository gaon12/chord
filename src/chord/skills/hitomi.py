"""Hitomi.la metadata search.

The site is a single-page app with no search API: the browser downloads
binary index files and does the filtering itself. Those files are the
API, once you know the shape.

``{area}/{name}-{language}.nozomi`` is a flat array of big-endian int32
gallery ids, newest first, so the newest N results are the first 4N
bytes - one ranged request rather than a megabyte download. Each id then
resolves through ``galleries/{id}.js``, which is JSON behind a ``var``
assignment.

This returns *metadata only* - titles, artists, tags, and the gallery
link. It does not fetch, mirror or post images; whoever asked follows
the link themselves.

Adult content is gated to channels Discord has marked age-restricted,
and to DMs. That is Discord's own rule about where this may be posted,
not a judgement about the asking: a server that posts it into an
ordinary channel is a server that gets reported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import ClassVar
from urllib.parse import quote

import httpx

from chord.context import channel_allows_age_restricted
from chord.skills._http import TIMEOUT_SECONDS, SkillHTTPError, get_text
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

#: Hitomi moved its static host; the old ltn.hitomi.la no longer resolves.
LTN_BASE = "https://ltn.gold-usergeneratedcontent.net"

#: Where a human goes to actually read the thing.
GALLERY_BASE = "https://hitomi.la"

#: The site refuses its own index files without a matching referer.
HITOMI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": f"{GALLERY_BASE}/",
}

#: Index areas, i.e. what a query can be a name *of*. These are the
#: site's own url segments, so they are not ours to rename.
AREAS = ("tag", "artist", "series", "character", "group", "type")

#: Results per search. Each one costs a metadata request, and a chat
#: window is not a gallery wall.
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

#: Tags listed per result: enough to tell what something is, not the
#: whole tag wall.
MAX_TAGS_SHOWN = 8

#: Languages the site indexes, plus the aliases people actually type.
LANGUAGE_ALIASES = {
    "한국어": "korean",
    "한글": "korean",
    "kr": "korean",
    "ko": "korean",
    "일본어": "japanese",
    "jp": "japanese",
    "ja": "japanese",
    "영어": "english",
    "en": "english",
    "중국어": "chinese",
    "zh": "chinese",
    "전체": "all",
    "any": "all",
}

DEFAULT_LANGUAGE = "korean"


def normalize_language(language: str | None) -> str:
    text = (language or DEFAULT_LANGUAGE).strip().lower()
    return LANGUAGE_ALIASES.get(text, text) or DEFAULT_LANGUAGE


def index_url(area: str, query: str, language: str) -> str:
    """The nozomi index holding ids for one search.

    An empty query means "everything in this language", which the site
    keeps at the root rather than under an area.
    """
    if not query:
        return f"{LTN_BASE}/{quote(f'index-{language}.nozomi', safe='.')}"
    return f"{LTN_BASE}/{quote(f'{area}/{query}-{language}.nozomi', safe='/.')}"


def parse_ids(data: bytes) -> list[int]:
    """Gallery ids from a nozomi block: big-endian int32, newest first."""
    usable = len(data) // 4
    return list(struct.unpack(f">{usable}i", data[: usable * 4])) if usable else []


async def fetch_ids(url: str, limit: int) -> list[int]:
    """Newest ``limit`` ids, as one ranged request.

    The full index for a popular tag is megabytes of ids we would throw
    away; asking for the first 4N bytes is the whole optimisation.
    """
    headers = {**HITOMI_HEADERS, "Range": f"bytes=0-{limit * 4 - 1}"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        raise SkillHTTPError(f"Could not reach hitomi ({exc}).") from exc

    if response.status_code == 404:
        raise SkillHTTPError("Nothing is indexed under that name, in that language.")
    if response.status_code >= 400:
        raise SkillHTTPError(f"hitomi answered HTTP {response.status_code}.")
    return parse_ids(response.content)


async def fetch_gallery(gallery_id: int) -> dict | None:
    """Metadata for one gallery, or None if it will not load.

    One bad id out of five is not worth failing a search over - the site
    unpublishes galleries and leaves the id in the index.
    """
    url = f"{LTN_BASE}/galleries/{gallery_id}.js"
    try:
        raw = await get_text(url, headers=HITOMI_HEADERS)
        # The file is a JS assignment, not JSON: `var galleryinfo = {...}`.
        return json.loads(raw.split("=", 1)[1].strip().rstrip(";"))
    except (SkillHTTPError, IndexError, ValueError) as exc:
        logger.info("Could not load hitomi gallery %s: %s", gallery_id, exc)
        return None


def _names(entries: list[dict] | None, key: str) -> list[str]:
    return [str(entry.get(key)) for entry in (entries or []) if entry.get(key)]


def _tag_names(entries: list[dict] | None) -> list[str]:
    """Tags with the site's gender prefix restored, as people read them."""
    names = []
    for entry in entries or []:
        tag = entry.get("tag")
        if not tag:
            continue
        if entry.get("female") in ("1", 1):
            tag = f"female:{tag}"
        elif entry.get("male") in ("1", 1):
            tag = f"male:{tag}"
        names.append(str(tag))
    return names


def format_gallery(data: dict) -> str:
    """One result as a compact block."""
    gallery_id = data.get("id")
    title = " ".join(str(data.get("title") or "(untitled)").split())
    lines = [f"[{gallery_id}] {title}"]

    detail = []
    artists = _names(data.get("artists"), "artist") + _names(data.get("groups"), "group")
    if artists:
        detail.append("artist: " + ", ".join(artists[:3]))
    parodies = _names(data.get("parodys"), "parody")
    if parodies:
        detail.append("series: " + ", ".join(parodies[:2]))
    if data.get("type"):
        detail.append(f"type: {data['type']}")
    if data.get("language_localname") or data.get("language"):
        detail.append(f"language: {data.get('language_localname') or data['language']}")
    if data.get("date"):
        detail.append(f"date: {str(data['date'])[:10]}")
    if data.get("files"):
        detail.append(f"pages: {len(data['files'])}")
    if detail:
        lines.append("  " + " | ".join(detail))

    tags = _tag_names(data.get("tags"))
    if tags:
        shown = ", ".join(tags[:MAX_TAGS_SHOWN])
        extra = f" (+{len(tags) - MAX_TAGS_SHOWN} more)" if len(tags) > MAX_TAGS_SHOWN else ""
        lines.append(f"  tags: {shown}{extra}")

    url = data.get("galleryurl") or f"/galleries/{gallery_id}.html"
    lines.append(f"  {GALLERY_BASE}{url}")
    return "\n".join(lines)


def _clamp_limit(limit: object) -> int:
    try:
        value = int(limit)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


class HitomiSearchSkill(Skill):
    name = "search_hitomi"
    description = (
        "Search hitomi.la for doujinshi/manga by tag, artist, series, "
        "character, group or type, and return the newest matches as "
        "metadata: title, artist, tags and the gallery link. Adult "
        "content - it only answers in age-restricted channels and DMs. "
        "Returns links, never images."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The name to look under, as hitomi spells it: an "
                    "artist, a series ('touhou project'), a tag, a "
                    "character. Leave empty for the newest uploads."
                ),
            },
            "area": {
                "type": "string",
                "enum": list(AREAS),
                "description": "What the query names. Defaults to 'tag'.",
            },
            "language": {
                "type": "string",
                "description": (
                    "korean (default), japanese, english, chinese, or 'all' for every language."
                ),
            },
            "limit": {
                "type": "integer",
                "description": f"How many results (default {DEFAULT_LIMIT}, max {MAX_LIMIT}).",
            },
        },
        "required": [],
    }

    async def run(
        self,
        query: str = "",
        area: str = "tag",
        language: str = DEFAULT_LANGUAGE,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        if not channel_allows_age_restricted():
            raise SkillHTTPError(
                "This one only works in an age-restricted channel or a DM - "
                "Discord requires adult content to stay there. Ask again in a "
                "channel marked NSFW, or in a DM."
            )

        area = (area or "tag").strip().lower()
        if area not in AREAS:
            raise SkillHTTPError(f"Unknown area '{area}'. Use one of: {', '.join(AREAS)}.")

        name = " ".join((query or "").split()).lower()
        lang = normalize_language(language)
        count = _clamp_limit(limit)

        ids = await fetch_ids(index_url(area, name, lang), count)
        if not ids:
            raise SkillHTTPError("That index exists but came back empty.")

        galleries = [g for g in await asyncio.gather(*(fetch_gallery(i) for i in ids)) if g]
        if not galleries:
            raise SkillHTTPError("Found matching ids, but none of their pages would load.")

        heading = f"hitomi · {area}:{name or 'newest'} · {lang} · {len(galleries)} result(s)"
        return (
            heading
            + "\n\n"
            + "\n\n".join(format_gallery(g) for g in galleries)
            + "\n\nThese are links and metadata only - no images were fetched."
        )
