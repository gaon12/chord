"""Hitomi.la metadata search.

The site is a single-page app with no search API: the browser downloads
binary index files and does the filtering itself. Those files are the
API, once you know the shape.

``{area}/{name}-{language}.nozomi`` is a flat array of big-endian int32
gallery ids, newest first, so the newest N results are the first 4N
bytes - one ranged request rather than a megabyte download. Each id then
resolves through ``galleries/{id}.js``, which is JSON behind a ``var``
assignment.

Free-text search is a second structure: a B-tree the browser walks with
ranged requests, keyed by the first four bytes of ``sha256(term)``. Each
hit points at an offset in a companion data file holding the matching
gallery ids. Multi-word queries intersect the sets, which is what the
site itself does - there is no server to ask.

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
import hashlib
import json
import logging
import struct
import time
from collections.abc import Sequence
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

#: Free-text search across titles and tags: the B-tree, not an index file.
AREA_SEARCH = "search"

#: A gallery number, straight to its metadata.
AREA_ID = "id"

#: Index areas, i.e. what a query can be an exact name *of*. These are
#: the site's own url segments, so they are not ours to rename.
INDEX_AREAS = ("tag", "artist", "series", "character", "group", "type")

AREAS = (AREA_SEARCH, AREA_ID, *INDEX_AREAS)

#: Where the search B-tree and its data file live.
SEARCH_INDEX_DIR = "galleriesindex"

#: Every node in the B-tree is padded to this, so a node is one ranged
#: request without having to know its real length first.
NODE_SIZE = 464

#: Branching factor: a node holds up to this many keys and one more
#: subnode address than that.
BRANCHING = 16

#: Give up on a search term whose posting list is absurd - a corrupt
#: length should not pull a gigabyte through a chat bot.
MAX_POSTING_BYTES = 8 * 1024 * 1024

#: How long a downloaded language index stays usable. New uploads
#: appear constantly, but an hour-old list of a hundred thousand ids is
#: wrong only about the newest few.
LANGUAGE_INDEX_TTL = 3600.0

#: Language indexes kept in memory at once. Korean is 400 kB on the
#: wire but a few megabytes as a set of ints, and a channel talks about
#: one or two languages, not nine.
LANGUAGE_INDEX_CACHE_SIZE = 2

#: {language: (monotonic time fetched, ids)}
_language_index_cache: dict[str, tuple[float, frozenset[int]]] = {}

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


def hash_term(term: str) -> bytes:
    """The B-tree key for a search word: first 4 bytes of its SHA-256."""
    return hashlib.sha256(term.encode("utf-8")).digest()[:4]


def split_terms(query: str) -> list[str]:
    """Search words, normalized the way the site normalizes them."""
    return [word for word in query.lower().replace("_", " ").split() if word]


def decode_node(data: bytes) -> tuple[list[bytes], list[tuple[int, int]], list[int]]:
    """One B-tree node: ``(keys, postings, subnode addresses)``.

    Layout, all big-endian: a key count then that many length-prefixed
    keys; a data count then that many (uint64 offset, int32 length)
    pairs; then exactly ``BRANCHING + 1`` uint64 subnode addresses,
    zero where there is no child.
    """
    position = 0
    (key_count,) = struct.unpack_from(">i", data, position)
    position += 4
    keys: list[bytes] = []
    for _ in range(key_count):
        (size,) = struct.unpack_from(">i", data, position)
        position += 4
        if not 0 < size <= 32:
            raise ValueError(f"implausible key size {size}")
        keys.append(data[position : position + size])
        position += size

    (data_count,) = struct.unpack_from(">i", data, position)
    position += 4
    postings: list[tuple[int, int]] = []
    for _ in range(data_count):
        (offset,) = struct.unpack_from(">Q", data, position)
        position += 8
        (length,) = struct.unpack_from(">i", data, position)
        position += 4
        postings.append((offset, length))

    subnodes: list[int] = []
    for _ in range(BRANCHING + 1):
        (address,) = struct.unpack_from(">Q", data, position)
        position += 8
        subnodes.append(address)
    return keys, postings, subnodes


def parse_ids(data: bytes) -> list[int]:
    """Gallery ids from a nozomi block: big-endian int32, newest first."""
    usable = len(data) // 4
    return list(struct.unpack(f">{usable}i", data[: usable * 4])) if usable else []


async def fetch_ids(url: str, limit: int) -> list[int]:
    """Newest ``limit`` ids, as one ranged request.

    The full index for a popular tag is megabytes of ids we would throw
    away; asking for the first 4N bytes is the whole optimisation.
    """
    try:
        async with hitomi_client() as client:
            response = await client.get(url, headers={"Range": f"bytes=0-{limit * 4 - 1}"})
    except httpx.RequestError as exc:
        raise SkillHTTPError(f"Could not reach hitomi ({exc}).") from exc

    if response.status_code == 404:
        raise SkillHTTPError("Nothing is indexed under that name, in that language.")
    if response.status_code >= 400:
        raise SkillHTTPError(f"hitomi answered HTTP {response.status_code}.")
    return parse_ids(response.content)


def hitomi_client() -> httpx.AsyncClient:
    """A client preloaded with the headers the site insists on."""
    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=HITOMI_HEADERS)


async def _get_range(client: httpx.AsyncClient, url: str, start: int, end: int) -> bytes:
    """One ranged GET, as the site's own browser client does it."""
    try:
        response = await client.get(url, headers={"Range": f"bytes={start}-{end}"})
    except httpx.RequestError as exc:
        raise SkillHTTPError(f"Could not reach hitomi ({exc}).") from exc
    if response.status_code >= 400:
        raise SkillHTTPError(f"hitomi answered HTTP {response.status_code}.")
    return response.content


async def search_index_version(client: httpx.AsyncClient) -> str:
    """Current index version; the file names carry it.

    Not cached: a stale version 404s every request that uses it, and one
    small GET is cheaper than being wrong for as long as a cache lasts.
    """
    url = f"{LTN_BASE}/{SEARCH_INDEX_DIR}/version?_={int(time.time() * 1000)}"
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SkillHTTPError(f"Could not read hitomi's search index version ({exc}).") from exc
    text = response.text.strip()
    if not text.isdigit():
        raise SkillHTTPError("hitomi did not report a usable search index version.")
    return text


async def locate_key(
    client: httpx.AsyncClient,
    index_url: str,
    key: bytes,
    address: int = 0,
    depth: int = 0,
) -> tuple | None:
    """Walk the B-tree for ``key``; ``(offset, length)`` or None.

    Every step depends on the node before it, so this is a chain of
    round trips that cannot be parallelised - which is exactly why the
    client is passed in rather than opened per hop.

    Depth is capped because the addresses come off the wire: a corrupt
    node pointing at itself must not become an infinite descent.
    """
    if depth > 16:
        return None
    keys, postings, subnodes = decode_node(
        await _get_range(client, index_url, address, address + NODE_SIZE - 1)
    )

    index = 0
    while index < len(keys) and key > keys[index]:
        index += 1
    if index < len(keys) and key == keys[index]:
        return postings[index] if index < len(postings) else None
    if index >= len(subnodes) or not subnodes[index]:
        return None
    return await locate_key(client, index_url, key, subnodes[index], depth + 1)


async def posting_ids(
    client: httpx.AsyncClient, data_url: str, offset: int, length: int
) -> list[int]:
    """Gallery ids a search term points at, newest first."""
    if not 0 < length <= MAX_POSTING_BYTES:
        raise SkillHTTPError("hitomi returned an implausible result set for that term.")
    blob = await _get_range(client, data_url, offset, offset + length - 1)
    (count,) = struct.unpack_from(">i", blob, 0)
    return list(struct.unpack_from(f">{count}i", blob, 4))


async def language_gallery_ids(language: str) -> frozenset[int]:
    """Every gallery id published in one language.

    This is how a free-text search gets scoped to a language: the search
    B-tree knows nothing about languages, so the two sets are
    intersected on this side. The alternative - resolve the newest few
    dozen hits and keep whichever happen to match - answers "no results"
    for any language that is not the bulk of the site, which for Korean
    is almost always wrong.

    The Korean index is 400 kB. Worth fetching once an hour to get an
    exact answer.
    """
    cached = _language_index_cache.get(language)
    now = time.monotonic()
    if cached and now - cached[0] < LANGUAGE_INDEX_TTL:
        return cached[1]

    url = f"{LTN_BASE}/index-{quote(language, safe='')}.nozomi"
    try:
        async with hitomi_client() as client:
            response = await client.get(url)
    except httpx.RequestError as exc:
        raise SkillHTTPError(f"Could not reach hitomi ({exc}).") from exc
    if response.status_code == 404:
        raise SkillHTTPError(f"hitomi does not index a language called '{language}'.")
    if response.status_code >= 400:
        raise SkillHTTPError(f"hitomi answered HTTP {response.status_code}.")

    ids = frozenset(parse_ids(response.content))
    while len(_language_index_cache) >= LANGUAGE_INDEX_CACHE_SIZE:
        oldest = min(_language_index_cache, key=lambda key: _language_index_cache[key][0])
        del _language_index_cache[oldest]
    _language_index_cache[language] = (now, ids)
    logger.info("Cached the %s gallery index (%d ids).", language, len(ids))
    return ids


async def search_ids(query: str) -> list[int]:
    """Gallery ids matching every word of ``query``, newest first.

    Multi-word queries intersect, because that is what the site does -
    the index maps one word to one posting list and nothing joins them
    server-side.
    """
    terms = split_terms(query)
    if not terms:
        return []

    async with hitomi_client() as client:
        version = await search_index_version(client)
        index_url = f"{LTN_BASE}/{SEARCH_INDEX_DIR}/galleries.{version}.index"
        data_url = f"{LTN_BASE}/{SEARCH_INDEX_DIR}/galleries.{version}.data"

        postings = await asyncio.gather(
            *(locate_key(client, index_url, hash_term(term)) for term in terms)
        )
        for term, posting in zip(terms, postings, strict=True):
            if posting is None:
                raise SkillHTTPError(f"No results: '{term}' does not appear in the search index.")

        id_lists = await asyncio.gather(
            *(posting_ids(client, data_url, offset, length) for offset, length in postings)
        )
    ordered, *rest = id_lists
    if not rest:
        return ordered
    common = set(ordered).intersection(*(set(other) for other in rest))
    return [gallery_id for gallery_id in ordered if gallery_id in common]


def _parse_gallery_js(raw: str) -> dict:
    """`var galleryinfo = {...}` is JSON wearing a JavaScript hat."""
    return json.loads(raw.split("=", 1)[1].strip().rstrip(";"))


async def fetch_gallery(gallery_id: int) -> dict | None:
    """Metadata for one gallery, or None if it will not load.

    One bad id out of five is not worth failing a search over - the site
    unpublishes galleries and leaves the id in its indexes.
    """
    try:
        raw = await get_text(
            f"{LTN_BASE}/galleries/{gallery_id}.js",
            headers=HITOMI_HEADERS,
        )
        return _parse_gallery_js(raw)
    except (SkillHTTPError, IndexError, ValueError) as exc:
        logger.info("Could not load hitomi gallery %s: %s", gallery_id, exc)
        return None


async def fetch_galleries(gallery_ids: Sequence[int]) -> list[dict]:
    """Metadata for several galleries, over one connection.

    Resolving a language-filtered search can mean forty of these, and
    the shared helper in _http opens a fresh client - so a fresh TLS
    handshake - per call. One client for the batch turned a 22-second
    search into a couple of seconds; nothing else about it changed.
    """
    if not gallery_ids:
        return []

    async def one(client: httpx.AsyncClient, gallery_id: int) -> dict | None:
        try:
            response = await client.get(f"{LTN_BASE}/galleries/{gallery_id}.js")
            response.raise_for_status()
            return _parse_gallery_js(response.text)
        except (httpx.HTTPError, IndexError, ValueError) as exc:
            logger.info("Could not load hitomi gallery %s: %s", gallery_id, exc)
            return None

    async with hitomi_client() as client:
        found = await asyncio.gather(*(one(client, i) for i in gallery_ids))
    return [gallery for gallery in found if gallery]


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
        "Search hitomi.la for doujinshi/manga and return metadata: "
        "title, artist, tags and the gallery link. Free-text by default "
        "(words from the title or tags), or an exact tag/artist/series/"
        "character/group/type index, or a gallery number. Adult content "
        "- it only answers in age-restricted channels and DMs. Returns "
        "links, never images."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look for. With area='search' (the default) "
                    "these are words from the title or tags, and every "
                    "word must match. With an index area it is the "
                    "exact name hitomi uses ('touhou project'). With "
                    "area='id' it is the gallery number. Leave empty "
                    "for the newest uploads."
                ),
            },
            "area": {
                "type": "string",
                "enum": list(AREAS),
                "description": (
                    "'search' (default) matches words anywhere; 'id' "
                    "takes a gallery number; the rest are exact index "
                    "lookups by that field."
                ),
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
        area: str = AREA_SEARCH,
        language: str = DEFAULT_LANGUAGE,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        if not channel_allows_age_restricted():
            raise SkillHTTPError(
                "This one only works in an age-restricted channel or a DM - "
                "Discord requires adult content to stay there. Ask again in a "
                "channel marked NSFW, or in a DM."
            )

        area = (area or AREA_SEARCH).strip().lower()
        if area not in AREAS:
            raise SkillHTTPError(f"Unknown area '{area}'. Use one of: {', '.join(AREAS)}.")

        name = " ".join((query or "").split()).lower()
        lang = normalize_language(language)
        count = _clamp_limit(limit)

        if area == AREA_ID:
            galleries = [await self._one_gallery(name)]
            heading = f"hitomi · id:{name}"
        elif area == AREA_SEARCH and name:
            galleries = await self._by_keywords(name, lang, count)
            heading = f"hitomi · search:{name} · {lang} · {len(galleries)} result(s)"
        else:
            # An empty search means "the newest", which the language
            # index answers directly and far more cheaply.
            lookup_area = "tag" if area == AREA_SEARCH else area
            galleries = await self._by_index(lookup_area, name, lang, count)
            heading = (
                f"hitomi · {lookup_area}:{name or 'newest'} · {lang} · {len(galleries)} result(s)"
            )

        return (
            heading
            + "\n\n"
            + "\n\n".join(format_gallery(g) for g in galleries)
            + "\n\nThese are links and metadata only - no images were fetched."
        )

    async def _one_gallery(self, raw_id: str) -> dict:
        if not raw_id.isdigit():
            raise SkillHTTPError(f"'{raw_id}' is not a gallery number.")
        gallery = await fetch_gallery(int(raw_id))
        if gallery is None:
            raise SkillHTTPError(f"No gallery {raw_id} - it may have been unpublished.")
        return gallery

    async def _by_index(self, area: str, name: str, lang: str, count: int) -> list[dict]:
        ids = await fetch_ids(index_url(area, name, lang), count)
        if not ids:
            raise SkillHTTPError("That index exists but came back empty.")
        galleries = await fetch_galleries(ids)
        if not galleries:
            raise SkillHTTPError("Found matching ids, but none of their pages would load.")
        return galleries

    async def _by_keywords(self, query: str, lang: str, count: int) -> list[dict]:
        """Free-text hits, narrowed to a language by set intersection."""
        ids = await search_ids(query)
        if not ids:
            raise SkillHTTPError(f"Nothing matched '{query}'.")

        if lang != "all":
            in_language = await language_gallery_ids(lang)
            matched = [gallery_id for gallery_id in ids if gallery_id in in_language]
            if not matched:
                raise SkillHTTPError(
                    f"'{query}' has {len(ids)} match(es), but none of them in "
                    f"{lang}. Try language='all'."
                )
            ids = matched

        galleries = await fetch_galleries(ids[:count])
        if not galleries:
            raise SkillHTTPError("Found matching ids, but none of their pages would load.")
        return galleries
