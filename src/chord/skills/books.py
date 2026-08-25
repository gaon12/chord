"""Book search across three catalogues, in order of who knows best.

1. 국립중앙도서관 - the National Library of Korea. Far and away the best
   answer for a Korean book, and the only one of the three that reliably
   has the Korean edition's publisher and year. Needs ``NL_API_KEY``.
2. Google Books - broad, multilingual, and key-less until it is not: the
   anonymous daily quota is per IP and shared, so it answers 429 more
   often than it answers. ``GOOGLE_BOOKS_API_KEY`` lifts that.
3. Open Library - key-less, no quota, weakest metadata for anything
   published in Korean. The one that is always there.

Each is tried until one returns results, so a missing key or an
exhausted quota costs a fallback rather than the answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import ClassVar

from chord.config import Settings
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

NL_SEARCH_URL = "https://www.nl.go.kr/NL/search/openApi/search.do"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"

DEFAULT_LIMIT = 5
MAX_LIMIT = 10

#: Blurbs are trimmed hard: this is a chat window, and a paragraph per
#: result buries the five things anyone is actually scanning for.
MAX_SUBTITLE = 100

#: 10 or 13 digits once the hyphens people type are taken out, with the
#: X that can end an ISBN-10.
_ISBN_RE = re.compile(r"^(?:\d{9}[\dXx]|\d{13})$")


def normalize_isbn(query: str) -> str | None:
    """The query as an ISBN, or None if it is not one.

    Worth detecting: every one of these catalogues searches an ISBN far
    better as an ISBN than as thirteen digits of free text.
    """
    candidate = re.sub(r"[\s-]", "", query or "")
    return candidate.upper() if _ISBN_RE.match(candidate) else None


@dataclass(frozen=True)
class Book:
    """One result, in whatever detail the catalogue had."""

    title: str
    authors: str = ""
    publisher: str = ""
    year: str = ""
    isbn: str = ""
    url: str = ""

    def render(self, position: int) -> str:
        lines = [f"{position}. {self.title}"]
        facts = [part for part in (self.authors, self.publisher, self.year) if part]
        if self.isbn:
            facts.append(f"ISBN {self.isbn}")
        if facts:
            lines.append("   " + " · ".join(facts))
        if self.url:
            lines.append(f"   {self.url}")
        return "\n".join(lines)


def _clean(value: object, limit: int = MAX_SUBTITLE) -> str:
    text = " ".join(str(value or "").split())
    # The National Library marks up its own titles with <br/> and the
    # occasional stray tag; nobody wants those read out.
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def _clamp_limit(limit: object) -> int:
    try:
        value = int(limit)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


# -- 국립중앙도서관 ------------------------------------------------------------------


async def search_nl(query: str, limit: int, settings: Settings) -> list[Book]:
    """National Library of Korea. Needs a key; best for Korean books."""
    key = (settings.nl_api_key or "").strip()
    if not key:
        raise SkillHTTPError("no NL_API_KEY is configured")

    data = await get_json(
        NL_SEARCH_URL,
        params={
            "key": key,
            "apiType": "json",
            "srchTarget": "total",
            "kwd": query,
            "pageNum": 1,
            "pageSize": limit,
        },
    )
    if not isinstance(data, dict):
        raise SkillHTTPError("the National Library returned something unreadable")
    if data.get("errorCode"):
        raise SkillHTTPError(f"National Library error {data['errorCode']}: {data.get('errorMsg')}")

    return [
        Book(
            title=_clean(row.get("titleInfo")) or "(제목 없음)",
            authors=_clean(row.get("authorInfo"), 60),
            publisher=_clean(row.get("pubInfo"), 40),
            year=_clean(row.get("pubYearInfo"), 10),
            isbn=_clean(row.get("isbn"), 20),
            url=_clean(row.get("detailLink"), 200),
        )
        for row in (data.get("result") or [])[:limit]
        if isinstance(row, dict)
    ]


# -- Google Books -------------------------------------------------------------------


async def search_google_books(query: str, limit: int, settings: Settings) -> list[Book]:
    """Google Books. Key-less until the shared daily quota runs out."""
    isbn = normalize_isbn(query)
    params: dict = {"q": f"isbn:{isbn}" if isbn else query, "maxResults": limit}
    key = (settings.google_books_api_key or "").strip()
    if key:
        params["key"] = key

    data = await get_json(GOOGLE_BOOKS_URL, params=params)
    if not isinstance(data, dict):
        raise SkillHTTPError("Google Books returned something unreadable")
    if data.get("error"):
        raise SkillHTTPError(f"Google Books: {data['error'].get('message', 'refused')}")

    books = []
    for item in (data.get("items") or [])[:limit]:
        info = item.get("volumeInfo") or {}
        identifiers = [
            entry.get("identifier")
            for entry in info.get("industryIdentifiers") or []
            if entry.get("type", "").startswith("ISBN")
        ]
        books.append(
            Book(
                title=_clean(info.get("title")) or "(untitled)",
                authors=_clean(", ".join(info.get("authors") or []), 60),
                publisher=_clean(info.get("publisher"), 40),
                year=_clean(info.get("publishedDate"), 10),
                isbn=_clean(identifiers[-1] if identifiers else "", 20),
                url=_clean(info.get("infoLink") or info.get("previewLink"), 200),
            )
        )
    return books


# -- Open Library --------------------------------------------------------------------


async def search_openlibrary(query: str, limit: int, _settings: Settings) -> list[Book]:
    """Open Library: no key, no quota, thinnest Korean metadata."""
    isbn = normalize_isbn(query)
    params: dict = {
        "limit": limit,
        "fields": "title,author_name,first_publish_year,isbn,key,publisher",
    }
    if isbn:
        params["isbn"] = isbn
    else:
        params["q"] = query

    data = await get_json(OPENLIBRARY_SEARCH_URL, params=params)
    if not isinstance(data, dict):
        raise SkillHTTPError("Open Library returned something unreadable")

    books = []
    for doc in (data.get("docs") or [])[:limit]:
        key = _clean(doc.get("key"), 60)
        isbns = doc.get("isbn") or []
        books.append(
            Book(
                title=_clean(doc.get("title")) or "(untitled)",
                authors=_clean(", ".join(doc.get("author_name") or []), 60),
                publisher=_clean(", ".join((doc.get("publisher") or [])[:1]), 40),
                year=_clean(doc.get("first_publish_year"), 10),
                isbn=_clean(isbns[0] if isbns else "", 20),
                url=f"https://openlibrary.org{key}" if key else "",
            )
        )
    return books


#: Catalogue registry - add one by adding an async function and an entry.
PROVIDERS: dict[str, object] = {
    "nl": search_nl,
    "google": search_google_books,
    "openlibrary": search_openlibrary,
}

#: Tried in this order: the Korean catalogue first because a Korean
#: channel asks about Korean books, then breadth, then the one that is
#: always up.
PROVIDER_ORDER: tuple[str, ...] = ("nl", "google", "openlibrary")

PROVIDER_LABELS = {
    "nl": "국립중앙도서관",
    "google": "Google Books",
    "openlibrary": "Open Library",
}


class BookSearchSkill(Skill):
    name = "search_books"
    description = (
        "Look up books by title, author, keyword or ISBN and return "
        "what the catalogues know: title, author, publisher, year, ISBN "
        "and a link (책 찾아줘, 이 책 정보, ISBN 조회). Searches the Korean "
        "National Library first, then Google Books, then Open Library."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A title, an author, keywords, or an ISBN-10/13 "
                    "(hyphens are fine - an ISBN is detected and looked "
                    "up as one)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": f"How many results (default {DEFAULT_LIMIT}, max {MAX_LIMIT}).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, query: str, limit: int = DEFAULT_LIMIT) -> str:
        text = " ".join((query or "").split())
        if not text:
            raise SkillHTTPError("What book should I look for?")

        count = _clamp_limit(limit)
        source, books = await self._search(text, count)

        isbn = normalize_isbn(text)
        heading = (
            f"'{text}'{' (ISBN)' if isbn else ''} — {len(books)} result(s) "
            f"[via {PROVIDER_LABELS.get(source, source)}]"
        )
        return heading + "\n" + "\n".join(book.render(i) for i, book in enumerate(books, 1))

    async def _search(self, query: str, limit: int) -> tuple[str, list[Book]]:
        """First catalogue that has something, with why the others did not."""
        problems: list[str] = []
        for name in PROVIDER_ORDER:
            provider = PROVIDERS.get(name)
            if provider is None:  # pragma: no cover - registry typo
                continue
            try:
                books = await provider(query, limit, self._settings)
            except SkillHTTPError as exc:
                logger.info("Book catalogue %s failed: %s", name, exc)
                problems.append(f"{name}: {exc}")
                continue
            if books:
                if name != PROVIDER_ORDER[0]:
                    logger.info("Answered '%s' with the %s catalogue.", query, name)
                return name, books
            problems.append(f"{name}: nothing found")

        raise SkillHTTPError(f"No catalogue had '{query}'. " + "; ".join(problems) + ".")
