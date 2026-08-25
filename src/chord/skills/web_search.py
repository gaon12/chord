"""Web search skill - DuckDuckGo first, extensible to other engines.

Queries DuckDuckGo's key-less ``lite`` HTML endpoint and parses the
result links (which come back as ``//duckduckgo.com/l/?uddg=<real url>``
redirects) into clean title/url/snippet entries.

A snippet is a preview, not a source: two lines of context DuckDuckGo
chose for a human deciding what to click. Answering from one means
answering from an advertisement for the answer. So the skill can also
open the top results and read them - ``read_pages`` - which is the
difference between "I found a page about it" and knowing what it says.

The engine lives behind a tiny provider registry (PROVIDERS) so extra
backends can be plugged in by adding one async function and one dict
entry - no changes to the skill class itself.
"""

from __future__ import annotations

import asyncio
import html as html_module
import logging
import re
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

from chord.skills._fetch import fetch_page
from chord.skills._http import SkillHTTPError, get_text
from chord.skills._readable import extract_readable
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"

#: DuckDuckGo answers an honest bot User-Agent with a 202 challenge page
#: - "select all squares containing a duck" - and zero results. Every
#: other skill here identifies itself truthfully, and should; this
#: endpoint is a scraped HTML page rather than an API, and it serves
#: browsers only.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

#: Words from that challenge page, so being blocked can be reported as
#: being blocked instead of as "nothing matched your search".
_CHALLENGE_MARKERS = ("bots use duckduckgo", "confirm this search was made by a human")

#: Number of results returned to the model.
MAX_RESULTS = 5

#: Most results the model may ask to have opened. Each one is a fetch
#: and a chunk of text; three is already a slow turn and a long prompt.
MAX_READ_PAGES = 3

#: Characters kept per opened page. Deliberately far below what
#: read_url returns for a single link: this text is multiplied by the
#: number of pages, and all of it lands in the channel history to be
#: re-sent with every later message.
MAX_PAGE_CHARS = 1200

_LINK_RE = re.compile(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result-snippet"[^>]*>(.*?)</td>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Strip tags/entities and squeeze whitespace."""
    text = _TAG_RE.sub("", text)
    text = html_module.unescape(text)
    return " ".join(text.split())


def decode_ddg_redirect(href: str) -> str | None:
    """Extract the real target from a //duckduckgo.com/l/?uddg= link."""
    if "uddg=" not in href:
        return None
    query = parse_qs(urlsplit(href).query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else None


def parse_results(raw_html: str, limit: int = MAX_RESULTS) -> list[dict]:
    """Parse lite-HTML into {title,url,snippet} dicts."""
    titles = []
    for match in _LINK_RE.finditer(raw_html):
        url = decode_ddg_redirect(match.group("href"))
        if not url or not url.startswith("http"):
            continue
        titles.append({"url": url, "title": _clean(match.group("title"))})

    snippets = [_clean(snippet) for snippet in _SNIPPET_RE.findall(raw_html)]

    results = []
    for index, entry in enumerate(titles[:limit]):
        entry["snippet"] = snippets[index][:200] if index < len(snippets) else ""
        results.append(entry)
    return results


async def search_duckduckgo(query: str) -> list[dict]:
    """Default engine: key-less DuckDuckGo lite."""
    raw_html = await get_text(
        DUCKDUCKGO_LITE_URL,
        params={"q": query},
        headers=BROWSER_HEADERS,
    )
    if is_challenge_page(raw_html):
        raise SkillHTTPError(
            "DuckDuckGo is asking for a human to solve a challenge, so "
            "search is unavailable right now. Try again in a few minutes."
        )
    return parse_results(raw_html)


def is_challenge_page(raw_html: str) -> bool:
    """Whether DuckDuckGo served its anti-bot page instead of results."""
    lowered = raw_html.lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


#: Engine registry - add a backend by adding an entry here.
PROVIDERS: dict[str, object] = {
    "duckduckgo": search_duckduckgo,
}
DEFAULT_PROVIDER = "duckduckgo"


def _clamp_read_pages(read_pages: object) -> int:
    """How many results to open, from whatever the model passed."""
    try:
        value = int(read_pages)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(value, MAX_READ_PAGES))


async def read_page_text(url: str) -> str:
    """Readable text of one result, or a short reason it is missing.

    Never raises. One dead link out of three must not cost the other
    two, and "could not be opened" is information the model can use -
    silently dropping the result would leave it wondering.
    """
    try:
        page = await fetch_page(url)
        _title, text = extract_readable(page)
    except Exception as exc:  # noqa: BLE001 - reported inline, per result
        logger.info("Could not read search result %s: %s", url, exc)
        return f"(could not be opened: {exc})"

    collapsed = " ".join(text.split())
    if not collapsed:
        return "(no readable text - probably rendered by JavaScript)"
    if len(collapsed) > MAX_PAGE_CHARS:
        return collapsed[:MAX_PAGE_CHARS] + " ..."
    return collapsed


async def read_results(results: list[dict], count: int) -> None:
    """Open the first ``count`` results, in parallel, in place.

    Concurrently because three sequential fetches of a slow news site
    is most of a chat turn spent waiting.
    """
    targets = results[:count]
    if not targets:
        return
    texts = await asyncio.gather(*(read_page_text(item["url"]) for item in targets))
    for item, text in zip(targets, texts, strict=True):
        item["page"] = text


def format_results(query: str, provider: str, results: list[dict]) -> str:
    lines = [f"Search results for '{query}'  [via {provider}]"]
    for i, item in enumerate(results, start=1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
        if item.get("page"):
            lines.append(f"   [page] {item['page']}")
    return "\n".join(lines)


class WebSearchSkill(Skill):
    name = "web_search"
    description = (
        "Search the web for current information (news, facts, docs, "
        "anything you are unsure about). Returns titles, links and "
        "short snippets. Snippets are previews, not sources: when the "
        "answer needs anything a two-line preview cannot carry - how, "
        "why, details, numbers, quotes - set read_pages to open the top "
        "results and read them, instead of guessing from the snippet."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords.",
            },
            "read_pages": {
                "type": "integer",
                "description": (
                    "How many of the top results to actually open and "
                    f"read (0-{MAX_READ_PAGES}, default 0). Use 1-2 "
                    "whenever the snippets are unlikely to contain the "
                    "answer; it is slower but it is the difference "
                    "between finding a page and knowing what it says."
                ),
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, read_pages: int = 0) -> str:
        query = query.strip()
        if not query:
            raise SkillHTTPError("Please provide a search query.")

        provider_name = DEFAULT_PROVIDER
        provider = PROVIDERS.get(provider_name)
        if provider is None:
            raise SkillHTTPError(f"No search provider '{provider_name}' available.")

        results = await provider(query)
        if not results:
            raise SkillHTTPError(f"No search results for '{query}'.")

        wanted = _clamp_read_pages(read_pages)
        if wanted:
            logger.info("Opening the top %d result(s) for '%s'.", wanted, query)
            await read_results(results, wanted)
        return format_results(query, "DuckDuckGo", results)
