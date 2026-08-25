"""Web search skill - DuckDuckGo first, extensible to other engines.

Queries DuckDuckGo's key-less ``lite`` HTML endpoint and parses the
result links (which come back as ``//duckduckgo.com/l/?uddg=<real url>``
redirects) into clean title/url/snippet entries.

A snippet is a preview, not a source: two lines of context DuckDuckGo
chose for a human deciding what to click. Answering from one means
answering from an advertisement for the answer. So the skill can also
open the top results and read them - ``read_pages`` - which is the
difference between "I found a page about it" and knowing what it says.

Engines live behind a small registry (PROVIDERS) and are tried in
order: DuckDuckGo first because it costs nothing, Keenable second
because it costs credits but answers when DuckDuckGo has decided we
look like a robot - which it does, regularly, and there is no polite
way to argue with a CAPTCHA.
"""

from __future__ import annotations

import asyncio
import html as html_module
import logging
import re
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from chord.config import Settings
from chord.skills._fetch import fetch_page
from chord.skills._http import SkillHTTPError, get_text
from chord.skills._readable import extract_readable, fence_untrusted
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

#: Keenable's MCP endpoint. Spoken to as plain JSON-RPC over HTTP
#: rather than through the MCP client: the server answers tools/call
#: without a session, and one POST does not need a protocol handshake,
#: a background task and a connection to keep alive.
KEENABLE_MCP_URL = "https://api.keenable.ai/mcp"

#: Keenable's search tool, and how long to wait for it. It searches a
#: live index rather than a cache, so it is slower than a scrape.
KEENABLE_TOOL = "search_web_pages"
KEENABLE_TIMEOUT = 45.0

#: Keenable returns paragraphs of context per result, not a two-line
#: preview. Kept long enough to be worth the credits and short enough
#: not to bury the answer.
KEENABLE_SNIPPET_CHARS = 600

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
_KEENABLE_FIELD_RE = re.compile(r"^(Title|URL):\s*(.+)$", re.MULTILINE)
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


async def search_duckduckgo(query: str, _settings: Settings | None = None) -> list[dict]:
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


def parse_keenable_results(text: str, limit: int = MAX_RESULTS) -> list[dict]:
    """Parse Keenable's text blocks into the shape DuckDuckGo produces.

    The tool answers with ``Title:`` / ``URL:`` / ``Snippets:`` blocks
    rather than JSON, so the split is on the next ``Title:`` at the
    start of a line.
    """
    results: list[dict] = []
    for block in re.split(r"\n(?=Title:\s)", text.strip()):
        fields = dict(_KEENABLE_FIELD_RE.findall(block))
        url = (fields.get("URL") or "").strip()
        if not url.startswith("http"):
            continue
        snippet = ""
        if "Snippets:" in block:
            snippet = " ".join(block.split("Snippets:", 1)[1].split())
        results.append(
            {
                "title": (fields.get("Title") or url).strip(),
                "url": url,
                "snippet": snippet[:KEENABLE_SNIPPET_CHARS],
            }
        )
        if len(results) >= limit:
            break
    return results


async def search_keenable(query: str, settings: Settings) -> list[dict]:
    """Fallback engine: Keenable's live web index, over its MCP endpoint."""
    api_key = (settings.keenable_api_key or "").strip()
    if not api_key:
        raise SkillHTTPError("no KEENABLE_API_KEY is configured")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": KEENABLE_TOOL, "arguments": {"query": query}},
    }
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        # The endpoint may answer either way; say both are acceptable.
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with httpx.AsyncClient(timeout=KEENABLE_TIMEOUT) as client:
            response = await client.post(KEENABLE_MCP_URL, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPStatusError as exc:
        raise SkillHTTPError(f"Keenable answered HTTP {exc.response.status_code}") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise SkillHTTPError(f"could not reach Keenable ({exc})") from exc

    if "error" in body:
        raise SkillHTTPError(f"Keenable refused the search ({body['error'].get('message')})")

    text = "\n".join(
        part.get("text", "")
        for part in (body.get("result") or {}).get("content", [])
        if part.get("type") == "text"
    )
    return parse_keenable_results(text)


#: Engine registry - add a backend by adding an entry here.
PROVIDERS: dict[str, object] = {
    "duckduckgo": search_duckduckgo,
    "keenable": search_keenable,
}

#: Tried in this order. DuckDuckGo is free and usually enough; Keenable
#: costs credits and is what answers when DuckDuckGo will not.
PROVIDER_ORDER: tuple[str, ...] = ("duckduckgo", "keenable")

#: How each engine is named to the model, so an answer can say where it
#: came from.
PROVIDER_LABELS = {"duckduckgo": "DuckDuckGo", "keenable": "Keenable"}


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
    if any(item.get("page") for item in results):
        # The snippets come from the engine; the page text comes from
        # whoever wrote the page, and that difference matters.
        return fence_untrusted("\n".join(lines), "web search results")
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
                "description": (
                    "What to search for. Describe the page you want in "
                    "a phrase rather than typing bare keywords - the "
                    "fallback engine matches on meaning, and it costs "
                    "the other engine nothing."
                ),
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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, query: str, read_pages: int = 0) -> str:
        query = query.strip()
        if not query:
            raise SkillHTTPError("Please provide a search query.")

        engine, results = await self._search(query)

        wanted = _clamp_read_pages(read_pages)
        if wanted:
            logger.info("Opening the top %d result(s) for '%s'.", wanted, query)
            await read_results(results, wanted)
        return format_results(query, PROVIDER_LABELS.get(engine, engine), results)

    async def _search(self, query: str) -> tuple[str, list[dict]]:
        """First engine that answers, with why the others did not.

        An engine that returns nothing counts as a failure and the next
        one is tried: DuckDuckGo serving an empty page is far more often
        a block than a query nobody has written about.
        """
        problems: list[str] = []
        for name in PROVIDER_ORDER:
            provider = PROVIDERS.get(name)
            if provider is None:  # pragma: no cover - registry typo
                continue
            try:
                results = await provider(query, self._settings)
            except SkillHTTPError as exc:
                logger.info("Search engine %s failed: %s", name, exc)
                problems.append(f"{name}: {exc}")
                continue
            if results:
                if name != PROVIDER_ORDER[0]:
                    logger.info("Answered '%s' with the %s fallback.", query, name)
                return name, results
            problems.append(f"{name}: no results")

        raise SkillHTTPError(f"Could not search for '{query}'. " + "; ".join(problems) + ".")
