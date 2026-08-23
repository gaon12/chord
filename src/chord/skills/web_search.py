"""Web search skill - DuckDuckGo first, extensible to other engines.

Queries DuckDuckGo's key-less ``lite`` HTML endpoint and parses the
result links (which come back as ``//duckduckgo.com/l/?uddg=<real url>``
redirects) into clean title/url/snippet entries.

The engine lives behind a tiny provider registry (PROVIDERS) so extra
backends can be plugged in by adding one async function and one dict
entry - no changes to the skill class itself.
"""

from __future__ import annotations

import html as html_module
import re
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

from chord.skills._http import SkillHTTPError, get_text
from chord.skills.base import Skill

DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"

#: Number of results returned to the model.
MAX_RESULTS = 5

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
    raw_html = await get_text(DUCKDUCKGO_LITE_URL, params={"q": query})
    return parse_results(raw_html)


#: Engine registry - add a backend by adding an entry here.
PROVIDERS: dict[str, object] = {
    "duckduckgo": search_duckduckgo,
}
DEFAULT_PROVIDER = "duckduckgo"


def format_results(query: str, provider: str, results: list[dict]) -> str:
    lines = [f"Search results for '{query}'  [via {provider}]"]
    for i, item in enumerate(results, start=1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
    return "\n".join(lines)


class WebSearchSkill(Skill):
    name = "web_search"
    description = (
        "Search the web for current information (news, facts, docs, "
        "anything you are unsure about). Returns titles, links and "
        "short snippets."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords.",
            }
        },
        "required": ["query"],
    }

    async def run(self, query: str) -> str:
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
        return format_results(query, "DuckDuckGo", results)
