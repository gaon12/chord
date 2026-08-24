"""Wikipedia summary skill - Korean Wikipedia action API (key-less).

Uses the MediaWiki ``action=query`` endpoint (more bot-friendly than
the REST summary route) with a descriptive User-Agent per Wikimedia's
robot policy:

    GET https://ko.wikipedia.org/w/api.php
        ?action=query&prop=extracts&exintro&explaintext
        &format=json&redirects=1&titles=<topic>
"""

from __future__ import annotations

import re
from typing import ClassVar

from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

WIKI_API_URL = "https://ko.wikipedia.org/w/api.php"
WIKI_PAGE_URL = "https://ko.wikipedia.org/wiki/{title}"

#: Keep answers chat-sized.
MAX_EXTRACT_CHARS = 600

# A descriptive UA is required by the Wikimedia robot policy.
WIKI_HEADERS = {
    "User-Agent": (
        "chord-discord-bot/0.1 (https://github.com/example/chord; chord@example.com) httpx"
    )
}


def clean_extract(text: str, limit: int = MAX_EXTRACT_CHARS) -> str:
    """Trim an extract to whole sentences within ``limit`` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(".", 0, limit)
    return text[: cut + 1] if cut > 0 else text[:limit]


class WikiSummarySkill(Skill):
    name = "get_wiki_summary"
    description = (
        "Look up a topic on Korean Wikipedia and return a short encyclopedia-style summary."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Article title or subject, e.g. '세종대왕'.",
            }
        },
        "required": ["topic"],
    }

    async def run(self, topic: str) -> str:
        topic = topic.strip()
        if not topic:
            raise SkillHTTPError("Please provide a topic to look up.")

        data = await get_json(
            WIKI_API_URL,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "redirects": 1,
                "titles": topic,
            },
            headers=WIKI_HEADERS,
        )
        pages = ((data.get("query") or {}).get("pages")) or {}
        if not pages:
            raise SkillHTTPError(f"Wikipedia returned no page for '{topic}'.")

        page = next(iter(pages.values()))
        title = page.get("title", topic)
        extract = clean_extract(str(page.get("extract", "")))
        if page.get("missing") or not extract:
            raise SkillHTTPError(f"No Wikipedia article found for '{topic}'.")

        link = WIKI_PAGE_URL.format(title=title.replace(" ", "_"))
        return f"{title}: {extract}\n{link}"
