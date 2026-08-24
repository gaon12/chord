"""News headline skill - Yonhap RSS with Google News fallback (key-less).

* Primary: Yonhap News section feeds ``https://www.yna.co.kr/rss/{section}.xml``
  (verified live: politics / economy / society / culture / sports, 120
  items each).
* Fallback & search: Google News RSS
  ``https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko`` - used when a
  keyword is given or when the section feed fails/returns too little.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import ClassVar

from chord.skills._http import SkillHTTPError, get_text
from chord.skills.base import Skill

YONHAP_RSS_URL = "https://www.yna.co.kr/rss/{section}.xml"
GOOGLE_NEWS_URL = "https://news.google.com/rss"
GOOGLE_NEWS_SEARCH_URL = "https://news.google.com/rss/search"

#: Sections verified against the live feed.
YONHAP_SECTIONS = ["politics", "economy", "society", "culture", "sports"]

#: Minimum items before we consider topping up from Google News.
MIN_YONHAP_ITEMS = 5

MAX_ITEMS = 8


def parse_rss_items(raw_xml: str, limit: int = MAX_ITEMS) -> list[dict[str, str]]:
    """Extract title/link/pubDate triples from an RSS document."""
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return []

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        items.append({"title": _clean_title(title), "link": link, "date": pub_date})
        if len(items) >= limit:
            break
    return items


def _clean_title(title: str) -> str:
    """Strip trailing attribution and HTML entities commonly in feeds."""
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"\s+-\s+[^-]{2,20}$", "", title)


class NewsSkill(Skill):
    name = "get_news"
    description = (
        "Get the latest Korean news headlines. Sections: politics "
        "(default), economy, society, culture, sports. With a query: "
        "searches Google News for that keyword."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": [*YONHAP_SECTIONS],
                "description": "News section. Default 'politics'.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional keyword - searches Google News instead of listing a section."
                ),
            },
        },
        "required": [],
    }

    async def run(self, section: str = "politics", query: str = "") -> str:
        if query.strip():
            return await self._search_google(query.strip())

        name = section.strip().lower() or "politics"
        if name not in YONHAP_SECTIONS:
            raise SkillHTTPError(
                f"Unknown section '{section}'. Use one of: {', '.join(YONHAP_SECTIONS)} "
                "- or provide a query to search."
            )

        items = await self._fetch_yonhap(name)
        source = "연합뉴스"
        if len(items) < MIN_YONHAP_ITEMS:
            extra = await self._fetch_google(f"{name} 뉴스", limit=MAX_ITEMS)
            seen = {item["title"] for item in items}
            fresh = [item for item in extra if item["title"] not in seen]
            if not items and fresh:
                # Yonhap contributed nothing at all - Google only.
                source, items = "Google News", fresh[:MAX_ITEMS]
            elif fresh:
                items += fresh[: MAX_ITEMS - len(items)]
                source = "연합뉴스 + Google News"

        if not items:
            raise SkillHTTPError(f"No news found for section '{name}'.")

        lines = [f"Latest {name} news  [via {source}]"]
        for index, item in enumerate(items[:MAX_ITEMS], start=1):
            lines.append(f"{index}. {item['title']}")
            if item.get("link"):
                lines.append(f"   {item['link']}")
        return "\n".join(lines)

    async def _fetch_yonhap(self, section: str) -> list[dict[str, str]]:
        try:
            raw_xml = await get_text(YONHAP_RSS_URL.format(section=section))
        except SkillHTTPError:
            # Section feed down -> Google News fallback takes over.
            return []
        return parse_rss_items(raw_xml)

    async def _search_google(self, query: str) -> str:
        items = await self._fetch_google(query)
        if not items:
            raise SkillHTTPError(f"No news results for '{query}'.")
        lines = [f"News for '{query}'  [via Google News]"]
        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item['title']}")
            if item.get("link"):
                lines.append(f"   {item['link']}")
        return "\n".join(lines)

    async def _fetch_google(self, query: str, limit: int = MAX_ITEMS) -> list[dict[str, str]]:
        import urllib.parse

        if query:
            url = (
                GOOGLE_NEWS_SEARCH_URL
                + "?"
                + urllib.parse.urlencode({"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
            )
        else:
            url = (
                GOOGLE_NEWS_URL
                + "?"
                + urllib.parse.urlencode({"hl": "ko", "gl": "KR", "ceid": "KR:ko"})
            )
        raw_xml = await get_text(url)
        return parse_rss_items(raw_xml, limit)
