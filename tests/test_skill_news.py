"""Tests for the news skill (Yonhap + Google News RSS, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.news import (
    YONHAP_RSS_URL,
    NewsSkill,
    parse_rss_items,
)

YONHAP_URL = YONHAP_RSS_URL.format(section="politics")
GOOGLE_SEARCH_URL = "https://news.google.com/rss/search"

EMPTY_RSS = '<?xml version="1.0"?><rss><channel></channel></rss>'

YONHAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss><channel>
<item><title>국회에서 법안 통과 - 연합뉴스</title><link>https://yna.kr/1</link><pubDate>Mon, 24 Aug 2026 09:00:00 +0900</pubDate></item>
<item><title>정부 발표 - 연합뉴스</title><link>https://yna.kr/2</link><pubDate>Mon, 24 Aug 2026 08:30:00 +0900</pubDate></item>
</channel></rss>"""

GOOGLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss><channel>
<item><title>구글 뉴스 기사 &amp; 더보기 - 어딘가신문사</title><link>https://news.google.com/abc</link><pubDate>Mon, 24 Aug 2026 10:00:00 +0900</pubDate></item>
</channel></rss>"""


# -- Parsing ------------------------------------------------------------------------


def test_parse_rss_extracts_and_cleans_titles():
    items = parse_rss_items(YONHAP_XML)

    assert len(items) == 2
    # Trailing attribution "- 연합뉴스" is stripped.
    assert items[0]["title"] == "국회에서 법안 통과"
    assert items[0]["link"] == "https://yna.kr/1"
    assert "2026" in items[0]["date"]


def test_parse_invalid_xml_returns_empty():
    assert parse_rss_items("<not-xml") == []


# -- Section mode ---------------------------------------------------------------------


@respx.mock
async def test_news_section_uses_yonhap():
    respx.get(YONHAP_URL).respond(text=YONHAP_XML)
    # Top-up source returns nothing so the Yonhap items stand alone.
    respx.get(GOOGLE_SEARCH_URL).respond(text=EMPTY_RSS)

    result = await NewsSkill().run()

    assert "[via 연합뉴스]" in result
    assert "1. 국회에서 법안 통과" in result
    assert "https://yna.kr/1" in result


@respx.mock
async def test_unknown_section_raises():
    with pytest.raises(SkillHTTPError, match="Unknown section"):
        await NewsSkill().run(section="celebrity")


# -- Google fallback / search ------------------------------------------------------------


@respx.mock
async def test_yonhap_failure_falls_back_to_google():
    respx.get(YONHAP_URL).respond(status_code=500)
    # Section feed dead -> the skill searches Google News for the section.
    respx.get(GOOGLE_SEARCH_URL).respond(text=GOOGLE_XML)

    result = await NewsSkill().run(section="politics")

    assert "[via Google News]" in result
    assert "구글 뉴스 기사" in result


@respx.mock
async def test_query_mode_searches_google():
    route = respx.get(GOOGLE_SEARCH_URL).respond(text=GOOGLE_XML)

    result = await NewsSkill().run(query="부동산")

    assert route.called
    assert "News for '부동산'" in result


@respx.mock
async def test_all_sources_empty_raises():
    respx.get(YONHAP_URL).respond(status_code=500)
    respx.get(GOOGLE_SEARCH_URL).respond(text=EMPTY_RSS)

    with pytest.raises(SkillHTTPError, match="No news found"):
        await NewsSkill().run(section="politics")
