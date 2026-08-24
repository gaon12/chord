"""Tests for the Wikipedia summary skill (action API, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.wikipedia import WIKI_API_URL, WikiSummarySkill, clean_extract


def _api_response(
    title="서울대학교", extract="서울대학교는 대한민국 최고의 국립 대학교이다.", missing=False
):
    pages = {"1": {"title": title, "extract": extract, "missing": True if missing else None}}
    return {"query": {"pages": pages}}


@respx.mock
async def test_wiki_happy_path_with_link():
    route = respx.get(WIKI_API_URL).respond(json=_api_response())

    result = await WikiSummarySkill().run(topic="서울대학교")

    assert route.called
    assert "서울대학교: 서울대학교는 대한민국" in result
    assert "https://ko.wikipedia.org/wiki/" in result


@respx.mock
async def test_wiki_missing_article_raises():
    respx.get(WIKI_API_URL).respond(json=_api_response(missing=True))

    with pytest.raises(SkillHTTPError, match="No Wikipedia article"):
        await WikiSummarySkill().run(topic="존재하지않는문서")


async def test_empty_topic_fails_fast():
    with pytest.raises(SkillHTTPError, match="provide a topic"):
        await WikiSummarySkill().run(topic="")


def test_clean_extract_trims_at_sentence_boundary():
    text = "First sentence. Second sentence. Third."
    assert clean_extract(text, limit=30) == "First sentence."


def test_clean_extract_squeezes_whitespace():
    assert clean_extract("hello   \n  world") == "hello world"
