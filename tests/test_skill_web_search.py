"""Tests for the web-search skill (DuckDuckGo lite, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.web_search import (
    DUCKDUCKGO_LITE_URL,
    WebSearchSkill,
    decode_ddg_redirect,
    format_results,
    parse_results,
)

DDG_HTML = """
<html><body>
<td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide&amp;rut=abc">Best &amp; Brightest Guide</a></td>
<td class="result-snippet">A <b>great</b> resource about   testing.</td>
<td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.org%2Fapi&amp;rut=def">API Docs</a></td>
<td class="result-snippet">Official documentation.</td>
<td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F%ED%95%9C%EA%B5%AD%EC%96%B4&amp;rut=ghi">Encoded</a></td>
</body></html>
"""


# -- Parsing ------------------------------------------------------------------------


def test_parse_results_extracts_links_and_snippets():
    results = parse_results(DDG_HTML)

    assert len(results) == 3
    assert results[0]["url"] == "https://example.com/guide"
    assert results[0]["title"] == "Best & Brightest Guide"
    assert results[0]["snippet"] == "A great resource about testing."
    assert results[1]["snippet"] == "Official documentation."


def test_parse_results_handles_encoded_urls():
    results = parse_results(DDG_HTML)
    assert "한국어" in results[2]["url"]  # percent-encoding is decoded


def test_decode_redirect_ignores_plain_links():
    assert decode_ddg_redirect("https://example.com/direct") is None
    assert (
        decode_ddg_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.test%2Fa")
        == "https://x.test/a"
    )


def test_format_results_is_clean():
    results = parse_results(DDG_HTML)
    text = format_results("query", "DuckDuckGo", results)
    lines = text.splitlines()
    assert lines[0] == "Search results for 'query'  [via DuckDuckGo]"
    assert lines[1] == "1. Best & Brightest Guide"
    assert lines[2] == "   https://example.com/guide"


# -- Skill-level behavior --------------------------------------------------------------


@respx.mock
async def test_web_search_happy_path():
    respx.get(DUCKDUCKGO_LITE_URL).respond(html=DDG_HTML)

    result = await WebSearchSkill().run(query="best testing guide")

    assert "Search results for 'best testing guide'" in result
    assert "https://example.com/guide" in result
    assert "Best & Brightest Guide" in result


@respx.mock
async def test_web_search_no_results_raises():
    respx.get(DUCKDUCKGO_LITE_URL).respond(html="<html><body></body></html>")

    with pytest.raises(SkillHTTPError, match="No search results"):
        await WebSearchSkill().run(query="nothing to see")


async def test_empty_query_fails_fast():
    with pytest.raises(SkillHTTPError, match="provide a search query"):
        await WebSearchSkill().run(query="   ")
