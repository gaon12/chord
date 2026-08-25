"""Tests for the web-search skill (DuckDuckGo lite, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.web_search import (
    BROWSER_HEADERS,
    DUCKDUCKGO_LITE_URL,
    MAX_PAGE_CHARS,
    MAX_READ_PAGES,
    WebSearchSkill,
    _clamp_read_pages,
    decode_ddg_redirect,
    format_results,
    is_challenge_page,
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


# -- Getting past the front door -------------------------------------------------------


CHALLENGE_HTML = """
<html><body><h1>DuckDuckGo</h1>
<p>Unfortunately, bots use DuckDuckGo too. Please complete the following
challenge to confirm this search was made by a human.</p>
<p>Select all squares containing a duck:</p>
</body></html>
"""


@respx.mock
async def test_the_search_request_does_not_identify_itself_as_a_bot():
    """An honest User-Agent gets a 202 challenge and zero results."""
    route = respx.get(DUCKDUCKGO_LITE_URL).respond(text=DDG_HTML)

    await WebSearchSkill().run(query="anything")

    sent = route.calls.last.request.headers["user-agent"]
    assert sent == BROWSER_HEADERS["User-Agent"]
    assert "bot" not in sent.lower()


def test_the_challenge_page_is_recognized():
    assert is_challenge_page(CHALLENGE_HTML) is True
    assert is_challenge_page(DDG_HTML) is False


@respx.mock
async def test_being_blocked_is_reported_as_being_blocked():
    """ "No results" would send the user off rewording a fine query."""
    respx.get(DUCKDUCKGO_LITE_URL).respond(status_code=202, text=CHALLENGE_HTML)

    with pytest.raises(SkillHTTPError, match="challenge"):
        await WebSearchSkill().run(query="python")


# -- Opening the results ----------------------------------------------------------------

#: The first two URLs in DDG_HTML, which is what read_pages opens.
FIRST_RESULT = "https://example.com/guide"
SECOND_RESULT = "https://docs.example.org/api"


def _ddg(route_html: str = DDG_HTML):
    return respx.get(DUCKDUCKGO_LITE_URL).respond(text=route_html)


@respx.mock
async def test_snippets_only_by_default():
    """Opening pages costs time and prompt length; it must be asked for."""
    _ddg()
    page = respx.get(FIRST_RESULT).respond(
        text="<p>본문</p>", headers={"content-type": "text/html"}
    )

    result = await WebSearchSkill().run(query="python")

    assert "[page]" not in result
    assert page.call_count == 0


@respx.mock
async def test_read_pages_opens_the_top_results_and_returns_their_text():
    _ddg()
    respx.get(FIRST_RESULT).respond(
        text="<p>첫 번째 페이지 본문입니다.</p>", headers={"content-type": "text/html"}
    )
    respx.get(SECOND_RESULT).respond(
        text="<p>두 번째 페이지 본문입니다.</p>", headers={"content-type": "text/html"}
    )

    result = await WebSearchSkill().run(query="python", read_pages=2)

    assert "첫 번째 페이지 본문입니다." in result
    assert "두 번째 페이지 본문입니다." in result


@respx.mock
async def test_only_the_requested_number_of_results_is_opened():
    _ddg()
    first = respx.get(FIRST_RESULT).respond(
        text="<p>본문</p>", headers={"content-type": "text/html"}
    )
    second = respx.get(SECOND_RESULT).respond(
        text="<p>본문</p>", headers={"content-type": "text/html"}
    )

    await WebSearchSkill().run(query="python", read_pages=1)

    assert first.call_count == 1
    assert second.call_count == 0


@respx.mock
async def test_one_dead_link_does_not_cost_the_other_results():
    """Three results, one 404 - the other two still carry the answer."""
    _ddg()
    respx.get(FIRST_RESULT).respond(status_code=404, text="gone")
    respx.get(SECOND_RESULT).respond(
        text="<p>살아있는 본문</p>", headers={"content-type": "text/html"}
    )

    result = await WebSearchSkill().run(query="python", read_pages=2)

    assert "could not be opened" in result
    assert "살아있는 본문" in result


@respx.mock
async def test_a_javascript_only_result_says_so_rather_than_looking_empty():
    _ddg()
    respx.get(FIRST_RESULT).respond(
        text="<div id='root'></div>", headers={"content-type": "text/html"}
    )

    result = await WebSearchSkill().run(query="python", read_pages=1)

    assert "no readable text" in result


@respx.mock
async def test_each_opened_page_is_capped():
    """The text is multiplied by the page count and kept in history."""
    _ddg()
    respx.get(FIRST_RESULT).respond(
        text="<p>" + ("가" * 20_000) + "</p>", headers={"content-type": "text/html"}
    )

    result = await WebSearchSkill().run(query="python", read_pages=1)

    page_line = next(line for line in result.splitlines() if "[page]" in line)
    assert len(page_line) < MAX_PAGE_CHARS + 100
    assert page_line.endswith("...")


@respx.mock
async def test_a_result_pointing_into_a_private_network_is_refused_not_fetched():
    """A search result is still a URL chord did not choose."""
    _ddg(DDG_HTML.replace("https%3A%2F%2Fexample.com%2Fguide", "http%3A%2F%2Flocalhost%2Fadmin"))

    result = await WebSearchSkill().run(query="python", read_pages=1)

    assert "not a public address" in result


@pytest.mark.parametrize(
    ("given", "expected"),
    [(0, 0), (1, 1), (99, MAX_READ_PAGES), (-3, 0), ("2", 2), (None, 0), ("많이", 0)],
)
def test_read_pages_is_clamped_to_something_sane(given, expected):
    assert _clamp_read_pages(given) == expected
