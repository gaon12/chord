"""Tests for the book search and its catalogue fallback chain."""

from __future__ import annotations

import pytest
import respx

from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.books import (
    GOOGLE_BOOKS_URL,
    MAX_LIMIT,
    NL_BASE,
    NL_SEARCH_URL,
    OPENLIBRARY_SEARCH_URL,
    Book,
    BookSearchSkill,
    _clamp_limit,
    normalize_isbn,
)


def _skill(**overrides) -> BookSearchSkill:
    return BookSearchSkill(
        Settings(_env_file=None, discord_token="t", openai_api_key="k", **overrides)
    )


def _nl_body(count: int = 1) -> dict:
    return {
        "total": count,
        "result": [
            {
                "titleInfo": "클린 코드 <br/>",
                "authorInfo": "로버트 C. 마틴 지음",
                "pubInfo": "인사이트",
                "pubYearInfo": "2013",
                "isbn": "9788966260959",
                "detailLink": "/NL/detail/1",
            }
        ]
        * count,
    }


def _google_body() -> dict:
    return {
        "items": [
            {
                "volumeInfo": {
                    "title": "Clean Code",
                    "authors": ["Robert C. Martin"],
                    "publisher": "Prentice Hall",
                    "publishedDate": "2008",
                    "industryIdentifiers": [
                        {"type": "ISBN_10", "identifier": "0132350882"},
                        {"type": "ISBN_13", "identifier": "9780132350884"},
                    ],
                    "infoLink": "https://books.google.com/x",
                }
            }
        ]
    }


def _openlibrary_body() -> dict:
    return {
        "docs": [
            {
                "title": "Clean Code",
                "author_name": ["Robert C. Martin"],
                "publisher": ["Prentice Hall"],
                "first_publish_year": 2008,
                "isbn": ["9780136083221"],
                "key": "/works/OL17618370W",
            }
        ]
    }


# -- ISBN detection -------------------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    ["9788966260959", "978-89-6626-095-9", "978 89 6626 095 9", "0132350882", "043942089X"],
)
def test_an_isbn_is_recognized_however_it_is_typed(given):
    assert normalize_isbn(given) is not None


@pytest.mark.parametrize("given", ["clean code", "12345", "9788966260959123", ""])
def test_other_queries_are_not_isbns(given):
    assert normalize_isbn(given) is None


def test_the_x_check_digit_is_kept_uppercase():
    assert normalize_isbn("043942089x") == "043942089X"


def test_the_result_count_is_clamped():
    assert _clamp_limit(99) == MAX_LIMIT
    assert _clamp_limit(0) == 1
    assert _clamp_limit("2") == 2


# -- Rendering ---------------------------------------------------------------------------


def test_a_result_reads_as_one_scannable_block():
    text = Book(
        title="Clean Code",
        authors="Robert C. Martin",
        publisher="Prentice Hall",
        year="2008",
        isbn="9780132350884",
        url="https://example.com/x",
    ).render(1)

    assert text.splitlines()[0] == "1. Clean Code"
    assert "Robert C. Martin · Prentice Hall · 2008 · ISBN 9780132350884" in text
    assert "https://example.com/x" in text


def test_a_result_with_only_a_title_still_renders():
    assert Book(title="Untitled").render(2) == "2. Untitled"


# -- The catalogue chain --------------------------------------------------------------------


@respx.mock
async def test_the_korean_catalogue_answers_first_when_it_has_a_key():
    """A Korean channel asks about Korean books; NL knows those editions."""
    nl = respx.get(NL_SEARCH_URL).respond(json=_nl_body())
    google = respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    result = await _skill(nl_api_key="nl-key").run(query="클린 코드")

    assert "[via 국립중앙도서관]" in result
    assert "클린 코드" in result
    assert "인사이트 · 2013" in result
    assert nl.call_count == 1
    assert google.call_count == 0


@respx.mock
async def test_markup_in_a_national_library_title_is_stripped():
    respx.get(NL_SEARCH_URL).respond(json=_nl_body())

    result = await _skill(nl_api_key="nl-key").run(query="클린 코드")

    assert "<br/>" not in result


@respx.mock
async def test_without_a_key_the_korean_catalogue_is_skipped():
    nl = respx.get(NL_SEARCH_URL).respond(json=_nl_body())
    respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    result = await _skill().run(query="clean code")

    assert "[via Google Books]" in result
    assert nl.call_count == 0


@respx.mock
async def test_an_error_code_from_the_national_library_falls_through():
    """It answers 200 with an error body rather than an HTTP status."""
    respx.get(NL_SEARCH_URL).respond(json={"errorCode": "010", "errorMsg": "인증키값이 없습니다"})
    respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    assert "[via Google Books]" in await _skill(nl_api_key="bad").run(query="clean code")


@respx.mock
async def test_an_exhausted_google_quota_falls_through_to_open_library():
    """The anonymous quota is per IP and shared, so this is the normal case."""
    respx.get(GOOGLE_BOOKS_URL).respond(status_code=429, json={})
    respx.get(OPENLIBRARY_SEARCH_URL).respond(json=_openlibrary_body())

    result = await _skill().run(query="clean code")

    assert "[via Open Library]" in result
    assert "Robert C. Martin" in result


@respx.mock
async def test_an_empty_catalogue_is_not_an_answer():
    respx.get(GOOGLE_BOOKS_URL).respond(json={"items": []})
    respx.get(OPENLIBRARY_SEARCH_URL).respond(json=_openlibrary_body())

    assert "[via Open Library]" in await _skill().run(query="clean code")


@respx.mock
async def test_when_every_catalogue_fails_the_reasons_are_reported():
    respx.get(GOOGLE_BOOKS_URL).respond(status_code=429, json={})
    respx.get(OPENLIBRARY_SEARCH_URL).respond(json={"docs": []})

    with pytest.raises(SkillHTTPError, match="No catalogue had"):
        await _skill().run(query="a book nobody wrote")


async def test_an_empty_query_fails_before_any_request():
    with pytest.raises(SkillHTTPError, match="What book"):
        await _skill().run(query="   ")


# -- What each catalogue is asked ------------------------------------------------------------


@respx.mock
async def test_an_isbn_is_looked_up_as_an_isbn_not_as_words():
    route = respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    await _skill().run(query="978-0-13-235088-4")

    assert route.calls.last.request.url.params["q"] == "isbn:9780132350884"


@respx.mock
async def test_open_library_gets_the_isbn_in_its_own_field():
    respx.get(GOOGLE_BOOKS_URL).respond(status_code=429, json={})
    route = respx.get(OPENLIBRARY_SEARCH_URL).respond(json=_openlibrary_body())

    await _skill().run(query="9780132350884")

    params = route.calls.last.request.url.params
    assert params["isbn"] == "9780132350884"
    assert "q" not in params


@respx.mock
async def test_a_google_key_is_sent_when_there_is_one():
    route = respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    await _skill(google_books_api_key="g-key").run(query="clean code")

    assert route.calls.last.request.url.params["key"] == "g-key"


@respx.mock
async def test_no_key_means_no_key_parameter():
    route = respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    await _skill().run(query="clean code")

    assert "key" not in route.calls.last.request.url.params


@respx.mock
async def test_the_isbn_13_is_preferred_over_the_isbn_10():
    respx.get(GOOGLE_BOOKS_URL).respond(json=_google_body())

    result = await _skill().run(query="clean code")

    assert "ISBN 9780132350884" in result


@respx.mock
async def test_a_relative_national_library_link_is_made_clickable():
    """It returns site-relative paths, which are a dead string in Discord."""
    respx.get(NL_SEARCH_URL).respond(json=_nl_body())

    result = await _skill(nl_api_key="nl-key").run(query="클린 코드")

    assert f"{NL_BASE}/NL/detail/1" in result


@respx.mock
async def test_an_absolute_link_is_left_alone():
    body = _nl_body()
    body["result"][0]["detailLink"] = "https://www.nl.go.kr/NL/already/absolute"
    respx.get(NL_SEARCH_URL).respond(json=body)

    result = await _skill(nl_api_key="nl-key").run(query="클린 코드")

    assert "https://www.nl.go.kr/NL/already/absolute" in result
    assert "www.nl.go.kr/https" not in result


@respx.mock
async def test_the_national_library_is_asked_for_books_only():
    """Unfiltered, a book query comes back led by journal articles."""
    route = respx.get(NL_SEARCH_URL).respond(json=_nl_body())

    await _skill(nl_api_key="nl-key").run(query="클린 코드")

    assert route.calls.last.request.url.params["category"] == "도서"
