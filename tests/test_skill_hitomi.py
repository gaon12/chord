"""Tests for the hitomi metadata search (indexes and galleries mocked)."""

from __future__ import annotations

import json
import struct

import pytest
import respx

from chord.context import reset_current_channel, set_current_channel
from chord.skills._http import SkillHTTPError
from chord.skills.hitomi import (
    DEFAULT_LIMIT,
    GALLERY_BASE,
    LTN_BASE,
    MAX_LIMIT,
    MAX_TAGS_SHOWN,
    HitomiSearchSkill,
    _clamp_limit,
    fetch_gallery,
    fetch_ids,
    format_gallery,
    index_url,
    normalize_language,
    parse_ids,
)


@pytest.fixture
def nsfw_channel():
    """An age-restricted channel, where the skill is allowed to answer."""
    token = set_current_channel(1, nsfw=True)
    yield
    reset_current_channel(token)


@pytest.fixture
def ordinary_channel():
    token = set_current_channel(1, nsfw=False, is_dm=False)
    yield
    reset_current_channel(token)


def _nozomi(*ids: int) -> bytes:
    return struct.pack(f">{len(ids)}i", *ids)


def _gallery(gallery_id: int = 111, **overrides) -> dict:
    data = {
        "id": str(gallery_id),
        "title": "  A   Title  ",
        "type": "doujinshi",
        "language": "korean",
        "language_localname": "한국어",
        "date": "2026-08-22 21:38:00-05",
        "artists": [{"artist": "someone"}],
        "parodys": [{"parody": "touhou project"}],
        "tags": [{"tag": "big breasts", "female": "1"}, {"tag": "glasses"}],
        "files": [{"name": "1.png"}, {"name": "2.png"}],
        "galleryurl": f"/doujinshi/a-title-{gallery_id}.html",
    }
    data.update(overrides)
    return data


def _gallery_js(data: dict) -> str:
    return "var galleryinfo = " + json.dumps(data, ensure_ascii=False)


def _mock_gallery(gallery_id: int, data: dict | None = None, text: str | None = None):
    body = text if text is not None else _gallery_js(data or _gallery(gallery_id))
    return respx.get(f"{LTN_BASE}/galleries/{gallery_id}.js").respond(text=body)


# -- Index addressing ---------------------------------------------------------------


def test_an_empty_query_addresses_the_whole_language_index():
    assert index_url("tag", "", "korean") == f"{LTN_BASE}/index-korean.nozomi"


def test_a_query_addresses_its_area():
    url = index_url("series", "touhou project", "korean")

    assert url == f"{LTN_BASE}/series/touhou%20project-korean.nozomi"


def test_a_gendered_tag_survives_url_encoding():
    """The site's own tag names carry a female:/male: prefix."""
    url = index_url("tag", "female:big breasts", "korean")

    assert url.endswith("/tag/female%3Abig%20breasts-korean.nozomi")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("한국어", "korean"),
        ("KR", "korean"),
        ("일본어", "japanese"),
        ("전체", "all"),
        ("", "korean"),
    ],
)
def test_language_aliases_are_accepted(given, expected):
    assert normalize_language(given) == expected


def test_an_unknown_language_is_passed_through_for_the_site_to_reject():
    assert normalize_language("klingon") == "klingon"


# -- The binary index ----------------------------------------------------------------


def test_ids_are_read_as_big_endian_int32():
    assert parse_ids(_nozomi(4144270, 4144236)) == [4144270, 4144236]


def test_a_partial_trailing_id_is_ignored():
    """A ranged request can land mid-integer; three bytes are not an id."""
    assert parse_ids(_nozomi(1, 2) + b"\x00\x01\x02") == [1, 2]


def test_no_bytes_means_no_ids():
    assert parse_ids(b"") == []


@respx.mock
async def test_only_the_first_ids_are_requested():
    """The full index for a popular tag is megabytes we would discard."""
    url = f"{LTN_BASE}/index-korean.nozomi"
    route = respx.get(url).respond(content=_nozomi(1, 2, 3))

    await fetch_ids(url, 3)

    assert route.calls.last.request.headers["range"] == "bytes=0-11"


@respx.mock
async def test_an_unindexed_name_says_so_plainly():
    url = f"{LTN_BASE}/tag/nope-korean.nozomi"
    respx.get(url).respond(status_code=404)

    with pytest.raises(SkillHTTPError, match="Nothing is indexed"):
        await fetch_ids(url, 5)


@respx.mock
async def test_an_index_error_is_reported_with_its_status():
    url = f"{LTN_BASE}/index-korean.nozomi"
    respx.get(url).respond(status_code=503)

    with pytest.raises(SkillHTTPError, match="HTTP 503"):
        await fetch_ids(url, 5)


# -- Gallery metadata -----------------------------------------------------------------


@respx.mock
async def test_gallery_json_is_unwrapped_from_its_javascript():
    _mock_gallery(111)

    assert (await fetch_gallery(111))["id"] == "111"


@respx.mock
async def test_a_gallery_that_will_not_parse_is_dropped_not_raised():
    """The site leaves unpublished ids in the index; one is not a failure."""
    _mock_gallery(111, text="this is not javascript at all")

    assert await fetch_gallery(111) is None


@respx.mock
async def test_a_missing_gallery_is_dropped():
    respx.get(f"{LTN_BASE}/galleries/111.js").respond(status_code=404)

    assert await fetch_gallery(111) is None


# -- Formatting -------------------------------------------------------------------------


def test_a_result_carries_what_you_need_to_decide():
    text = format_gallery(_gallery(111))

    assert "[111] A Title" in text  # whitespace squeezed
    assert "artist: someone" in text
    assert "series: touhou project" in text
    assert "pages: 2" in text
    assert f"{GALLERY_BASE}/doujinshi/a-title-111.html" in text


def test_gender_prefixes_are_restored_on_tags():
    text = format_gallery(_gallery(111))

    assert "female:big breasts" in text
    assert "glasses" in text


def test_a_tag_wall_is_cut_short():
    tags = [{"tag": f"t{i}"} for i in range(30)]

    text = format_gallery(_gallery(111, tags=tags))

    assert f"(+{30 - MAX_TAGS_SHOWN} more)" in text


def test_a_gallery_with_almost_no_metadata_still_renders():
    assert "(untitled)" in format_gallery({"id": "9"})


@pytest.mark.parametrize(
    ("given", "expected"),
    [(1, 1), (99, MAX_LIMIT), (0, 1), (-4, 1), ("3", 3), (None, DEFAULT_LIMIT)],
)
def test_the_result_count_is_clamped(given, expected):
    assert _clamp_limit(given) == expected


# -- Where it is allowed to answer --------------------------------------------------------


async def test_an_ordinary_channel_is_refused(ordinary_channel):
    """Discord requires adult content to stay in age-restricted channels."""
    with pytest.raises(SkillHTTPError, match="age-restricted channel or a DM"):
        await HitomiSearchSkill().run(query="touhou project", area="series")


async def test_out_of_band_is_refused():
    """No channel bound at all: the safe answer to "where am I" is "not here"."""
    with pytest.raises(SkillHTTPError, match="age-restricted"):
        await HitomiSearchSkill().run(query="x")


@respx.mock
async def test_a_dm_is_allowed():
    token = set_current_channel(1, is_dm=True)
    try:
        respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
        _mock_gallery(111)

        assert "[111]" in await HitomiSearchSkill().run()
    finally:
        reset_current_channel(token)


# -- Searching ------------------------------------------------------------------------------


@respx.mock
async def test_a_search_returns_metadata_and_links(nsfw_channel):
    respx.get(f"{LTN_BASE}/series/touhou%20project-korean.nozomi").respond(
        content=_nozomi(111, 222)
    )
    _mock_gallery(111)
    _mock_gallery(222)

    result = await HitomiSearchSkill().run(query="Touhou Project", area="series", limit=2)

    assert "hitomi · series:touhou project · korean · 2 result(s)" in result
    assert "[111]" in result and "[222]" in result
    assert "no images were fetched" in result


@respx.mock
async def test_one_dead_gallery_does_not_sink_the_search(nsfw_channel):
    respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111, 222))
    _mock_gallery(111)
    respx.get(f"{LTN_BASE}/galleries/222.js").respond(status_code=404)

    result = await HitomiSearchSkill().run(limit=2)

    assert "1 result(s)" in result
    assert "[111]" in result


@respx.mock
async def test_a_search_where_nothing_loads_says_so(nsfw_channel):
    respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
    respx.get(f"{LTN_BASE}/galleries/111.js").respond(status_code=404)

    with pytest.raises(SkillHTTPError, match="none of their pages would load"):
        await HitomiSearchSkill().run()


async def test_an_unknown_area_lists_the_real_ones(nsfw_channel):
    with pytest.raises(SkillHTTPError, match="artist"):
        await HitomiSearchSkill().run(query="x", area="publisher")


@respx.mock
async def test_the_site_gets_the_referer_it_demands(nsfw_channel):
    """Its own index files 403 without one."""
    route = respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
    _mock_gallery(111)

    await HitomiSearchSkill().run()

    assert route.calls.last.request.headers["referer"] == f"{GALLERY_BASE}/"
