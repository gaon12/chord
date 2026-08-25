"""Tests for the hitomi metadata search (indexes and galleries mocked)."""

from __future__ import annotations

import json
import struct

import pytest
import respx

from chord.context import reset_current_channel, set_current_channel
from chord.skills._http import SkillHTTPError
from chord.skills.hitomi import (
    BRANCHING,
    DEFAULT_LIMIT,
    GALLERY_BASE,
    LTN_BASE,
    MAX_LIMIT,
    MAX_TAGS_SHOWN,
    NODE_SIZE,
    SEARCH_INDEX_DIR,
    HitomiSearchSkill,
    _clamp_limit,
    _language_index_cache,
    decode_node,
    fetch_gallery,
    fetch_ids,
    format_gallery,
    hash_term,
    index_url,
    normalize_language,
    parse_ids,
    search_ids,
    split_terms,
)


@pytest.fixture(autouse=True)
def _empty_language_cache():
    """The language index is memoized for an hour; tests start clean."""
    _language_index_cache.clear()
    yield
    _language_index_cache.clear()


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


# -- The search B-tree --------------------------------------------------------------------

VERSION = "1787641157"
INDEX_URL = f"{LTN_BASE}/{SEARCH_INDEX_DIR}/galleries.{VERSION}.index"
DATA_URL = f"{LTN_BASE}/{SEARCH_INDEX_DIR}/galleries.{VERSION}.data"


def _node(keys=(), postings=(), subnodes=()) -> bytes:
    """One B-tree node in the site's own layout, padded to NODE_SIZE."""
    out = struct.pack(">i", len(keys))
    for key in keys:
        out += struct.pack(">i", len(key)) + key
    out += struct.pack(">i", len(postings))
    for offset, length in postings:
        out += struct.pack(">Qi", offset, length)
    addresses = list(subnodes) + [0] * (BRANCHING + 1 - len(subnodes))
    for address in addresses:
        out += struct.pack(">Q", address)
    return out.ljust(NODE_SIZE, b"\x00")


def _posting_blob(*ids: int) -> bytes:
    return struct.pack(">i", len(ids)) + struct.pack(f">{len(ids)}i", *ids)


def _mock_version():
    return respx.get(url__startswith=f"{LTN_BASE}/{SEARCH_INDEX_DIR}/version").respond(text=VERSION)


def _mock_posting(offset: int, length: int, blob: bytes):
    return respx.get(DATA_URL, headers={"Range": f"bytes={offset}-{offset + length - 1}"}).respond(
        content=blob
    )


def _mock_node(address: int, node: bytes):
    return respx.get(
        INDEX_URL, headers={"Range": f"bytes={address}-{address + NODE_SIZE - 1}"}
    ).respond(content=node)


def test_terms_are_lowercased_and_underscores_become_spaces():
    assert split_terms("Touhou_Project  REIMU ") == ["touhou", "project", "reimu"]


def test_a_key_is_the_first_four_bytes_of_the_terms_sha256():
    assert len(hash_term("touhou")) == 4
    assert hash_term("touhou") != hash_term("reimu")


def test_a_node_round_trips_through_the_decoder():
    keys, postings, subnodes = decode_node(_node([b"abcd"], [(1234, 56)], [0, 99]))

    assert keys == [b"abcd"]
    assert postings == [(1234, 56)]
    assert subnodes[1] == 99
    assert len(subnodes) == BRANCHING + 1


def test_an_implausible_key_size_is_rejected():
    """The node came off the wire; a 4 GB key is corruption, not a key."""
    with pytest.raises(ValueError, match="implausible key size"):
        decode_node(struct.pack(">ii", 1, 999999) + b"\x00" * 400)


@respx.mock
async def test_a_single_term_search_returns_its_posting_list():
    _mock_version()
    _mock_node(0, _node([hash_term("reimu")], [(0, 12)]))
    respx.get(DATA_URL).respond(content=_posting_blob(300, 200, 100))

    assert await search_ids("reimu") == [300, 200, 100]


@respx.mock
async def test_the_tree_is_descended_when_the_root_does_not_hold_the_key():
    _mock_version()
    # Root's only key sorts after ours, so the walk takes subnode 0.
    _mock_node(0, _node([b"\xff\xff\xff\xff"], [(0, 0)], [NODE_SIZE]))
    _mock_node(NODE_SIZE, _node([hash_term("reimu")], [(0, 12)]))
    respx.get(DATA_URL).respond(content=_posting_blob(7))

    assert await search_ids("reimu") == [7]


@respx.mock
async def test_a_term_that_is_not_indexed_says_which_one():
    _mock_version()
    _mock_node(0, _node([b"\xff\xff\xff\xff"], [(0, 0)]))

    with pytest.raises(SkillHTTPError, match="'nothing' does not appear"):
        await search_ids("nothing")


@respx.mock
async def test_multiple_terms_are_intersected_newest_first():
    """The index maps one word to one list; nothing joins them for us."""
    _mock_version()
    # A B-tree node stores its keys in order and the walk relies on it,
    # so the posting each term gets follows from where its hash sorts.
    first, second = sorted([hash_term("a"), hash_term("b")])
    _mock_node(0, _node([first, second], [(0, 16), (100, 16)]))

    # Matched on the range, not on call order: the two lookups are
    # fired concurrently and arrive in whichever order they arrive.
    _mock_posting(0, 16, _posting_blob(500, 400, 300))
    _mock_posting(100, 16, _posting_blob(400, 300, 200))

    assert await search_ids("a b") == [400, 300]


@respx.mock
async def test_an_absurd_posting_length_is_refused():
    _mock_version()
    _mock_node(0, _node([hash_term("x")], [(0, 999_999_999)]))

    with pytest.raises(SkillHTTPError, match="implausible result set"):
        await search_ids("x")


@respx.mock
async def test_a_bad_index_version_is_reported():
    respx.get(url__startswith=f"{LTN_BASE}/{SEARCH_INDEX_DIR}/version").respond(text="<html>")

    with pytest.raises(SkillHTTPError, match="usable search index version"):
        await search_ids("x")


# -- Search through the skill -----------------------------------------------------------------


def _mock_search(term: str, *ids: int):
    _mock_version()
    _mock_node(0, _node([hash_term(term)], [(0, 4 + 4 * len(ids))]))
    respx.get(DATA_URL).respond(content=_posting_blob(*ids))


@respx.mock
async def test_free_text_is_the_default_area(nsfw_channel):
    _mock_search("reimu", 111)
    respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111, 222))
    _mock_gallery(111)

    result = await HitomiSearchSkill().run(query="reimu")

    assert "search:reimu" in result
    assert "[111]" in result


@respx.mock
async def test_a_language_narrows_by_intersecting_the_language_index(nsfw_channel):
    """Scanning the newest hits would answer "none" for anything but Japanese."""
    _mock_search("reimu", 999, 111)
    respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
    _mock_gallery(111)

    result = await HitomiSearchSkill().run(query="reimu", language="korean")

    assert "[111]" in result
    assert "[999]" not in result


@respx.mock
async def test_matches_in_no_other_language_say_to_widen(nsfw_channel):
    _mock_search("reimu", 999)
    respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))

    with pytest.raises(SkillHTTPError, match="Try language='all'"):
        await HitomiSearchSkill().run(query="reimu", language="korean")


@respx.mock
async def test_language_all_skips_the_intersection_entirely(nsfw_channel):
    _mock_search("reimu", 999)
    index = respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
    _mock_gallery(999)

    await HitomiSearchSkill().run(query="reimu", language="all")

    assert index.call_count == 0


@respx.mock
async def test_the_language_index_is_downloaded_once(nsfw_channel):
    """400 kB per search would be rude; per hour is not."""
    _mock_search("reimu", 111)
    index = respx.get(f"{LTN_BASE}/index-korean.nozomi").respond(content=_nozomi(111))
    _mock_gallery(111)

    await HitomiSearchSkill().run(query="reimu")
    await HitomiSearchSkill().run(query="reimu")

    assert index.call_count == 1


@respx.mock
async def test_an_unindexed_language_is_reported(nsfw_channel):
    _mock_search("reimu", 111)
    respx.get(f"{LTN_BASE}/index-klingon.nozomi").respond(status_code=404)

    with pytest.raises(SkillHTTPError, match="does not index a language"):
        await HitomiSearchSkill().run(query="reimu", language="klingon")


# -- Lookup by gallery number ------------------------------------------------------------------


@respx.mock
async def test_a_gallery_number_goes_straight_to_its_metadata(nsfw_channel):
    _mock_gallery(4139704, _gallery(4139704))

    result = await HitomiSearchSkill().run(query="4139704", area="id")

    assert "hitomi · id:4139704" in result
    assert "[4139704]" in result


async def test_a_non_numeric_id_is_refused(nsfw_channel):
    with pytest.raises(SkillHTTPError, match="not a gallery number"):
        await HitomiSearchSkill().run(query="touhou", area="id")


@respx.mock
async def test_an_unpublished_gallery_says_so(nsfw_channel):
    respx.get(f"{LTN_BASE}/galleries/111.js").respond(status_code=404)

    with pytest.raises(SkillHTTPError, match="may have been unpublished"):
        await HitomiSearchSkill().run(query="111", area="id")
