"""Tests for chord.fonts - fetching and caching the chart font.

The CDN is mocked throughout: the point of a cache is that it is hit
once, and a test suite that downloads 4.6 MB to prove it would be
missing the point twice over.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from chord.config import Settings
from chord.fonts import (
    CACHED_FONT_NAME,
    MAX_FONT_BYTES,
    NOTO_SANS_KR_URL,
    ensure_font,
    forget_resolved_font,
    is_usable_font,
)


def _pillow_default_font_bytes() -> bytes | None:
    """A real (tiny) font to serve from the fake CDN.

    Borrowed from Pillow's own built-in rather than committed to the
    repo: "is this really a font?" is only worth testing against bytes
    FreeType actually accepts. Pillow hands its default over as an
    in-memory buffer on some builds and as a path on others.
    """
    from pathlib import Path

    from PIL import ImageFont

    try:
        source = ImageFont.load_default(size=12).path
    except Exception:  # pragma: no cover - depends on the Pillow build
        return None
    if hasattr(source, "getvalue"):
        return source.getvalue()
    try:  # pragma: no cover - only on builds that expose a real path
        return Path(source).read_bytes()
    except OSError:
        return None


@pytest.fixture
def font_bytes() -> bytes:
    data = _pillow_default_font_bytes()
    if not data:  # pragma: no cover - depends on the Pillow build
        pytest.skip("Pillow has no default font to borrow")
    return data


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        font_cache_dir=tmp_path / "fonts",
        **overrides,
    )


# -- Validation --------------------------------------------------------------------


def test_a_missing_file_is_not_a_usable_font(tmp_path):
    assert is_usable_font(tmp_path / "nope.otf") is False


def test_an_html_error_page_is_not_a_usable_font(tmp_path):
    """This is the failure the check exists for: a cached 404 body."""
    fake = tmp_path / "NotoSansKR-Regular.otf"
    fake.write_bytes(b"<!DOCTYPE html><title>404 Not Found</title>")

    assert is_usable_font(fake) is False


def test_a_real_font_file_is_usable(tmp_path, font_bytes):
    real = tmp_path / "real.ttf"
    real.write_bytes(font_bytes)

    assert is_usable_font(real) is True


# -- Downloading and caching ---------------------------------------------------------


@respx.mock
async def test_the_font_is_downloaded_and_cached_on_first_use(tmp_path, font_bytes):
    route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)
    settings = _settings(tmp_path)

    path = await ensure_font(settings)

    assert path == str(tmp_path / "fonts" / CACHED_FONT_NAME)
    assert (tmp_path / "fonts" / CACHED_FONT_NAME).read_bytes() == font_bytes
    assert route.call_count == 1


@respx.mock
async def test_a_cached_font_is_reused_without_touching_the_network(tmp_path, font_bytes):
    cached = tmp_path / "fonts" / CACHED_FONT_NAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(font_bytes)
    route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)

    path = await ensure_font(_settings(tmp_path))

    assert path == str(cached)
    assert route.call_count == 0


@respx.mock
async def test_the_answer_is_memoized_for_the_process(tmp_path, font_bytes):
    """One filesystem check per install, not one per chart."""
    route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)
    settings = _settings(tmp_path)

    first = await ensure_font(settings)
    (tmp_path / "fonts" / CACHED_FONT_NAME).unlink()  # cache pulled from under it
    second = await ensure_font(settings)

    assert first == second
    assert route.call_count == 1


@respx.mock
async def test_simultaneous_charts_download_the_font_only_once(tmp_path, font_bytes):
    """Two channels asking at the same moment on a cold cache."""
    route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)
    settings = _settings(tmp_path)

    results = await asyncio.gather(*(ensure_font(settings) for _ in range(4)))

    assert len(set(results)) == 1
    assert route.call_count == 1


# -- Configured font ------------------------------------------------------------------


@respx.mock
async def test_a_configured_font_skips_the_download_entirely(tmp_path, font_bytes):
    own = tmp_path / "my.ttf"
    own.write_bytes(font_bytes)
    route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)

    path = await ensure_font(_settings(tmp_path, chart_font_path=own))

    assert path == str(own)
    assert route.call_count == 0


@respx.mock
async def test_a_broken_configured_font_falls_back_to_noto(tmp_path, font_bytes, caplog):
    broken = tmp_path / "broken.ttf"
    broken.write_bytes(b"not a font")
    respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)

    with caplog.at_level("WARNING"):
        path = await ensure_font(_settings(tmp_path, chart_font_path=broken))

    assert path == str(tmp_path / "fonts" / CACHED_FONT_NAME)
    assert "CHART_FONT_PATH" in caplog.text


# -- Failing safely ---------------------------------------------------------------------


@respx.mock
async def test_a_404_is_not_cached_as_a_font(tmp_path, caplog):
    """Caching an error page would silently kill Korean labels forever."""
    respx.get(NOTO_SANS_KR_URL).respond(status_code=404, text="Not Found")

    with caplog.at_level("WARNING"):
        await ensure_font(_settings(tmp_path))

    assert not (tmp_path / "fonts" / CACHED_FONT_NAME).exists()
    assert "Could not download" in caplog.text


@respx.mock
async def test_a_body_that_is_not_a_font_is_discarded(tmp_path, caplog):
    """A captive portal answering 200 with HTML is the realistic case."""
    respx.get(NOTO_SANS_KR_URL).respond(content=b"<html>login first</html>")

    with caplog.at_level("WARNING"):
        await ensure_font(_settings(tmp_path))

    assert not (tmp_path / "fonts" / CACHED_FONT_NAME).exists()
    assert "not a font" in caplog.text


@respx.mock
async def test_an_oversized_download_is_abandoned(tmp_path, caplog):
    respx.get(NOTO_SANS_KR_URL).respond(content=b"x" * (MAX_FONT_BYTES + 1))

    with caplog.at_level("WARNING"):
        await ensure_font(_settings(tmp_path))

    assert not (tmp_path / "fonts" / CACHED_FONT_NAME).exists()
    assert "limit" in caplog.text


@respx.mock
async def test_no_partial_file_is_left_behind_after_a_failure(tmp_path):
    respx.get(NOTO_SANS_KR_URL).respond(content=b"nope")

    await ensure_font(_settings(tmp_path))

    assert list((tmp_path / "fonts").glob("*.part")) == []


@respx.mock
async def test_an_offline_host_falls_through_instead_of_raising(tmp_path):
    """A bot with no internet still answers - with ASCII labels if it must."""
    respx.get(NOTO_SANS_KR_URL).mock(side_effect=httpx.ConnectError("no route to host"))

    path = await ensure_font(_settings(tmp_path))

    # Either a system font was found, or nothing was - never an exception.
    assert path is None or is_usable_font(path)


async def test_forgetting_the_answer_makes_the_next_call_resolve_again(tmp_path, font_bytes):
    with respx.mock:
        route = respx.get(NOTO_SANS_KR_URL).respond(content=font_bytes)
        settings = _settings(tmp_path)

        await ensure_font(settings)
        forget_resolved_font()
        (tmp_path / "fonts" / CACHED_FONT_NAME).unlink()
        await ensure_font(settings)

        assert route.call_count == 2
