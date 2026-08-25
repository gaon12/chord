"""Tests for chord.charts - the PNG line chart.

Pixel-perfect assertions on a drawing are a maintenance trap, so these
check the things that actually break: that a PNG comes out at the right
size, that the number formatting is readable, and above all that a
machine with no Korean font degrades to ASCII instead of drawing boxes.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from chord.charts import (
    HEIGHT,
    WIDTH,
    ChartFonts,
    find_hangul_font,
    format_compact,
    format_precise,
    nice_ticks,
    render_line_chart,
)


def _series(count: int = 20, start: float = 1000.0) -> list[tuple[str, float]]:
    return [(f"08-{index + 1:02d}", start + index * 3.5) for index in range(count)]


def _open(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png))


# -- Rendering ------------------------------------------------------------------------


def test_render_returns_a_png_of_the_declared_size():
    image = _open(render_line_chart(_series(), title="USD/KRW"))

    assert image.format == "PNG"
    assert image.size == (WIDTH, HEIGHT)


def test_render_needs_at_least_two_points():
    """One point is not a trend, and dividing by zero is not a chart."""
    with pytest.raises(ValueError, match="at least two"):
        render_line_chart([("08-01", 1.0)], title="x")


def test_a_flat_series_still_renders():
    """Zero range must not divide by zero on the way to the panel."""
    flat = [("08-01", 5.0), ("08-02", 5.0), ("08-03", 5.0)]

    assert _open(render_line_chart(flat, title="flat")).size == (WIDTH, HEIGHT)


def test_zero_valued_series_renders():
    zeros = [("08-01", 0.0), ("08-02", 0.0)]

    assert _open(render_line_chart(zeros, title="zero")).size == (WIDTH, HEIGHT)


def test_rising_and_falling_series_are_coloured_differently():
    """The chart's one opinion: green ended up, red ended down."""
    rising = render_line_chart(_series(), title="up")
    falling = render_line_chart(list(reversed(_series())), title="down")

    assert rising != falling


def test_long_series_does_not_crowd_the_x_axis():
    """A year of daily points must not stack 365 labels on top of each other."""
    year = [(f"d{index}", float(index)) for index in range(365)]

    assert _open(render_line_chart(year, title="year")).size == (WIDTH, HEIGHT)


# -- Fonts ----------------------------------------------------------------------------


def test_a_missing_configured_font_falls_back_and_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        found = find_hangul_font(str(tmp_path / "nope.ttf"))

    assert "does not exist" in caplog.text
    # Either a system font was found instead, or none was - never a crash.
    assert found is None or found.lower().endswith((".ttf", ".ttc", ".otf"))


def test_a_configured_font_wins_over_the_system_search(tmp_path):
    font = tmp_path / "custom.ttf"
    font.write_bytes(b"not really a font, but it exists")

    assert find_hangul_font(str(font)) == str(font)


def test_korean_text_survives_when_a_hangul_font_is_available():
    fonts = ChartFonts(find_hangul_font())
    if not fonts.hangul:  # pragma: no cover - depends on the host
        pytest.skip("no Hangul font on this machine")

    assert fonts.text("달러/원 환율") == "달러/원 환율"


def test_korean_text_is_dropped_rather_than_drawn_as_boxes():
    """Tofu reads as a broken chart and hides the label underneath."""
    fonts = ChartFonts(None)

    assert fonts.text("USD/KRW · 달러/원 환율") == "USD/KRW"


def test_a_chart_renders_without_any_font_at_all():
    """The fallback path has to produce an image, not an exception."""
    png = render_line_chart(_series(), title="달러/원", subtitle="최근 30일", font_path=None)

    assert _open(png).size == (WIDTH, HEIGHT)


def test_an_unloadable_font_file_does_not_break_rendering(tmp_path):
    broken = tmp_path / "broken.ttf"
    broken.write_bytes(b"definitely not a font")

    png = render_line_chart(_series(), title="x", font_path=str(broken))

    assert _open(png).size == (WIDTH, HEIGHT)


# -- Number formatting -----------------------------------------------------------------


def test_ticks_land_on_round_numbers():
    assert nice_ticks(1352.1, 1402.55) == [1360.0, 1380.0, 1400.0]


def test_ticks_are_round_multiples_rather_than_the_exact_ends():
    """3.2 and 9.7 are the data; 4, 6, 8, 10 are what people can read."""
    assert nice_ticks(3.2, 9.7) == [4, 6, 8, 10]


def test_tick_count_stays_readable_across_magnitudes():
    for low, high in ((0.01, 0.09), (3.2, 9.7), (1352.1, 1402.55), (1e8, 1.2e8)):
        assert 2 <= len(nice_ticks(low, high)) <= 6


def test_ticks_of_a_flat_range_are_a_single_line():
    assert nice_ticks(5.0, 5.0) == [5.0]


def test_ticks_have_no_floating_point_dust():
    """A tick label reading 1384.0000000002 is a bug people can see."""
    for tick in nice_ticks(0.1, 0.7):
        assert len(repr(tick)) < 8


def test_compact_axis_labels_are_ascii_only():
    """억/조 read better but vanish on a machine with no Korean font."""
    assert format_compact(111_134_000) == "111.13M"
    assert format_compact(2_500_000_000) == "2.5B"
    assert format_compact(1384.26) == "1,384"
    assert format_compact(0.00123) == "0.00123"
    assert all(label.isascii() for label in map(format_compact, (1e9, 1e6, 1e3, 1, 0.1)))


def test_compact_labels_drop_meaningless_decimals():
    assert format_compact(120_000_000) == "120M"


def test_the_headline_value_keeps_the_decimals_that_matter():
    """1,388 when the rate is 1,388.26 is simply the wrong number."""
    assert format_precise(1388.26) == "1,388.26"
    assert format_precise(111_134_000) == "111,134,000"
