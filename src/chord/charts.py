"""Drawing a value series as a PNG chart Discord can display inline.

Discord renders no charts of its own and does not preview SVG
attachments, so a trend has to arrive as a raster image. This module
draws one with Pillow: a line chart on a dark card that sits naturally
in the client, sized for a chat window rather than a report.

Pillow rather than matplotlib on purpose. matplotlib brings numpy and
friends - tens of megabytes and about a second of import time - to draw
one polyline, and resolves fonts through a cache that is exactly what
produces tofu boxes on a machine whose Korean font it never found.
Here the font is an explicit file path, so "글자 깨짐" is a question with
a checkable answer: :func:`find_hangul_font` either returns a path or
says it found nothing, and text is downgraded rather than drawn as
boxes.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

#: Everything is drawn at this multiple and scaled back down, because
#: Pillow draws aliased lines. A 2x render plus a LANCZOS downscale is
#: the cheapest antialiasing available and costs a few milliseconds.
SUPERSAMPLE = 2

#: Final image size. Wide enough to read on a phone in Discord's inline
#: preview without anyone having to tap it open.
WIDTH = 900
HEIGHT = 460

# Discord's own dark palette, so the card does not glare in a channel.
COLOR_BACKGROUND = (30, 31, 34)
COLOR_PANEL = (43, 45, 49)
COLOR_GRID = (63, 65, 71)
COLOR_TEXT = (219, 222, 225)
COLOR_MUTED = (148, 155, 164)
COLOR_UP = (59, 165, 93)
COLOR_DOWN = (237, 66, 69)
COLOR_FLAT = (114, 137, 218)

#: Fonts that ship with the usual host platforms and cover Hangul.
#: Ordered by how good they look, not by platform: only one will exist.
FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NotoSansKR-VF.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    # Linux - Noto CJK is what most distros package, Nanum is common in KR
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
)


def scaled(value: float) -> int:
    """A layout coordinate in supersampled pixels."""
    return int(round(value * SUPERSAMPLE))


@lru_cache(maxsize=8)
def find_hangul_font(configured: str | None = None) -> str | None:
    """Path to a font that can draw Hangul, or None if there is none.

    ``configured`` (``CHART_FONT_PATH``) wins outright - an operator who
    names a font has a reason, and a missing file there is a mistake
    worth a warning rather than a silent fallback. Cached because this
    hits the filesystem and the answer cannot change while the bot runs.
    """
    if configured:
        if Path(configured).is_file():
            return configured
        logger.warning(
            "CHART_FONT_PATH points at %s, which does not exist; "
            "falling back to the system font search.",
            configured,
        )

    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    logger.warning(
        "No Hangul-capable font found, so chart labels will be drawn in "
        "ASCII only (Korean text is dropped rather than rendered as "
        "boxes). Install a Korean font - fonts-nanum or Noto Sans KR - "
        "or point CHART_FONT_PATH at a .ttf/.otf file."
    )
    return None


class ChartFonts:
    """The three text sizes a chart needs, from one font file."""

    def __init__(self, path: str | None) -> None:
        #: False when only the built-in font is available, which cannot
        #: draw Hangul - see :meth:`text`.
        self.hangul = path is not None
        self.title = self._load(path, 26)
        self.value = self._load(path, 26)
        self.label = self._load(path, 17)

    @staticmethod
    def _load(path: str | None, size: int):
        size = scaled(size)
        if path is None:
            # Pillow >= 10.1 scales its built-in font, so the fallback is
            # still legible instead of a 10px bitmap.
            return ImageFont.load_default(size=size)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            logger.warning("Could not load font %s; using the built-in one.", path)
            return ImageFont.load_default(size=size)

    def text(self, text: str) -> str:
        """Drop what the loaded font cannot draw.

        A missing glyph renders as a tofu box, which reads as a broken
        chart and hides the label underneath it. With no Hangul font
        available, "USD/KRW 30d" beats "□□□ 30d".
        """
        if self.hangul:
            return text
        ascii_only = "".join(char if char.isascii() else " " for char in text)
        # Dropping the letters of "달러/원" leaves a bare "/" behind, which
        # is noise pretending to be a label - keep only real words.
        words = [word for word in ascii_only.split() if any(c.isalnum() for c in word)]
        return " ".join(words)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values spanning roughly ``[low, high]``.

    Axis labels people can read are 1, 2, 2.5 or 5 times a power of ten
    apart - never 1384.2617 apart, which is what dividing the range by
    five gives you. Landing on round numbers means the end ticks do not
    line up with the data ends: they can fall just inside or up to half
    a step outside. :func:`render_line_chart` drops the ones that land
    outside the padded axis, so no tick is ever drawn off the panel.
    """
    if not math.isfinite(low) or not math.isfinite(high):
        return []
    if high <= low:
        # A flat series still deserves one labelled line through it.
        return [low]

    rough = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(rough))
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if step >= rough:
            break

    ticks: list[float] = []
    value = math.floor(low / step) * step
    while value < high + step / 2:
        if value >= low - step / 2:
            # Kill the float dust that makes a tick read 1384.0000000002.
            ticks.append(round(value, 10))
        value += step
    return ticks


#: Headroom above and below the data, as a fraction of its range. A
#: line touching the top edge of the panel reads as clipped.
AXIS_PADDING = 0.08


def format_compact(value: float) -> str:
    """Axis-friendly number, ASCII only so any font can draw it.

    Korean units (억, 조) would read better on a KRW chart but are
    exactly the characters that vanish when no Hangul font is present,
    and an axis with no labels is worse than an axis in English.
    """
    magnitude = abs(value)
    if magnitude >= 1e9:
        return _trim(f"{value / 1e9:,.2f}") + "B"
    if magnitude >= 1e6:
        return _trim(f"{value / 1e6:,.2f}") + "M"
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def _trim(text: str) -> str:
    """Drop decimals that say nothing: 120.00M is noisier than 120M."""
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_precise(value: float) -> str:
    """The headline number, where the decimals are the whole point.

    An axis tick reading 1,388 is fine; a headline exchange rate reading
    1,388 when it is 1,388.26 is just wrong, and 111,134,000 as
    "111.13M" hides the won. Precision scales with magnitude.
    """
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def render_line_chart(
    points: Sequence[tuple[str, float]],
    *,
    title: str,
    subtitle: str = "",
    font_path: str | None = None,
) -> bytes:
    """Draw ``points`` as a line chart and return PNG bytes.

    Args:
        points: ``(label, value)`` oldest first. Labels are drawn as
            given - dates, hours, anything - so this module needs to
            know nothing about what is being plotted.
        title: Top-left heading, e.g. ``USD/KRW``.
        subtitle: Smaller line under it, e.g. the period and source.
        font_path: A font to use; ``None`` searches the system.

    Raises:
        ValueError: With fewer than two points there is no line to draw.
    """
    if len(points) < 2:
        raise ValueError("A chart needs at least two data points.")

    values = [float(value) for _label, value in points]
    fonts = ChartFonts(font_path if font_path else find_hangul_font())
    accent = _accent_for(values)

    image = Image.new("RGBA", (scaled(WIDTH), scaled(HEIGHT)), (*COLOR_BACKGROUND, 255))
    draw = ImageDraw.Draw(image)

    plot = _plot_box()
    draw.rectangle(plot, fill=(*COLOR_PANEL, 255))

    low, high = min(values), max(values)
    axis_low, axis_high = _axis_bounds(low, high)
    ticks = [tick for tick in nice_ticks(low, high) if axis_low <= tick <= axis_high]

    _draw_grid(draw, plot, ticks, axis_low, axis_high, fonts)
    pixels = _to_pixels(values, plot, axis_low, axis_high)
    image = _draw_area(image, pixels, plot, accent)
    draw = ImageDraw.Draw(image)
    draw.line(pixels, fill=(*accent, 255), width=scaled(2.5), joint="curve")
    _draw_last_point(draw, pixels[-1], accent)
    _draw_x_labels(draw, [label for label, _value in points], pixels, plot, fonts)
    _draw_heading(draw, title, subtitle, values, accent, fonts)

    return _to_png(image)


def _plot_box() -> tuple[int, int, int, int]:
    """The rectangle the data is drawn inside, in supersampled pixels."""
    return (scaled(92), scaled(104), scaled(WIDTH - 26), scaled(HEIGHT - 46))


def _accent_for(values: list[float]) -> tuple[int, int, int]:
    """Green when it ended up, red when down - the chart's one opinion."""
    if values[-1] > values[0]:
        return COLOR_UP
    if values[-1] < values[0]:
        return COLOR_DOWN
    return COLOR_FLAT


def _axis_bounds(low: float, high: float) -> tuple[float, float]:
    """Value range the plot area spans.

    Padded on both sides so the peak has air above it rather than
    sitting on the panel edge, and never zero-height: a flat series
    still has to divide by something.
    """
    span = high - low
    if span <= 0:
        # A perfectly flat series: invent a range so it draws mid-panel.
        padding = abs(low) * 0.01 or 1.0
        return low - padding, high + padding
    padding = span * AXIS_PADDING
    return low - padding, high + padding


def _to_pixels(
    values: list[float],
    plot: tuple[int, int, int, int],
    axis_low: float,
    axis_high: float,
) -> list[tuple[int, int]]:
    left, top, right, bottom = plot
    span = axis_high - axis_low
    step = (right - left) / (len(values) - 1)
    return [
        (
            int(left + index * step),
            int(bottom - (value - axis_low) / span * (bottom - top)),
        )
        for index, value in enumerate(values)
    ]


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    ticks: list[float],
    axis_low: float,
    axis_high: float,
    fonts: ChartFonts,
) -> None:
    left, top, right, bottom = plot
    span = axis_high - axis_low or 1.0
    for tick in ticks:
        y = int(bottom - (tick - axis_low) / span * (bottom - top))
        draw.line([(left, y), (right, y)], fill=(*COLOR_GRID, 255), width=scaled(0.5))
        label = format_compact(tick)
        width = _text_width(draw, label, fonts.label)
        draw.text(
            (left - scaled(10) - width, y - scaled(9)),
            label,
            font=fonts.label,
            fill=(*COLOR_MUTED, 255),
        )


def _draw_area(
    image: Image.Image,
    pixels: list[tuple[int, int]],
    plot: tuple[int, int, int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    """Translucent fill under the line, composited over the panel."""
    _left, _top, _right, bottom = plot
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    polygon = [(pixels[0][0], bottom), *pixels, (pixels[-1][0], bottom)]
    ImageDraw.Draw(overlay).polygon(polygon, fill=(*accent, 48))
    return Image.alpha_composite(image, overlay)


def _draw_last_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    accent: tuple[int, int, int],
) -> None:
    x, y = point
    radius = scaled(4)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=(*accent, 255),
        outline=(*COLOR_BACKGROUND, 255),
        width=scaled(1.5),
    )


def _draw_x_labels(
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    pixels: list[tuple[int, int]],
    plot: tuple[int, int, int, int],
    fonts: ChartFonts,
) -> None:
    """Up to five date labels, skipped rather than overlapped."""
    left, _top, right, bottom = plot
    wanted = min(5, len(labels))
    indices = (
        [round(i * (len(labels) - 1) / (wanted - 1)) for i in range(wanted)] if wanted > 1 else [0]
    )

    last_end = left - scaled(1000)
    for index in indices:
        text = fonts.text(str(labels[index]))
        if not text:
            continue
        width = _text_width(draw, text, fonts.label)
        x = pixels[index][0] - width // 2
        x = max(left, min(x, right - width))
        if x < last_end + scaled(12):
            continue
        draw.text((x, bottom + scaled(12)), text, font=fonts.label, fill=(*COLOR_MUTED, 255))
        last_end = x + width


def _draw_heading(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    values: list[float],
    accent: tuple[int, int, int],
    fonts: ChartFonts,
) -> None:
    draw.text(
        (scaled(28), scaled(26)),
        fonts.text(title),
        font=fonts.title,
        fill=(*COLOR_TEXT, 255),
    )
    if subtitle:
        draw.text(
            (scaled(28), scaled(64)),
            fonts.text(subtitle),
            font=fonts.label,
            fill=(*COLOR_MUTED, 255),
        )

    latest = f"{format_precise(values[-1])}  {_change_text(values)}"
    width = _text_width(draw, latest, fonts.value)
    draw.text(
        (scaled(WIDTH - 28) - width, scaled(26)),
        latest,
        font=fonts.value,
        fill=(*accent, 255),
    )


def _change_text(values: list[float]) -> str:
    """Percentage move across the whole series, signed."""
    first, last = values[0], values[-1]
    if not first:
        return ""
    percent = (last - first) / abs(first) * 100
    return f"{percent:+.2f}%"


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS).save(buffer, format="PNG")
    return buffer.getvalue()
