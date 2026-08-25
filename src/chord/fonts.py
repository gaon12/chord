"""Getting hold of a font that can draw Hangul.

Charts are only as good as the font behind them, and "whatever the host
happens to have installed" is a bad foundation: the same bot draws a
different chart on Windows, on a Debian container and on a Mac, and on a
slim container it draws no Korean at all. So chord fetches one known
font - Noto Sans KR, the pan-CJK face Google publishes under the Open
Font License - and caches it next to the bot's other runtime state.

Resolution order, first hit wins:

1. ``CHART_FONT_PATH``, if it names a font that loads. An operator who
   picked a font gets that font, and no network call happens at all.
2. The cached download in ``FONT_CACHE_DIR``.
3. A fresh download from the CDN, written atomically into the cache.
4. Whatever Hangul-capable font the host has (:func:`find_hangul_font`).

Only step 3 touches the network, and only once per install. Every step
can fail without breaking anything: the end of the chain is ``None``,
which :mod:`chord.charts` handles by drawing ASCII labels rather than
tofu boxes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
from PIL import ImageFont

from chord.charts import find_hangul_font
from chord.config import Settings

logger = logging.getLogger(__name__)

#: Noto Sans KR Regular, from the CDN mirror of Google's noto-cjk repo.
#: Pinned to a release tag rather than @main so the bytes behind this URL
#: cannot change under a running install. The Korean subset OTF (4.6 MB)
#: rather than the variable TTF (10.4 MB): a chart needs one weight.
NOTO_SANS_KR_URL = (
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@Sans2.004"
    "/Sans/SubsetOTF/KR/NotoSansKR-Regular.otf"
)

#: Filename inside the cache directory. Naming the actual font means a
#: human looking in the cache can tell what it is.
CACHED_FONT_NAME = "NotoSansKR-Regular.otf"

#: Refuse anything wildly larger than the font we expect - a redirect to
#: something else should not fill a disk.
MAX_FONT_BYTES = 16 * 1024 * 1024

#: The download happens once per install and blocks the first chart, so
#: it may take a moment, but it must not hang a chat turn forever.
DOWNLOAD_TIMEOUT = 60.0

#: The resolution is memoized: it hits the filesystem and possibly the
#: network, and the answer cannot change while the process runs.
_resolved: str | None = None
_resolved_done = False

#: Serializes resolution, so two channels asking for a chart at the same
#: moment on a cold cache download the font once rather than twice.
_lock = asyncio.Lock()


def forget_resolved_font() -> None:
    """Drop the memoized answer, so the next call resolves again."""
    global _resolved, _resolved_done
    _resolved, _resolved_done = None, False


def is_usable_font(path: Path | str) -> bool:
    """Whether Pillow can actually open ``path`` as a font.

    Checked rather than assumed because the failure it catches is
    invisible otherwise: a CDN error page or a truncated download saved
    under a ``.otf`` name would sit in the cache forever, and every
    chart after it would silently lose its Korean.
    """
    try:
        ImageFont.truetype(str(path), 12)
    except (OSError, ValueError):
        return False
    return True


async def ensure_font(settings: Settings) -> str | None:
    """Path to a Hangul-capable font, downloading it once if needed.

    Returns None only when every step of the chain failed, which
    :func:`chord.charts.render_line_chart` renders as ASCII-only labels.
    """
    global _resolved, _resolved_done
    if _resolved_done:
        return _resolved

    async with _lock:
        # Another turn may have resolved it while this one waited.
        if _resolved_done:
            return _resolved
        _resolved = await _resolve(settings)
        _resolved_done = True
    return _resolved


async def _resolve(settings: Settings) -> str | None:
    configured = settings.chart_font_path
    if configured:
        if is_usable_font(configured):
            logger.info("Using the configured chart font %s.", configured)
            return str(configured)
        logger.warning(
            "CHART_FONT_PATH is %s, which is missing or not a font Pillow "
            "can read; falling back to the bundled Noto Sans KR.",
            configured,
        )

    cached = Path(settings.font_cache_dir) / CACHED_FONT_NAME
    if is_usable_font(cached):
        logger.debug("Using the cached chart font %s.", cached)
        return str(cached)

    logger.info("Downloading the chart font (Noto Sans KR) to %s ...", cached)
    if await _download(NOTO_SANS_KR_URL, cached):
        return str(cached)

    system = find_hangul_font()
    if system:
        logger.info("Falling back to the system font %s for charts.", system)
    return system


async def _download(url: str, destination: Path) -> bool:
    """Fetch a font into ``destination``; False when it did not work.

    Written to a ``.part`` file and moved into place only after it loads
    as a font, so an interrupted or bogus download never becomes the
    cache entry every later chart trusts.
    """
    partial = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with (
            httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            written = 0
            with partial.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    written += len(chunk)
                    if written > MAX_FONT_BYTES:
                        raise ValueError(f"larger than the {MAX_FONT_BYTES}-byte limit")
                    handle.write(chunk)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning("Could not download the chart font from %s: %s", url, exc)
        _discard(partial)
        return False

    if not is_usable_font(partial):
        logger.warning(
            "What %s returned is not a font Pillow can read; discarding it "
            "rather than caching a broken chart font.",
            url,
        )
        _discard(partial)
        return False

    try:
        partial.replace(destination)
    except OSError as exc:
        logger.warning("Could not move the downloaded font into place: %s", exc)
        _discard(partial)
        return False

    logger.info("Cached the chart font at %s (%d bytes).", destination, destination.stat().st_size)
    return True


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a locked temp file is not worth failing over
        logger.debug("Could not remove the partial font download %s.", path)
