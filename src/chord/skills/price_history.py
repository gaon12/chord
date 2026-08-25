"""Price-history skill - how a number moved, drawn as a chart.

Three domains share one tool on purpose. "환율 추이 보여줘", "테슬라 한 달
차트", "비트코인 일주일 흐름" are the same question asked about different
markets, and every tool definition is re-sent with every request - three
near-identical schemas would cost input tokens on every single message
to save the model one enum value.

Sources are the key-less ones the single-value skills already use:

* exchange - Frankfurter time series (ECB reference rates)
* stock    - Yahoo Finance chart endpoint
* crypto   - Upbit daily/weekly candles (KRW markets)

The numbers go back to the model as text; the chart is attached to the
reply through :mod:`chord.attachments`, because the model cannot look at
a PNG and should not be asked to pay tokens for one.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

from chord.attachments import attach
from chord.charts import format_precise, render_line_chart
from chord.config import Settings
from chord.fonts import ensure_font
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

FRANKFURTER_SERIES_URL = "https://api.frankfurter.dev/v1/{start}..{end}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
UPBIT_DAY_CANDLES_URL = "https://api.upbit.com/v1/candles/days"
UPBIT_WEEK_CANDLES_URL = "https://api.upbit.com/v1/candles/weeks"

DEFAULT_DAYS = 30

#: Two points make a line; below that there is nothing to show.
MIN_DAYS = 2

#: A chat window is not a trading terminal - beyond a year the daily
#: wiggle is noise and the request gets slow for no gain.
MAX_DAYS = 365

#: Upbit refuses a count above 200, so longer crypto ranges switch to
#: weekly candles rather than paging through history.
UPBIT_MAX_CANDLES = 200

#: Currency pairs default to won: a Korean channel asking about "달러"
#: means USD/KRW, and spelling both out every time is friction.
DEFAULT_QUOTE_CURRENCY = "KRW"

_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_PAIR_SEPARATORS = re.compile(r"[/\-_\s:]+")


@dataclass(frozen=True)
class Series:
    """One fetched history, ready to summarize and to draw."""

    title: str
    subtitle: str
    source: str
    points: list[tuple[str, float]]


class PriceHistorySkill(Skill):
    name = "get_price_history"
    description = (
        "Show how a price moved over time and post a chart image of it: "
        "exchange rates, stocks or crypto. Use this for any question "
        "about a trend, a period or a history, or when a graph or chart "
        "is asked for (추이, 흐름, 변동, 그래프, 차트, 최근 한 달, 지난주). "
        "For a single current value use get_exchange_rate, "
        "get_stock_price or get_crypto_price instead."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["exchange", "stock", "crypto"],
                "description": "Which market the symbol belongs to.",
            },
            "symbol": {
                "type": "string",
                "description": (
                    "What to chart. exchange: a currency pair like "
                    "'USD/KRW' (a bare 'USD' means USD/KRW). stock: a "
                    "ticker like 'AAPL' or '005930.KS' for KOSPI. "
                    "crypto: a coin symbol like 'BTC' (Upbit KRW market)."
                ),
            },
            "days": {
                "type": "integer",
                "description": f"How many days back to chart (default {DEFAULT_DAYS}, max {MAX_DAYS}).",
            },
        },
        "required": ["kind", "symbol"],
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, kind: str, symbol: str, days: int = DEFAULT_DAYS) -> str:
        kind = (kind or "").strip().lower()
        fetch = _FETCHERS.get(kind)
        if fetch is None:
            raise SkillHTTPError(f"Unknown kind '{kind}'. Use exchange, stock or crypto.")

        days = _clamp_days(days)
        series = await fetch(symbol.strip(), days)
        if len(series.points) < 2:
            raise SkillHTTPError(
                f"Only {len(series.points)} data point(s) came back for "
                f"'{symbol}' - not enough to chart. Try a longer period."
            )

        # Resolved here rather than in the renderer because the first
        # call may download the font, and that has to be awaited.
        font_path = await ensure_font(self._settings)
        return _summarize(series, self._attach_chart(series, kind, font_path))

    def _attach_chart(self, series: Series, kind: str, font_path: str | None) -> bool:
        """Render and attach the chart; False if it could not be sent.

        A drawing failure must never cost the answer - the numbers in
        the summary are the substance, so a broken font or an unwritable
        buffer downgrades the reply instead of raising through it.
        """
        try:
            png = render_line_chart(
                series.points,
                title=series.title,
                subtitle=series.subtitle,
                font_path=font_path,
            )
        except Exception:  # noqa: BLE001 - the text answer still stands
            logger.exception("Could not render the %s chart for %s", kind, series.title)
            return False
        return attach(_filename(kind, series), png)


def _clamp_days(days: int) -> int:
    """Keep a model-chosen period inside what the sources will serve."""
    try:
        days = int(days)
    except (TypeError, ValueError):
        return DEFAULT_DAYS
    return max(MIN_DAYS, min(days, MAX_DAYS))


def _filename(kind: str, series: Series) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", series.title).strip("-").lower() or kind
    return f"{kind}-{slug}.png"


# -- Sources -------------------------------------------------------------------------


def parse_currency_pair(symbol: str) -> tuple[str, str]:
    """Split 'USD/KRW', 'usd-krw' or 'USDKRW' into base and quote.

    A bare 'USD' becomes USD/KRW: the audience is a Korean channel, and
    making people spell out the obvious half is friction.
    """
    cleaned = _PAIR_SEPARATORS.sub(" ", symbol.strip()).strip()
    parts = cleaned.split()
    if len(parts) == 1 and len(parts[0]) == 6:
        parts = [parts[0][:3], parts[0][3:]]
    if len(parts) == 1:
        parts.append(DEFAULT_QUOTE_CURRENCY)
    if len(parts) != 2:
        raise SkillHTTPError(f"'{symbol}' is not a currency pair - try 'USD/KRW'.")

    base, quote = (part.upper() for part in parts)
    for code in (base, quote):
        if not _CURRENCY_RE.match(code):
            raise SkillHTTPError(f"'{code}' is not a 3-letter currency code.")
    if base == quote:
        raise SkillHTTPError(f"{base} to {quote} is always 1 - pick two currencies.")
    return base, quote


async def fetch_exchange_series(symbol: str, days: int) -> Series:
    """ECB reference rates over a date range, via Frankfurter."""
    base, quote = parse_currency_pair(symbol)
    end = date.today()
    start = end - timedelta(days=days)

    data = await get_json(
        FRANKFURTER_SERIES_URL.format(start=start.isoformat(), end=end.isoformat()),
        params={"base": base, "symbols": quote},
    )
    rates = data.get("rates") or {}
    points = [
        (day[5:], float(values[quote]))
        for day, values in sorted(rates.items())
        if isinstance(values, dict) and quote in values
    ]
    if not points:
        raise SkillHTTPError(f"No {base}/{quote} rates published for that period.")

    return Series(
        title=f"{base}/{quote}",
        # ECB publishes on business days only, so the point count is not
        # the day count and saying so keeps the summary honest.
        subtitle=f"{start.isoformat()} - {end.isoformat()} · {len(points)} business days",
        source="ECB reference rates (Frankfurter)",
        points=points,
    )


async def fetch_stock_series(symbol: str, days: int) -> Series:
    """Daily closes from the endpoint that powers the Yahoo website."""
    ticker = symbol.upper()
    end = int(time.time())
    data = await get_json(
        YAHOO_CHART_URL.format(symbol=ticker),
        params={
            "period1": end - days * 86_400,
            "period2": end,
            # Past a year of daily bars the chart is mush; weekly reads.
            "interval": "1wk" if days > 180 else "1d",
        },
    )
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        error = (data.get("chart") or {}).get("error")
        detail = error.get("description") if isinstance(error, dict) else None
        raise SkillHTTPError(detail or f"No history found for '{ticker}'.")

    block = result[0]
    meta = block.get("meta") or {}
    timestamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []

    points = [
        (datetime.fromtimestamp(stamp, tz=UTC).strftime("%m-%d"), float(close))
        # Yahoo pads holidays and halts with nulls; they are gaps, not zeros.
        for stamp, close in zip(timestamps, closes, strict=False)
        if close is not None
    ]
    if not points:
        raise SkillHTTPError(f"Yahoo returned no closing prices for '{ticker}'.")

    name = meta.get("longName") or meta.get("shortName") or ticker
    currency = meta.get("currency") or ""
    return Series(
        title=f"{name} ({ticker})",
        subtitle=f"last {days} days · {len(points)} sessions"
        + (f" · {currency}" if currency else ""),
        source="Yahoo Finance",
        points=points,
    )


async def fetch_crypto_series(symbol: str, days: int) -> Series:
    """Upbit KRW-market candles, daily up to Upbit's own limit."""
    coin = symbol.upper().removeprefix("KRW-").strip()
    if not coin:
        raise SkillHTTPError("Which coin? Try 'BTC'.")
    market = f"KRW-{coin}"

    weekly = days > UPBIT_MAX_CANDLES
    url = UPBIT_WEEK_CANDLES_URL if weekly else UPBIT_DAY_CANDLES_URL
    count = math.ceil(days / 7) if weekly else days

    data = await get_json(url, params={"market": market, "count": count})
    if not isinstance(data, list) or not data:
        raise SkillHTTPError(f"Upbit has no {market} market, or returned nothing.")

    # Upbit answers newest first; a chart reads oldest first.
    points = [
        (str(candle.get("candle_date_time_kst", ""))[5:10], float(candle["trade_price"]))
        for candle in reversed(data)
        if candle.get("trade_price") is not None
    ]
    if not points:
        raise SkillHTTPError(f"No usable candles came back for {market}.")

    unit = "weekly" if weekly else "daily"
    return Series(
        title=f"{coin}/KRW",
        subtitle=f"last {days} days · {len(points)} {unit} candles (KST)",
        source="Upbit",
        points=points,
    )


_FETCHERS = {
    "exchange": fetch_exchange_series,
    "stock": fetch_stock_series,
    "crypto": fetch_crypto_series,
}


# -- Reporting ------------------------------------------------------------------------


def _summarize(series: Series, chart_attached: bool) -> str:
    """What the model gets back: the numbers, and what to say about them.

    The closing instruction matters. Without it a model that cannot see
    the image will happily narrate a shape it invented ("완만한 상승 후
    급락") from nothing at all.
    """
    values = [value for _label, value in series.points]
    first, last = values[0], values[-1]
    low_label, low = min(series.points, key=lambda point: point[1])
    high_label, high = max(series.points, key=lambda point: point[1])
    change = (last - first) / abs(first) * 100 if first else 0.0

    lines = [
        f"{series.title} — {series.subtitle}",
        (
            f"latest {format_precise(last)} | "
            f"first {format_precise(first)} | "
            f"change {change:+.2f}% | "
            f"low {format_precise(low)} on {low_label} | "
            f"high {format_precise(high)} on {high_label}"
        ),
        f"Source: {series.source}.",
    ]
    if chart_attached:
        lines.append(
            "A chart image is attached to this reply. Point the user at "
            "it and use only the numbers above - you cannot see the "
            "image, so do not describe its shape."
        )
    else:
        lines.append("The chart image could not be attached; answer with the numbers only.")
    return "\n".join(lines)
