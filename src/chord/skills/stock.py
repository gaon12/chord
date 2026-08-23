"""Stock-price skill - latest quote via Yahoo Finance (key-less).

Uses the public chart endpoint that powers the Yahoo website:

    GET https://query1.finance.yahoo.com/v8/finance/chart/<SYMBOL>

It works for US tickers (AAPL) and international ones with an exchange
suffix (005930.KS = Samsung Electronics on KOSPI).
"""

from __future__ import annotations

from typing import Any, ClassVar

from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

#: Compact display symbols for common currencies.
CURRENCY_SYMBOLS = {"USD": "$", "KRW": "₩", "JPY": "¥", "EUR": "€", "GBP": "£"}


def format_price(value: float | None, currency: str) -> str:
    """Render a price with its currency, e.g. 231.4 USD -> '$231.40'."""
    if value is None:
        return "n/a"
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    return f"{symbol}{value:,.2f}"


class StockPriceSkill(Skill):
    name = "get_stock_price"
    description = (
        "Get the latest stock price and daily change for a ticker symbol. "
        "US examples: AAPL, MSFT, TSLA. Korean stocks need a suffix: "
        ".KS for KOSPI (e.g. 005930.KS Samsung) or .KQ for KOSDAQ."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": (
                    "Ticker symbol, optionally with exchange suffix (e.g. 'AAPL', '005930.KS')."
                ),
            }
        },
        "required": ["symbol"],
    }

    async def run(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        meta = await _fetch_quote_meta(symbol)

        price = meta.get("regularMarketPrice")
        previous = meta.get("chartPreviousClose") or meta.get("previousClose")
        currency = meta.get("currency", "")
        name = meta.get("longName") or meta.get("shortName") or symbol

        line = f"{name} ({symbol}): {format_price(price, currency)}"

        if price is not None and previous:
            change_pct = (price - float(previous)) / float(previous) * 100
            direction = "+" if change_pct >= 0 else ""
            line += (
                f", {direction}{change_pct:.2f}% today "
                f"(prev close {format_price(float(previous), currency)})."
            )
        else:
            line += "."
        return line


async def _fetch_quote_meta(symbol: str) -> dict[str, Any]:
    """Fetch the chart payload and return the ``meta`` block."""
    data = await get_json(CHART_URL.format(symbol=symbol))
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        error = (data.get("chart") or {}).get("error")
        detail = error.get("description") if isinstance(error, dict) else None
        raise SkillHTTPError(detail or f"No quote found for '{symbol}'.")
    meta = result[0].get("meta") or {}
    if not meta:
        raise SkillHTTPError(f"Quote data for '{symbol}' is empty.")
    return meta
