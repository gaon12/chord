"""Crypto price skill - Upbit public ticker API (key-less).

Upbit's market-data endpoint requires no authentication for quotes:

    GET https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-ETH

Only KRW-quoted markets are used (``KRW-<SYMBOL>``), which is what a
Korean chat audience means by coin prices anyway.
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"

#: Human-readable names for the most common markets.
COIN_NAMES = {
    "BTC": "비트코인",
    "ETH": "이더리움",
    "XRP": "리플",
    "SOL": "솔라나",
    "DOGE": "도지코인",
    "ADA": "에이다",
    "AVAX": "아발란체",
    "DOT": "폴카닷",
}


def format_krw(value: float | int | None) -> str:
    """Format won amounts with separators; sub-won values keep decimals."""
    if value is None:
        return "?"
    if float(value).is_integer():
        return f"{int(value):,}원"
    return f"{value:,.2f}원"


def format_change(change: str | None, rate: float | None) -> str:
    """Signed percent string from Upbit's direction word + ratio."""
    if rate is None:
        return "?"
    percent = rate * 100
    sign = "-" if (change or "").upper() == "FALL" else "+"
    return f"{sign}{abs(percent):.2f}%"


def format_volume(value: float | None) -> str:
    """24h trade volume in compact human units (조 / 억)."""
    if value is None:
        return "?"
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}조원"
    if value >= 100_000_000:
        return f"{value / 100_000_000:,.1f}억원"
    return f"{int(value):,}원"


class CryptoPriceSkill(Skill):
    name = "get_crypto_price"
    description = (
        "Get current KRW prices for cryptocurrencies traded on Upbit "
        "(BTC, ETH, XRP, SOL, DOGE ...): price, change and 24h volume."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "coins": {
                "type": "string",
                "description": (
                    "Comma-separated coin symbols (default 'BTC'). "
                    "Examples: 'BTC', 'BTC,ETH', 'BTC,ETH,SOL,XRP'."
                ),
            }
        },
        "required": [],
    }

    async def run(self, coins: str = "BTC") -> str:
        symbols = [part.strip().upper() for part in coins.split(",") if part.strip()]
        symbols = symbols[:10] or ["BTC"]
        markets = ",".join(f"KRW-{symbol}" for symbol in symbols)

        data = await get_json(UPBIT_TICKER_URL, params={"markets": markets})
        if not isinstance(data, list) or not data:
            raise SkillHTTPError(f"No ticker data found for '{coins}'. Check the symbols.")

        lines: list[str] = []
        for entry in data:
            symbol = str(entry.get("market", "")).removeprefix("KRW-")
            name = COIN_NAMES.get(symbol, symbol)
            line = (
                f"{name}({symbol}): {format_krw(entry.get('trade_price'))} "
                f"({format_change(entry.get('change'), entry.get('change_rate'))}), "
                f"24h 거래대금 {format_volume(entry.get('acc_trade_price_24h'))}"
            )
            lines.append(line)
        return "\n".join(lines)
