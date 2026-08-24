"""Tests for the crypto price skill (Upbit ticker, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.crypto import CryptoPriceSkill, format_change, format_krw, format_volume

UPBIT_URL = "https://api.upbit.com/v1/ticker"


def _ticker_response():
    return [
        {
            "market": "KRW-BTC",
            "trade_price": 105_672_000.0,
            "change": "FALL",
            "change_rate": 0.0099685,
            "acc_trade_price_24h": 512_345_678_901.0,
        },
        {
            "market": "KRW-ETH",
            "trade_price": 4_123_000.0,
            "change": "RISE",
            "change_rate": 0.0312,
            "acc_trade_price_24h": 234_500_000.0,
        },
    ]


@respx.mock
async def test_crypto_happy_path_multiple_coins():
    respx.get(UPBIT_URL).respond(json=_ticker_response())

    result = await CryptoPriceSkill().run(coins="btc, eth")

    assert "비트코인(BTC): 105,672,000원 (-1.00%)" in result
    assert "24h 거래대금 5,123.5억원" in result
    assert "이더리움(ETH): 4,123,000원 (+3.12%)" in result


@respx.mock
async def test_crypto_default_is_btc():
    route = respx.get(UPBIT_URL).respond(json=[_ticker_response()[0]])

    await CryptoPriceSkill().run()

    # GET params live in the URL query string, not the body.
    assert "KRW-BTC" in str(route.calls[0].request.url)


@respx.mock
async def test_unknown_symbol_reports_error():
    respx.get(UPBIT_URL).respond(json=[])

    with pytest.raises(SkillHTTPError, match="No ticker data"):
        await CryptoPriceSkill().run(coins="NOPE")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "?"), (105_672_000.0, "105,672,000원"), (123.45, "123.45원")],
)
def test_format_krw(value, expected):
    assert format_krw(value) == expected


def test_format_change_signs():
    assert format_change("FALL", 0.01) == "-1.00%"
    assert format_change("RISE", 0.025) == "+2.50%"
    assert format_change("EVEN", 0) == "+0.00%"


def test_format_volume_units():
    assert format_volume(512_345_678_901) == "5,123.5억원"
    assert format_volume(234_500_000) == "2.3억원"
