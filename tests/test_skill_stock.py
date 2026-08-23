"""Tests for the stock-price skill (Yahoo Finance chart API, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.stock import StockPriceSkill, format_price

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"


def _chart_response(price=232.5, previous=229.8, currency="USD"):
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "AAPL",
                        "longName": "Apple Inc.",
                        "regularMarketPrice": price,
                        "chartPreviousClose": previous,
                        "currency": currency,
                        "exchangeName": "NMS",
                    }
                }
            ],
            "error": None,
        }
    }


@respx.mock
async def test_stock_happy_path_with_change():
    respx.get(CHART_URL + "AAPL").respond(json=_chart_response())

    result = await StockPriceSkill().run(symbol="aapl")

    assert "Apple Inc. (AAPL)" in result
    assert "$232.50" in result
    # 232.5 vs previous close of 229.8 is about +1.17 percent.
    assert "+1.17%" in result
    assert "$229.80" in result


@respx.mock
async def test_stock_negative_change():
    respx.get(CHART_URL + "MSFT").respond(json=_chart_response(price=100.0, previous=110.0))

    result = await StockPriceSkill().run(symbol="MSFT")

    assert "-9.09%" in result


@respx.mock
async def test_korean_ticker_uses_won_symbol():
    # Samsung Electronics on KOSPI uses the .KS suffix.
    respx.get(CHART_URL + "005930.KS").respond(
        json=_chart_response(price=71500, previous=70000, currency="KRW")
    )

    result = await StockPriceSkill().run(symbol="005930.KS")

    won = "\u20a9"  # won sign, spelled out to survive any editor encoding
    assert f"{won}71,500.00" in result
    assert "+2.14%" in result


@respx.mock
async def test_unknown_symbol_reports_readable_error():
    respx.get(CHART_URL + "NOPE").respond(
        json={
            "chart": {
                "result": None,
                "error": {
                    "code": "Not Found",
                    "description": "No data found, symbol may be delisted",
                },
            }
        }
    )

    with pytest.raises(SkillHTTPError, match="symbol may be delisted"):
        await StockPriceSkill().run(symbol="NOPE")


@respx.mock
async def test_missing_previous_close_still_replies_price():
    payload = _chart_response()
    payload["chart"]["result"][0]["meta"].pop("chartPreviousClose")
    respx.get(CHART_URL + "AAPL").respond(json=payload)

    result = await StockPriceSkill().run(symbol="AAPL")

    assert "$232.50" in result
    assert "%" not in result  # no change info without a baseline


@pytest.mark.parametrize(
    ("value", "currency", "expected"),
    [
        (232.5, "USD", "$232.50"),
        (71500, "KRW", "\u20a971,500.00"),
        (1234.5, "CHF", "CHF 1,234.50"),
        (None, "USD", "n/a"),
    ],
)
def test_format_price(value, currency, expected):
    assert format_price(value, currency) == expected
