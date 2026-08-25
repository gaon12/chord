"""Tests for the price-history skill (all three sources mocked)."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest
import respx

from chord.attachments import collected, reset_attachments, start_collecting
from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.price_history import (
    DEFAULT_DAYS,
    MAX_DAYS,
    PriceHistorySkill,
    _clamp_days,
    parse_currency_pair,
)

FX_PATTERN = re.compile(r"https://api\.frankfurter\.dev/v1/.*")
YAHOO_PATTERN = re.compile(r"https://query1\.finance\.yahoo\.com/v8/finance/chart/.*")
UPBIT_DAYS = "https://api.upbit.com/v1/candles/days"
UPBIT_WEEKS = "https://api.upbit.com/v1/candles/weeks"


def _skill() -> PriceHistorySkill:
    return PriceHistorySkill(Settings(_env_file=None, discord_token="t", openai_api_key="k"))


@pytest.fixture(autouse=True)
def offline_font(monkeypatch):
    """No test downloads a font; chord.fonts has its own tests for that."""

    async def no_font(_settings):
        return None

    monkeypatch.setattr("chord.skills.price_history.ensure_font", no_font)


@pytest.fixture
def turn():
    """A collecting chat turn, so attachments have somewhere to land."""
    token = start_collecting()
    yield
    reset_attachments(token)


def _fx_series(count: int = 5) -> dict:
    start = date(2026, 8, 1)
    return {
        "base": "USD",
        "rates": {
            (start + timedelta(days=index)).isoformat(): {"KRW": 1380.0 + index}
            for index in range(count)
        },
    }


def _yahoo_series(closes: list[float | None]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "meta": {"longName": "Apple Inc.", "currency": "USD"},
                    "timestamp": [1_754_000_000 + index * 86_400 for index in range(len(closes))],
                    "indicators": {"quote": [{"close": closes}]},
                }
            ]
        }
    }


def _upbit_candles(prices: list[float]) -> list[dict]:
    """Upbit answers newest first - that ordering is the point."""
    return [
        {
            "candle_date_time_kst": f"2026-08-{len(prices) - index:02d}T09:00:00",
            "trade_price": price,
        }
        for index, price in enumerate(reversed(prices))
    ]


# -- Currency pairs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    ["USD/KRW", "usd-krw", "USD KRW", "USDKRW", "usd_krw", "USD:KRW"],
)
def test_currency_pairs_are_accepted_in_every_shape_people_type(symbol):
    assert parse_currency_pair(symbol) == ("USD", "KRW")


def test_a_bare_currency_means_against_the_won():
    """A Korean channel asking about 달러 means USD/KRW."""
    assert parse_currency_pair("usd") == ("USD", "KRW")


def test_a_pair_of_the_same_currency_is_rejected():
    with pytest.raises(SkillHTTPError, match="always 1"):
        parse_currency_pair("USD/USD")


def test_a_non_currency_is_rejected_before_the_request():
    with pytest.raises(SkillHTTPError, match="3-letter"):
        parse_currency_pair("BITCOIN/KRW")


# -- Period ----------------------------------------------------------------------------


def test_days_are_clamped_to_what_the_sources_serve():
    assert _clamp_days(5000) == MAX_DAYS
    assert _clamp_days(0) == 2
    assert _clamp_days(30) == 30


def test_a_nonsense_period_falls_back_to_the_default():
    """Models pass strings and nulls; neither is worth an error."""
    assert _clamp_days("소수") == DEFAULT_DAYS
    assert _clamp_days(None) == DEFAULT_DAYS


# -- Exchange rates ---------------------------------------------------------------------


@respx.mock
async def test_exchange_history_summarizes_and_attaches_a_chart(turn):
    respx.get(FX_PATTERN).respond(json=_fx_series())

    result = await _skill().run(kind="exchange", symbol="USD/KRW", days=5)

    assert "USD/KRW" in result
    assert "latest 1,384.00" in result
    assert "change +0.29%" in result
    assert "chart image is attached" in result

    files = collected()
    assert [file.filename for file in files] == ["exchange-usd-krw.png"]
    assert files[0].data.startswith(b"\x89PNG")


@respx.mock
async def test_exchange_history_is_ordered_oldest_first(turn):
    """Dicts come back in whatever order; a chart needs chronology."""
    respx.get(FX_PATTERN).respond(
        json={"rates": {"2026-08-03": {"KRW": 3.0}, "2026-08-01": {"KRW": 1.0}}}
    )

    result = await _skill().run(kind="exchange", symbol="USD/KRW", days=5)

    assert "first 1.00" in result
    assert "latest 3.00" in result


@respx.mock
async def test_an_empty_rate_series_is_reported_not_charted(turn):
    respx.get(FX_PATTERN).respond(json={"rates": {}})

    with pytest.raises(SkillHTTPError, match="No USD/KRW rates"):
        await _skill().run(kind="exchange", symbol="USD/KRW", days=5)


@respx.mock
async def test_a_single_data_point_is_not_a_chart(turn):
    respx.get(FX_PATTERN).respond(json={"rates": {"2026-08-01": {"KRW": 1380.0}}})

    with pytest.raises(SkillHTTPError, match="not enough to chart"):
        await _skill().run(kind="exchange", symbol="USD/KRW", days=5)


# -- Stocks ------------------------------------------------------------------------------


@respx.mock
async def test_stock_history_uses_the_closing_prices(turn):
    respx.get(YAHOO_PATTERN).respond(json=_yahoo_series([100.0, 110.0, 120.0]))

    result = await _skill().run(kind="stock", symbol="aapl", days=30)

    assert "Apple Inc. (AAPL)" in result
    assert "latest 120.00" in result
    assert "change +20.00%" in result


@respx.mock
async def test_holidays_are_gaps_in_the_line_not_zeros(turn):
    """Yahoo pads non-trading days with null; charting them as 0 would lie."""
    respx.get(YAHOO_PATTERN).respond(json=_yahoo_series([100.0, None, 120.0]))

    result = await _skill().run(kind="stock", symbol="AAPL", days=30)

    assert "low 100.00" in result
    assert "2 sessions" in result


@respx.mock
async def test_an_unknown_ticker_reports_yahoos_own_reason(turn):
    respx.get(YAHOO_PATTERN).respond(
        json={"chart": {"result": [], "error": {"description": "No data found for NOPE"}}}
    )

    with pytest.raises(SkillHTTPError, match="No data found for NOPE"):
        await _skill().run(kind="stock", symbol="NOPE", days=30)


@respx.mock
async def test_long_stock_ranges_switch_to_weekly_bars(turn):
    """A year of daily bars in a chat-sized image is mush."""
    route = respx.get(YAHOO_PATTERN).respond(json=_yahoo_series([1.0, 2.0]))

    await _skill().run(kind="stock", symbol="AAPL", days=365)

    assert route.calls.last.request.url.params["interval"] == "1wk"


# -- Crypto -------------------------------------------------------------------------------


@respx.mock
async def test_crypto_history_is_flipped_into_chronological_order(turn):
    respx.get(UPBIT_DAYS).respond(json=_upbit_candles([100.0, 200.0, 300.0]))

    result = await _skill().run(kind="crypto", symbol="btc", days=3)

    assert "BTC/KRW" in result
    assert "first 100.00" in result
    assert "latest 300.00" in result


@respx.mock
async def test_a_market_prefix_is_accepted_as_well_as_a_bare_coin(turn):
    route = respx.get(UPBIT_DAYS).respond(json=_upbit_candles([1.0, 2.0]))

    await _skill().run(kind="crypto", symbol="KRW-ETH", days=2)

    assert route.calls.last.request.url.params["market"] == "KRW-ETH"


@respx.mock
async def test_ranges_past_upbits_limit_use_weekly_candles(turn):
    """Upbit caps count at 200, so a year is asked for in weeks."""
    route = respx.get(UPBIT_WEEKS).respond(json=_upbit_candles([1.0, 2.0]))

    await _skill().run(kind="crypto", symbol="BTC", days=350)

    assert route.calls.last.request.url.params["count"] == "50"


@respx.mock
async def test_an_unknown_market_is_reported(turn):
    respx.get(UPBIT_DAYS).respond(json=[])

    with pytest.raises(SkillHTTPError, match="no KRW-NOPE market"):
        await _skill().run(kind="crypto", symbol="NOPE", days=5)


# -- Degrading gracefully ------------------------------------------------------------------


async def test_an_unknown_kind_says_what_the_options_are():
    with pytest.raises(SkillHTTPError, match="exchange, stock or crypto"):
        await _skill().run(kind="bonds", symbol="x")


@respx.mock
async def test_a_broken_renderer_still_answers_with_the_numbers(turn, monkeypatch):
    """The chart is the bonus; the numbers are the answer."""

    def boom(*args, **kwargs):
        raise RuntimeError("no font, no canvas, no luck")

    monkeypatch.setattr("chord.skills.price_history.render_line_chart", boom)
    respx.get(FX_PATTERN).respond(json=_fx_series())

    result = await _skill().run(kind="exchange", symbol="USD/KRW", days=5)

    assert "latest 1,384.00" in result
    assert "could not be attached" in result
    assert collected() == []


@respx.mock
async def test_outside_a_chat_turn_the_summary_says_so():
    """No collection is open, so promising an image would be a lie."""
    respx.get(FX_PATTERN).respond(json=_fx_series())

    result = await _skill().run(kind="exchange", symbol="USD/KRW", days=5)

    assert "could not be attached" in result
