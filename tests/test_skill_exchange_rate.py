"""Tests for the exchange-rate skill (Frankfurter API, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.exchange_rate import ExchangeRateSkill, format_amount

FX_URL = "https://api.frankfurter.dev/v1/latest"


def _fx_response():
    return {
        "amount": 1.0,
        "base": "USD",
        "date": "2026-08-21",
        "rates": {"KRW": 1384.23},
    }


@respx.mock
async def test_exchange_rate_happy_path():
    respx.get(FX_URL).respond(json=_fx_response())

    result = await ExchangeRateSkill().run(base="usd", target="krw", amount=1.0)

    assert "1 USD = 1,384.23 KRW" in result
    assert "2026-08-21" in result


@respx.mock
async def test_exchange_rate_multiplies_amount():
    respx.get(FX_URL).respond(json=_fx_response())

    result = await ExchangeRateSkill().run(base="USD", target="KRW", amount=100)

    assert "100 USD = 138,423 KRW" in result


@respx.mock
async def test_unknown_currency_pair_reports_error():
    respx.get(FX_URL).respond(json={"rates": {}})

    with pytest.raises(SkillHTTPError, match="No rate available"):
        await ExchangeRateSkill().run(base="USD", target="XXX")


async def test_invalid_currency_code_fails_fast_without_request():
    with pytest.raises(SkillHTTPError, match="not a valid"):
        await ExchangeRateSkill().run(base="DOLLAR", target="KRW")


def test_format_amount():
    assert format_amount(1.0) == "1"
    assert format_amount(100) == "100"
    assert format_amount(1384.23) == "1,384.23"
    assert format_amount(1234567.891) == "1,234,567.89"
