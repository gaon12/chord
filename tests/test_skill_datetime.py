"""Tests for date/time and timezone skills (no network involved)."""

from __future__ import annotations

from chord.skills.datetime_info import (
    ConvertTimezoneSkill,
    CurrentDatetimeSkill,
    parse_datetime,
    resolve_timezone,
)
from chord.skills.registry import SkillRegistry


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(CurrentDatetimeSkill())
    registry.register(ConvertTimezoneSkill())
    return registry


# -- get_current_datetime ---------------------------------------------------------


async def test_current_datetime_default_is_seoul():
    result = await _registry().execute("get_current_datetime", "{}")

    assert "Asia/Seoul" in result
    assert "UTC+09:00" in result
    # Weekday name is included for the model's convenience.
    assert any(
        day in result
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    )


async def test_current_datetime_other_zone():
    result = await _registry().execute("get_current_datetime", '{"timezone": "America/New_York"}')
    assert "America/New_York" in result
    offset = "-05:00" in result or "-04:00" in result  # EST vs EDT
    assert offset


async def test_unknown_timezone_becomes_error_text():
    result = await _registry().execute("get_current_datetime", '{"timezone": "Mars/Olympus"}')
    assert "Unknown timezone" in result


# -- convert_timezone ----------------------------------------------------------------


async def test_convert_timezone_seoul_to_new_york():
    result = await ConvertTimezoneSkill().run(
        datetime_text="2026-08-23 14:00",
        from_timezone="Asia/Seoul",
        to_timezone="America/New_York",
    )

    assert "(Asia/Seoul)" in result
    assert "(America/New_York)" in result
    # 14:00 KST on Aug 23 is 01:00 EDT (or 00:00 EST) the same day.
    assert "2026-08-" in result
    assert "01:00" in result or "00:00" in result


async def test_convert_timezone_defaults_source_to_seoul():
    result = await ConvertTimezoneSkill().run(
        datetime_text="2026-08-23 09:00",
        to_timezone="UTC",
    )

    assert "00:00" in result  # 09:00 KST == 00:00 UTC
    assert "(UTC)" in result


async def test_bad_datetime_text_returns_error_text():
    registry = SkillRegistry()
    registry.register(ConvertTimezoneSkill())

    result = await registry.execute(
        "convert_timezone",
        {"datetime_text": "tomorrow morning", "to_timezone": "UTC"},
    )
    assert "Could not parse" in result


# -- Helpers -------------------------------------------------------------------------


def test_resolve_timezone_aliases():
    assert resolve_timezone("kst").key == "Asia/Seoul"
    assert resolve_timezone("seoul").key == "Asia/Seoul"
    assert resolve_timezone("").key == "Asia/Seoul"
    assert resolve_timezone("UTC").key == "UTC"


def test_parse_datetime_accepts_multiple_formats():
    zone = resolve_timezone("Asia/Seoul")
    assert parse_datetime("2026-08-23T14:00:00", zone).hour == 14
    assert parse_datetime("2026-08-23 14:00", zone).minute == 0
    assert parse_datetime("2026.08.23 09:30:00", zone).minute == 30


def test_parse_datetime_keeps_explicit_offset():
    zone = resolve_timezone("Asia/Seoul")
    moment = parse_datetime("2026-08-23T00:00:00+00:00", zone)
    assert moment.utcoffset().total_seconds() == 9 * 3600
