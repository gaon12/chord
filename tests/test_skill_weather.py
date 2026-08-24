"""Tests for the weather skill - providers, selection and rendering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
import respx

import quota_helpers
from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.weather import (
    OPEN_METEO_URL,
    WEATHERAPI_URL,
    WeatherSkill,
    describe_weather_code,
    in_korea,
    kma_base_timestamp,
    render_report,
    to_grid,
)

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
KMA_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

DEG = "\u00b0"  # degree sign, escaped so editor encodings can never break it


def _settings(**keys) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


def _geocode_ok(name="Seoul", country="South Korea", lat=37.566, lon=126.9784):
    return {"results": [{"name": name, "country": country, "latitude": lat, "longitude": lon}]}


def _open_meteo_response():
    return {
        "current": {
            "temperature_2m": 22.4,
            "apparent_temperature": 23.1,
            "relative_humidity_2m": 60,
            "wind_speed_10m": 8.2,
            "weather_code": 2,
        }
    }


def _weatherapi_response():
    return {
        "current": {
            "temp_c": 21.0,
            "feelslike_c": 22.5,
            "humidity": 55,
            "wind_kph": 12.3,
            "condition": {"text": "Partly cloudy"},
        }
    }


def _kma_response():
    items = [
        {"category": cat, "obsrValue": value}
        for cat, value in {
            "T1H": "18.5",
            "REH": "65",
            "WSD": "2.1",
            "PTY": "0",
        }.items()
    ]
    return {"response": {"body": {"items": {"item": items}}}}


# -- Provider selection -----------------------------------------------------------


@respx.mock
async def test_open_meteo_used_when_no_keys():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(OPEN_METEO_URL).respond(json=_open_meteo_response())

    result = await WeatherSkill(_settings()).run(city="Seoul")

    assert "[via Open-Meteo]" in result
    assert f"22.4{DEG}C" in result
    assert f"(feels like 23.1{DEG}C)" in result


@respx.mock
async def test_kma_preferred_for_korean_city_when_key_present():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(KMA_URL).respond(json=_kma_response())

    result = await WeatherSkill(_settings(kma_api_key="secret")).run(city="Seoul")

    assert "via KMA" in result
    assert f"18.5{DEG}C" in result
    assert "7.6 km/h" in result  # m/s converted to km/h


@respx.mock
async def test_weatherapi_for_worldwide_city():
    respx.get(GEO_URL).respond(
        json=_geocode_ok(name="Paris", country="France", lat=48.8566, lon=2.3522)
    )
    respx.get(WEATHERAPI_URL).respond(json=_weatherapi_response())

    result = await WeatherSkill(_settings(weatherapi_api_key="wkey")).run(city="Paris")

    assert "[via WeatherAPI.com]" in result
    assert "Partly cloudy" in result
    assert f"21.0{DEG}C" in result


@respx.mock
async def test_kma_failure_falls_back_to_weatherapi():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(KMA_URL).respond(status_code=500)
    respx.get(WEATHERAPI_URL).respond(json=_weatherapi_response())

    settings = _settings(kma_api_key="k", weatherapi_api_key="w")
    result = await WeatherSkill(settings).run(city="Seoul")

    assert "[via WeatherAPI.com]" in result


@respx.mock
async def test_unknown_city_raises_readable_error():
    respx.get(GEO_URL).respond(json={"results": None})

    with pytest.raises(SkillHTTPError, match="Could not find a city named"):
        await WeatherSkill(_settings()).run(city="Nowhereville")


# -- Rendering ----------------------------------------------------------------------


def test_render_report_aligns_fields():
    from chord.skills.weather import WeatherReport

    report = WeatherReport(
        place="Seoul, South Korea",
        condition="clear sky",
        temperature_c=22.44,
        feels_like_c=23.11,
        humidity_pct=60,
        wind_kmh=8.25,
        source="Open-Meteo",
    )
    lines = render_report(report).splitlines()
    assert lines[0] == "Weather in Seoul, South Korea  (clear sky)  [via Open-Meteo]"
    assert lines[1] == f"- Temperature: 22.4{DEG}C (feels like 23.1{DEG}C)"
    assert lines[2] == "- Humidity  : 60%"
    assert lines[3] == "- Wind      : 8.2 km/h"


def test_render_report_handles_missing_values():
    from chord.skills.weather import WeatherReport

    report = WeatherReport("X", "?", None, None, None, None, "src")
    text = render_report(report)
    assert "?" in text
    assert "(" not in text.splitlines()[1]


# -- Helpers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0, "clear sky"), (95, "thunderstorm"), (123, "weather code 123"), (None, "unknown")],
)
def test_describe_weather_code(code, expected):
    assert describe_weather_code(code) == expected


def test_in_korea_bbox():
    assert in_korea(37.566, 126.978)  # Seoul
    assert in_korea(33.4996, 126.5312)  # Jeju
    assert not in_korea(35.6762, 139.6503)  # Tokyo
    assert not in_korea(-33.86, 151.2)  # Sydney


def test_to_grid_known_reference_points():
    # Official reference conversions for well-known cities.
    assert to_grid(37.5665, 126.9780) == (60, 127)  # Seoul
    assert to_grid(35.1796, 129.0756) == (98, 76)  # Busan


def test_kma_base_timestamp_rounds_to_last_valid_observation():
    # 10:50 KST -> base time 1000 same day.
    now = datetime(2026, 8, 23, 1, 50, tzinfo=UTC)  # 10:50 KST
    date, time = kma_base_timestamp(now)
    assert date == "20260823" and time == "1000"

    # 10:20 KST -> previous hour (0900).
    now = datetime(2026, 8, 23, 1, 20, tzinfo=UTC)
    date, time = kma_base_timestamp(now)
    assert date == "20260823" and time == "0900"

    # 00:20 KST -> previous day's 2300.
    now = datetime(2026, 8, 22, 15, 20, tzinfo=UTC)
    date, time = kma_base_timestamp(now)
    assert date == "20260822" and time == "2300"


def test_kst_offset_math_is_consistent():
    kst = timezone(timedelta(hours=9))
    utc_now = datetime(2026, 8, 23, 15, 0, tzinfo=UTC)
    kst_now = utc_now.astimezone(kst)
    # 15:00 UTC is midnight at the start of the next KST day.
    assert (kst_now.day, kst_now.hour) == (24, 0)


@respx.mock
async def test_exhausted_weatherapi_falls_back_to_open_meteo():
    quota_helpers.prefill_quota("weatherapi", 100_000)
    respx.get(GEO_URL).respond(json=_geocode_ok())
    weatherapi_route = respx.get(WEATHERAPI_URL).respond(json=_weatherapi_response())
    respx.get(OPEN_METEO_URL).respond(json=_open_meteo_response())

    settings = _settings(weatherapi_api_key="wkey")
    result = await WeatherSkill(settings).run(city="Seoul")

    assert not weatherapi_route.called  # quota checked before any request
    assert "[via Open-Meteo]" in result
