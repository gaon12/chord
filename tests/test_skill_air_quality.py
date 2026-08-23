"""Tests for the air-quality skill (AirKorea + Open-Meteo, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.air_quality import (
    AIRKOREA_URL,
    AirQualitySkill,
    grade_pm10,
    grade_pm25,
    resolve_sido,
)

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

DEG = ""  # no degree signs in this module; kept for symmetric naming


def _settings(**keys) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


def _geocode_ok(name="Seoul", admin1="Seoul", lat=37.566, lon=126.9784):
    return {
        "results": [
            {
                "name": name,
                "country": "South Korea",
                "admin1": admin1,
                "latitude": lat,
                "longitude": lon,
            }
        ]
    }


def _aq_response(pm10=41, pm2_5=19, us_aqi=52):
    return {"current": {"pm10": pm10, "pm2_5": pm2_5, "us_aqi": us_aqi}}


def _airkorea_response():
    return {
        "response": {
            "body": {
                "items": [
                    {
                        "stationName": "중구",
                        "pm10Value": "18",
                        "pm10Grade": "1",
                        "pm25Value": "12",
                        "pm25Grade": "1",
                    },
                    {
                        "stationName": "노원구",
                        "pm10Value": "-1",  # no data flag must be skipped
                        "pm25Value": "-1",
                    },
                ]
            }
        }
    }


# -- Provider selection -----------------------------------------------------------


@respx.mock
async def test_open_meteo_used_when_no_keys():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AQ_URL).respond(json=_aq_response())

    result = await AirQualitySkill(_settings()).run(city="Seoul")

    assert "[via Open-Meteo]" in result
    assert "PM10  : 41 ug/m3 (moderate)" in result


@respx.mock
async def test_airkorea_preferred_for_korea_when_key_present():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AIRKOREA_URL).respond(json=_airkorea_response())

    settings = _settings(airkorea_api_key="secret")
    result = await AirQualitySkill(settings).run(city="Seoul")

    assert "[via AirKorea" in result
    assert "PM10  : 18 ug/m3 (good)" in result
    assert "PM2.5 : 12 ug/m3 (good)" in result
    # US AQI is not part of the AirKorea payload.
    assert "US AQI" not in result


@respx.mock
async def test_airkorea_failure_falls_back_to_open_meteo():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AIRKOREA_URL).respond(status_code=500)
    respx.get(AQ_URL).respond(json=_aq_response())

    settings = _settings(airkorea_api_key="secret")
    result = await AirQualitySkill(settings).run(city="Seoul")

    assert "[via Open-Meteo]" in result


@respx.mock
async def test_non_korean_city_skips_airkorea():
    respx.get(GEO_URL).respond(
        json={
            "results": [
                {
                    "name": "Tokyo",
                    "country": "Japan",
                    "admin1": "Tokyo",
                    "latitude": 35.6762,
                    "longitude": 139.6503,
                }
            ]
        }
    )
    respx.get(AQ_URL).respond(json=_aq_response())

    settings = _settings(airkorea_api_key="secret")
    result = await AirQualitySkill(settings).run(city="Tokyo")

    assert "[via Open-Meteo]" in result


# -- Rendering ----------------------------------------------------------------------


@respx.mock
async def test_missing_values_fall_back_to_na():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AQ_URL).respond(json={"current": {}})

    result = await AirQualitySkill(_settings()).run(city="Seoul")

    assert "PM10  : n/a" in result
    assert "PM2.5 : n/a" in result


@respx.mock
async def test_unknown_city_raises():
    respx.get(GEO_URL).respond(json={"results": None})

    with pytest.raises(SkillHTTPError, match="Could not find a city named"):
        await AirQualitySkill(_settings()).run(city="Atlantis")


# -- Helpers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "good"),
        (30, "good"),
        (31, "moderate"),
        (80, "moderate"),
        (81, "poor"),
        (150, "poor"),
        (151, "very poor"),
    ],
)
def test_pm10_bands(value, expected):
    assert grade_pm10(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "good"),
        (15, "good"),
        (16, "moderate"),
        (35, "moderate"),
        (36, "poor"),
        (75, "poor"),
        (76, "very poor"),
    ],
)
def test_pm25_bands(value, expected):
    assert grade_pm25(value) == expected


@pytest.mark.parametrize(
    ("admin1", "expected"),
    [
        ("Seoul", "서울"),
        ("Gyeonggi-do", "경기"),
        ("Sejong Special Self-Governing City", "세종"),
        ("Jeollanam-do", "전남"),
        ("Tokyo", ""),  # unknown -> empty means 'no AirKorea lookup'
    ],
)
def test_resolve_sido(admin1, expected):
    assert resolve_sido(admin1) == expected
