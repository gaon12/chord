"""Tests for the air-quality skill (Open-Meteo AQ API, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.air_quality import AirQualitySkill, grade_pm10, grade_pm25

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _geocode_ok():
    return {
        "results": [
            {
                "name": "Seoul",
                "country": "South Korea",
                "latitude": 37.566,
                "longitude": 126.9784,
            }
        ]
    }


def _aq_response(pm10=41, pm2_5=19, us_aqi=52):
    return {"current": {"pm10": pm10, "pm2_5": pm2_5, "us_aqi": us_aqi}}


@respx.mock
async def test_air_quality_happy_path():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AQ_URL).respond(json=_aq_response())

    result = await AirQualitySkill().run(city="Seoul")

    assert "Seoul, South Korea" in result
    assert "PM10 41" in result and "(moderate)" in result
    assert "PM2.5 19" in result and "(moderate)" in result
    assert "US AQI 52" in result


@respx.mock
async def test_good_air_quality_labels():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AQ_URL).respond(json=_aq_response(pm10=20, pm2_5=10))

    result = await AirQualitySkill().run(city="Seoul")

    assert "PM10 20" in result and "(good)" in result
    assert "PM2.5 10" in result and "(good)" in result


@respx.mock
async def test_missing_values_fall_back_to_na():
    respx.get(GEO_URL).respond(json=_geocode_ok())
    respx.get(AQ_URL).respond(json={"current": {}})

    result = await AirQualitySkill().run(city="Seoul")

    assert "PM10 n/a" in result
    assert "PM2.5 n/a" in result


@respx.mock
async def test_unknown_city_raises():
    respx.get(GEO_URL).respond(json={"results": None})

    with pytest.raises(SkillHTTPError, match="Could not find a city named"):
        await AirQualitySkill().run(city="Atlantis")


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
