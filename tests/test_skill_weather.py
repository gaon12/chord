"""Tests for the weather skill - all HTTP traffic is mocked with respx."""

from __future__ import annotations

import pytest
import respx

from chord.skills._http import SkillHTTPError
from chord.skills.weather import WeatherSkill, describe_weather_code

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _geocode_response():
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


def _forecast_response():
    return {
        "current": {
            "temperature_2m": 22.4,
            "apparent_temperature": 23.1,
            "relative_humidity_2m": 60,
            "wind_speed_10m": 8.2,
            "weather_code": 2,
        }
    }


@respx.mock
async def test_weather_happy_path():
    respx.get(GEO_URL).respond(json=_geocode_response())
    respx.get(FORECAST_URL).respond(json=_forecast_response())

    result = await WeatherSkill().run(city="Seoul")

    assert "Seoul, South Korea" in result
    assert "22.4" in result and "23.1" in result
    assert "partly cloudy" in result
    assert "60%" in result


def test_weather_tool_definition_shape():
    tool = WeatherSkill().to_openai_tool()
    function = tool["function"]
    assert function["name"] == "get_weather"
    assert function["parameters"]["required"] == ["city"]


@respx.mock
async def test_unknown_city_raises_readable_error():
    respx.get(GEO_URL).respond(json={"results": None})

    with pytest.raises(SkillHTTPError, match="Could not find a city named"):
        await WeatherSkill().run(city="Nowhereville")


@respx.mock
async def test_forecast_api_error_raises_readable_error():
    respx.get(GEO_URL).respond(json=_geocode_response())
    respx.get(FORECAST_URL).respond(status_code=500)

    # _http wraps HTTP failures in SkillHTTPError; the registry turns
    # that into readable text for the LLM.
    with pytest.raises(SkillHTTPError, match="answered HTTP 500"):
        await WeatherSkill().run(city="Seoul")


# -- describe_weather_code --------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, "clear sky"),
        (3, "overcast"),
        (95, "thunderstorm"),
        (123, "weather code 123"),  # unmapped codes degrade gracefully
        (None, "unknown"),
    ],
)
def test_describe_weather_code(code, expected):
    assert describe_weather_code(code) == expected
