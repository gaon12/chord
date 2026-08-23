"""Weather skill - current conditions via Open-Meteo (no API key needed).

Pipeline: city name -> coordinates (Open-Meteo geocoding) -> current
weather (Open-Meteo forecast API). Both endpoints are free and
key-less; an optional OpenWeather key could be layered in later by
adding another provider class - the skill structure keeps that easy.
"""

from __future__ import annotations

from typing import Any, ClassVar

from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (https://open-meteo.com/en/docs),
# shortened to the groups users actually care about.
WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def describe_weather_code(code: int | None) -> str:
    """Human-readable description for a WMO code."""
    if code is None:
        return "unknown"
    return WEATHER_CODES.get(code, f"weather code {code}")


class WeatherSkill(Skill):
    name = "get_weather"
    description = (
        "Get the current weather for a city: temperature, feels-like, "
        "humidity, wind and sky condition."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Seoul' or 'New York'.",
            }
        },
        "required": ["city"],
    }

    async def run(self, city: str) -> str:
        location = await _geocode(city)
        current = await _fetch_current(location["latitude"], location["longitude"])

        temperature = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        condition = describe_weather_code(current.get("weather_code"))

        parts = [
            f"Weather in {location['name']}, {location['country']}: {condition}.",
            f"Temperature: {temperature}°C (feels like {feels_like}°C).",
            f"Humidity: {humidity}%.",
            f"Wind speed: {wind} km/h.",
        ]
        return " ".join(parts)


async def _geocode(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates using Open-Meteo geocoding."""
    data = await get_json(GEOCODING_URL, params={"name": city, "count": 1})
    results = data.get("results") or []
    if not results:
        raise SkillHTTPError(f"Could not find a city named '{city}'.")
    first = results[0]
    return {
        "name": first.get("name", city),
        "country": first.get("country", ""),
        "latitude": first["latitude"],
        "longitude": first["longitude"],
    }


async def _fetch_current(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch the 'current' block from the Open-Meteo forecast API."""
    data = await get_json(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "weather_code",
                ]
            ),
        },
    )
    current = data.get("current")
    if not current:
        raise SkillHTTPError("Weather API returned no current data.")
    return current
