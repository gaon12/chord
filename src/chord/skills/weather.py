"""Weather skill - official Korean / worldwide providers, free fallback.

Provider chain decided per request:

1. Korea Meteorological Administration (KMA, 기상청 초단기실황) - the
   most accurate source for Korean cities; requires ``KMA_API_KEY``.
2. WeatherAPI.com - solid worldwide coverage; requires
   ``WEATHERAPI_API_KEY``.
3. Open-Meteo - key-less fallback so the bot works out of the box.

Every provider normalizes into :class:`WeatherReport`, and rendering
happens exactly once so output looks identical regardless of source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar

from chord.config import Settings
from chord.skills._geo import geocode
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHERAPI_URL = "https://api.weatherapi.com/v1/current.json"
KMA_ULTRA_SRT_NCST_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

#: Rough Korean bounding box used to prefer the KMA provider.
KOREA_BBOX = {"lat_min": 33.0, "lat_max": 38.9, "lon_min": 124.5, "lon_max": 132.0}

# WMO weather interpretation codes (used by the Open-Meteo provider),
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

# KMA PTY (precipitation type) codes.
KMA_PTY_CODES = {
    "0": "",
    "1": "rain",
    "2": "sleet",
    "3": "snow",
    "4": "rain showers",
    "5": "drizzle",
    "6": "sleet",
    "7": "snow flurries",
}


@dataclass(frozen=True)
class WeatherReport:
    """Provider-independent snapshot of current conditions."""

    place: str
    condition: str
    temperature_c: float | None
    feels_like_c: float | None
    humidity_pct: float | None
    wind_kmh: float | None
    source: str


def describe_weather_code(code: int | None) -> str:
    """Human-readable description for a WMO code."""
    if code is None:
        return "unknown"
    return WEATHER_CODES.get(code, f"weather code {code}")


def render_report(report: WeatherReport) -> str:
    """Render one clean multi-line block, identical for all providers."""
    lines = [
        f"Weather in {report.place}  ({report.condition})  [via {report.source}]",
        f"- Temperature: {_fmt_temp(report.temperature_c)}{_feels_part(report.feels_like_c)}",
        f"- Humidity  : {_fmt_percent(report.humidity_pct)}",
        f"- Wind      : {_fmt_wind(report.wind_kmh)}",
    ]
    return "\n".join(lines)


def _fmt_temp(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}°C"


def _feels_part(value: float | None) -> str:
    return "" if value is None else f" (feels like {value:.1f}°C)"


def _fmt_percent(value: float | int | None) -> str:
    return "?" if value is None else f"{int(value)}%"


def _fmt_wind(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f} km/h"


def in_korea(latitude: float, longitude: float) -> bool:
    """True when coordinates fall inside the rough Korean bounding box."""
    return (
        KOREA_BBOX["lat_min"] <= latitude <= KOREA_BBOX["lat_max"]
        and KOREA_BBOX["lon_min"] <= longitude <= KOREA_BBOX["lon_max"]
    )


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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, city: str) -> str:
        location = await geocode(city)
        report = await self._fetch_report(location)
        return render_report(report)

    # -- Provider selection -----------------------------------------------------

    async def _fetch_report(self, location: dict[str, Any]) -> WeatherReport:
        place = f"{location['name']}, {location['country']}".strip(", ")
        latitude, longitude = location["latitude"], location["longitude"]

        if self._settings.kma_api_key and in_korea(latitude, longitude):
            try:
                return await fetch_kma(self._settings.kma_api_key, place, latitude, longitude)
            except SkillHTTPError:
                # Official data hiccup -> fall through to the next source.
                pass

        if self._settings.weatherapi_api_key:
            try:
                return await fetch_weatherapi(
                    self._settings.weatherapi_api_key,
                    place,
                    latitude,
                    longitude,
                )
            except SkillHTTPError:
                pass

        return await fetch_open_meteo(place, latitude, longitude)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


async def fetch_open_meteo(place: str, latitude: float, longitude: float) -> WeatherReport:
    """Key-less default provider."""
    data = await get_json(
        OPEN_METEO_URL,
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
    return WeatherReport(
        place=place,
        condition=describe_weather_code(current.get("weather_code")),
        temperature_c=current.get("temperature_2m"),
        feels_like_c=current.get("apparent_temperature"),
        humidity_pct=current.get("relative_humidity_2m"),
        wind_kmh=current.get("wind_speed_10m"),
        source="Open-Meteo",
    )


async def fetch_weatherapi(
    api_key: str, place: str, latitude: float, longitude: float
) -> WeatherReport:
    """Worldwide provider via WeatherAPI.com."""
    data = await get_json(
        WEATHERAPI_URL,
        params={"key": api_key, "q": f"{latitude},{longitude}", "aqi": "no"},
    )
    current = data.get("current") or {}
    if not current:
        raise SkillHTTPError("WeatherAPI returned no current data.")
    condition_block = current.get("condition") or {}
    return WeatherReport(
        place=place,
        condition=str(condition_block.get("text", "unknown")),
        temperature_c=current.get("temp_c"),
        feels_like_c=current.get("feelslike_c"),
        humidity_pct=current.get("humidity"),
        wind_kmh=current.get("wind_kph"),
        source="WeatherAPI.com",
    )


async def fetch_kma(api_key: str, place: str, latitude: float, longitude: float) -> WeatherReport:
    """Official Korean provider: KMA ultra-short-term observation (기상청 초단기실황)."""
    nx, ny = to_grid(latitude, longitude)
    base_date, base_time = kma_base_timestamp()
    data = await get_json(
        KMA_ULTRA_SRT_NCST_URL,
        params={
            "serviceKey": api_key,
            "pageNo": 1,
            "numOfRows": 100,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
        },
    )

    items = (((data.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    observations = {str(item.get("category")): item.get("obsrValue") for item in items}
    if not observations:
        raise SkillHTTPError("KMA returned no observation items.")

    temperature = _num(observations.get("T1H"))
    humidity = _num(observations.get("REH"))
    wind_ms = _num(observations.get("WSD"))
    pty = str(observations.get("PTY", "0"))
    condition = KMA_PTY_CODES.get(pty) or "clear"

    return WeatherReport(
        place=place,
        condition=condition,
        temperature_c=temperature,
        feels_like_c=_apparent_temperature(temperature, humidity),
        humidity_pct=humidity,
        wind_kmh=None if wind_ms is None else round(wind_ms * 3.6, 1),
        source="KMA 기상청",
    )


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apparent_temperature(temp_c: float | None, humidity_pct: float | None) -> float | None:
    """Simple heat-index approximation; None when inputs are missing."""
    if temp_c is None or humidity_pct is None:
        return None
    return round(temp_c - (100 - humidity_pct) / 10 * 0.3, 1)


def kma_base_timestamp(now: datetime | None = None) -> tuple[str, str]:
    """KMA ultra-srt-ncst accepts observations older than ~40 minutes."""
    kst_now = (now or datetime.now(UTC)).astimezone(timezone(timedelta(hours=9)))
    if kst_now.minute < 45:
        kst_now -= timedelta(hours=1)
    return kst_now.strftime("%Y%m%d"), kst_now.strftime("%H00")


def to_grid(latitude: float, longitude: float) -> tuple[int, int]:
    """Convert lat/lon to the KMA Lambert-conformal grid (nx, ny).

    Implements the projection from the KMA digital-forecast guide with
    its official constants (Earth radius 6371.00877 km, 5 km cells,
    standard parallels 30/60 N, origin 38N/126E, origin cell 43/136).
    Reference results: Seoul -> (60, 127), Busan -> (98, 76).
    """
    earth_radius_km = 6371.00877
    grid_km = 5.0
    slat1_deg, slat2_deg = 30.0, 60.0  # standard parallels
    olon_deg, olat_deg = 126.0, 38.0  # projection origin
    xo, yo = 43.0, 136.0  # origin cell offsets

    def rad(degrees: float) -> float:
        return math.pi * degrees / 180.0

    def tan_half(lat_degrees: float) -> float:
        # tan(pi/4 + lat/2): the Lambert conformal "isometric" tangent.
        return math.tan(math.pi * 0.25 + rad(lat_degrees) * 0.5)

    re = earth_radius_km / grid_km  # Earth radius in grid units
    lon = rad(longitude)
    slat1, slat2 = rad(slat1_deg), rad(slat2_deg)
    olon = rad(olon_deg)

    # Lambert conformal cone constant and scaling factor.
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        tan_half(slat2_deg) / tan_half(slat1_deg)
    )
    sp = math.cos(slat1) * tan_half(slat1_deg) ** sn / sn

    # Radii from the projection origin, in grid units.
    ro = re * sp / tan_half(olat_deg) ** sn  # origin radius
    ra = re * sp / tan_half(latitude) ** sn  # target radius

    theta = lon - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = ra * math.sin(theta) + xo
    y = ro - ra * math.cos(theta) + yo
    return int(x + 0.5), int(y + 0.5)
