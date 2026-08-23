"""Air-quality skill (fine dust / 미세먼지) via Open-Meteo (key-less).

Reports PM10 and PM2.5 concentrations plus the US AQI, and labels each
particulate level using the Korean Ministry of Environment daily bands,
which Korean users expect for "미세먼지" questions:

    PM10  : good <=30 | moderate <=80 | poor <=150 | very poor >150
    PM2.5 : good <=15 | moderate <=35 | poor <=75  | very poor >75

An optional AirKorea key could later add station-level accuracy; the
free CAMS-based data below already covers the whole world.
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills._geo import geocode
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

#: (upper bound inclusive, label) pairs, checked in order.
PM10_BANDS: list[tuple[float, str]] = [
    (30, "good"),
    (80, "moderate"),
    (150, "poor"),
]
PM25_BANDS: list[tuple[float, str]] = [
    (15, "good"),
    (35, "moderate"),
    (75, "poor"),
]


def grade_pm10(value: float) -> str:
    """Label a PM10 concentration using Korean daily standards."""
    return _grade(value, PM10_BANDS)


def grade_pm25(value: float) -> str:
    """Label a PM2.5 concentration using Korean daily standards."""
    return _grade(value, PM25_BANDS)


def _grade(value: float, bands: list[tuple[float, str]]) -> str:
    for upper_bound, label in bands:
        if value <= upper_bound:
            return label
    return "very poor"


class AirQualitySkill(Skill):
    name = "get_air_quality"
    description = (
        "Get current air quality for a city: fine dust PM10, ultra-fine "
        "dust PM2.5 (in micrograms per cubic meter) and US AQI."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Seoul'.",
            }
        },
        "required": ["city"],
    }

    async def run(self, city: str) -> str:
        location = await geocode(city)
        data = await get_json(
            AIR_QUALITY_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": ",".join(["pm10", "pm2_5", "us_aqi"]),
            },
        )
        current = data.get("current")
        if current is None:
            raise SkillHTTPError("Air-quality API returned no current data.")

        pm10 = current.get("pm10")
        pm25 = current.get("pm2_5")
        us_aqi = current.get("us_aqi")

        parts = [f"Air quality in {location['name']}, {location['country']}:"]

        if pm10 is not None:
            parts.append(f"PM10 {pm10} ug/m3 ({grade_pm10(float(pm10))})")
        else:
            parts.append("PM10 n/a")
        if pm25 is not None:
            parts.append(f"PM2.5 {pm25} ug/m3 ({grade_pm25(float(pm25))})")
        else:
            parts.append("PM2.5 n/a")
        if us_aqi is not None:
            parts.append(f"US AQI {us_aqi}")

        return " ".join(parts) + "."
