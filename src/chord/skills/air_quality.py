"""Air-quality skill (fine dust / 미세먼지).

Provider chain per request:

1. AirKorea (에어코리아) - official Korean station data when
   ``AIRKOREA_API_KEY`` is set and the city is in Korea; the province
   (시도) name comes from geocoding's admin1 field.
2. Open-Meteo air quality (CAMS model) - key-less fallback covering
   the whole world.

Both normalize into :class:`AirReport`, rendered once so output looks
identical regardless of source. Particulate levels are labeled with
the Korean Ministry of Environment daily bands:

    PM10  : good <=30 | moderate <=80 | poor <=150 | very poor >150
    PM2.5 : good <=15 | moderate <=35 | poor <=75  | very poor >75
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from chord.config import Settings
from chord.skills._geo import geocode
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIRKOREA_URL = "https://apis.data.go.kr/B552584/CtprvnRltmMesureDnsty/getCtprvnRltmMesureDnsty"

#: English admin1 names from geocoding -> Korean province (시도) names.
AIRKOREA_SIDO: list[tuple[str, str]] = [
    ("seoul", "서울"),
    ("busan", "부산"),
    ("daegu", "대구"),
    ("incheon", "인천"),
    ("gwangju", "광주"),
    ("daejeon", "대전"),
    ("ulsan", "울산"),
    ("sejong", "세종"),
    ("gyeonggi", "경기"),
    ("gangwon", "강원"),
    ("chungcheongbuk", "충북"),
    ("chungcheongnam", "충남"),
    ("jeollabuk", "전북"),
    ("jeonbuk", "전북"),
    ("jeollanam", "전남"),
    ("jeonnam", "전남"),
    ("gyeongsangbuk", "경북"),
    ("gyeongbuk", "경북"),
    ("gyeongsangnam", "경남"),
    ("gyeongnam", "경남"),
    ("jeju", "제주"),
]

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


@dataclass(frozen=True)
class AirReport:
    """Provider-independent air-quality snapshot."""

    place: str
    pm10_ugm3: float | None
    pm25_ugm3: float | None
    us_aqi: int | None
    source: str


def render_report(report: AirReport) -> str:
    """Render one clean multi-line block."""
    lines = [f"Air quality in {report.place}  [via {report.source}]"]

    def particulate(label: str, value: float | None, grader) -> str:
        if value is None:
            return f"- {label:<6}: n/a"
        return f"- {label:<6}: {int(value)} ug/m3 ({grader(float(value))})"

    lines.append(particulate("PM10", report.pm10_ugm3, grade_pm10))
    lines.append(particulate("PM2.5", report.pm25_ugm3, grade_pm25))
    if report.us_aqi is not None:
        lines.append(f"- US AQI : {report.us_aqi}")
    return "\n".join(lines)


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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, city: str) -> str:
        location = await geocode(city)
        report = await self._fetch_report(location)
        return render_report(report)

    async def _fetch_report(self, location: dict[str, Any]) -> AirReport:
        place = f"{location['name']}, {location['country']}".strip(", ")
        latitude, longitude = location["latitude"], location["longitude"]

        key = self._settings.airkorea_api_key
        if key and location["country"] == "South Korea":
            sido = resolve_sido(location.get("admin1", ""))
            if sido:
                try:
                    return await fetch_airkorea(key, place, sido)
                except SkillHTTPError:
                    pass  # fall through to the worldwide source

        return await fetch_open_meteo(place, latitude, longitude)


# -- Providers -----------------------------------------------------------------


async def fetch_open_meteo(place: str, latitude: float, longitude: float) -> AirReport:
    """Key-less worldwide provider (CAMS model data)."""
    data = await get_json(
        AIR_QUALITY_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(["pm10", "pm2_5", "us_aqi"]),
        },
    )
    current = data.get("current")
    if current is None:
        raise SkillHTTPError("Air-quality API returned no current data.")
    return AirReport(
        place=place,
        pm10_ugm3=_num(current.get("pm10")),
        pm25_ugm3=_num(current.get("pm2_5")),
        us_aqi=current.get("us_aqi"),
        source="Open-Meteo",
    )


async def fetch_airkorea(api_key: str, place: str, sido_name: str) -> AirReport:
    """Official Korean provider: province-level real-time measurements."""
    data = await get_json(
        AIRKOREA_URL,
        params={
            "serviceKey": api_key,
            "returnType": "json",
            "numOfRows": 100,
            "pageNo": 1,
            "sidoName": sido_name,
            "ver": "1.3",
        },
    )
    items = ((data.get("response") or {}).get("body") or {}).get("items") or []

    def pick(field: str) -> float | None:
        for item in items:
            raw = item.get(field)
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return None

    # Station rows carry measurement flags (-1 = no data); skip them.
    pm10 = pick("pm10Value")
    pm25 = pick("pm25Value")
    if pm10 is None and pm25 is None:
        raise SkillHTTPError("AirKorea returned no usable measurements.")

    return AirReport(
        place=place,
        pm10_ugm3=pm10,
        pm25_ugm3=pm25,
        us_aqi=None,
        source="AirKorea 에어코리아",
    )


def resolve_sido(admin1: str) -> str:
    """Map an English admin1 name to a Korean province name.

    Matching is prefix-based because Open-Meteo returns forms like
    'Sejong Special Self-Governing City' or 'Gyeonggi-do'.
    """
    normalized = admin1.strip().lower()
    for prefix, sido in AIRKOREA_SIDO:
        if normalized.startswith(prefix):
            return sido
    return ""


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
