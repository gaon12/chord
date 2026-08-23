"""Map and navigation skills.

Two tools with automatic provider selection:

* ``find_places``    - keyword place search.
  Korea + ``KAKAO_REST_API_KEY`` -> Kakao Local keyword search;
  otherwise OpenStreetMap Nominatim (key-less, worldwide).

* ``get_directions`` - driving route between two places.
  Both endpoints inside Korea + Kakao key -> Kakao Navi;
  otherwise the public OSRM demo server (key-less, worldwide).

Both tools geocode free-text place names first (shared Open-Meteo
geocoder), so callers never have to deal with raw coordinates.
"""

from __future__ import annotations

from typing import ClassVar

from chord.config import Settings
from chord.skills._geo import geocode
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_NAVI_URL = "https://apis-navi.kakaomobility.com/v1/directions"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"


def is_korea(location: dict) -> bool:
    return location.get("country") == "South Korea"


def format_duration(seconds: float | int | None) -> str:
    """Human duration like '1 h 25 min' / '35 min'."""
    if seconds is None:
        return "?"
    minutes = round(int(seconds) / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min"


def format_distance(meters: float | int | None) -> str:
    """Human distance like '850 m' / '27.8 km'."""
    if meters is None:
        return "?"
    if meters < 1000:
        return f"{int(meters)} m"
    return f"{meters / 1000:.1f} km"


# ---------------------------------------------------------------------------
# find_places
# ---------------------------------------------------------------------------


async def search_kakao(api_key: str, query: str) -> list[dict]:
    data = await get_json(
        KAKAO_KEYWORD_URL,
        params={"query": query, "size": 5},
        headers={"Authorization": f"KakaoAK {api_key}"},
    )
    documents = data.get("documents") or []
    return [
        {
            "title": doc.get("place_name", ""),
            "address": doc.get("road_address_name") or doc.get("address_name") or "",
            "phone": doc.get("phone", ""),
            "url": doc.get("place_url", ""),
        }
        for doc in documents
    ]


async def search_nominatim(query: str) -> list[dict]:
    data = await get_json(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "limit": 5},
    )
    results = data if isinstance(data, list) else []
    return [
        {
            "title": item.get("name") or (item.get("display_name", "").split(",")[0]),
            "address": item.get("display_name", ""),
            "phone": "",
            "url": "",
        }
        for item in results
    ]


class FindPlacesSkill(Skill):
    name = "find_places"
    description = (
        "Search places by keyword (e.g. 'coffee shop Hongdae', 'Gyeongbokgung'). "
        "Returns names, addresses and links."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, ideally including an area.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, query: str) -> str:
        if not query.strip():
            raise SkillHTTPError("Please provide a search query.")

        # Decide the provider by where the query seems to point; a quick
        # geocode of the query itself tells us the country context.
        location = await geocode(query)
        places: list[dict] = []
        source = ""
        if self._settings.kakao_rest_api_key and is_korea(location):
            try:
                places = await search_kakao(self._settings.kakao_rest_api_key, query)
                source = "Kakao"
            except SkillHTTPError:
                pass
        if not places:
            places = await search_nominatim(query)
            source = "OpenStreetMap"

        if not places:
            raise SkillHTTPError(f"No places found for '{query}'.")

        lines = [f"Places for '{query}'  [via {source}]"]
        for i, place in enumerate(places[:5], start=1):
            entry = f"{i}. {place['title']} - {place['address']}"
            if place["phone"]:
                entry += f" | tel {place['phone']}"
            if place["url"]:
                entry += f" | {place['url']}"
            lines.append(entry)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# get_directions
# ---------------------------------------------------------------------------


def osrm_coords(origin: dict, destination: dict) -> str:
    """OSRM wants 'lon,lat;lon,lat' in that (reversed) order."""
    return (
        f"{origin['longitude']},{origin['latitude']};"
        f"{destination['longitude']},{destination['latitude']}"
    )


async def route_osrm(origin: dict, destination: dict) -> tuple[str, str]:
    """Key-less driving route via the OSRM demo server."""
    data = await get_json(OSRM_URL.format(coords=osrm_coords(origin, destination)))
    routes = data.get("routes") or []
    if not routes:
        raise SkillHTTPError("OSRM found no driving route between those places.")
    return format_distance(routes[0].get("distance")), format_duration(routes[0].get("duration"))


async def route_kakao(api_key: str, origin: dict, destination: dict) -> tuple[str, str]:
    """Kakao Navi driving route for Korean endpoints (WGS84 coordinates)."""
    data = await get_json(
        KAKAO_NAVI_URL,
        params={
            "origin": f"{origin['longitude']},{origin['latitude']}",
            "destination": f"{destination['longitude']},{destination['latitude']}",
        },
        headers={"Authorization": f"KakaoAK {api_key}"},
    )
    routes = data.get("routes") or []
    if not routes:
        raise SkillHTTPError("Kakao Navi found no route between those places.")
    summary = routes[0].get("summary") or {}
    # Kakao reports distance in meters and duration in milliseconds.
    distance = format_distance(summary.get("distance"))
    duration_ms = summary.get("duration")
    seconds = None if duration_ms is None else float(duration_ms) / 1000
    return distance, format_duration(seconds)


class GetDirectionsSkill(Skill):
    name = "get_directions"
    description = (
        "Get a driving route between two places: distance and estimated "
        "time. Provide origin and destination as place names."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Starting place, e.g. 'Incheon'.",
            },
            "destination": {
                "type": "string",
                "description": "Destination place, e.g. 'Seoul'.",
            },
        },
        "required": ["origin", "destination"],
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, origin: str, destination: str) -> str:
        origin_loc = await geocode(origin)
        destination_loc = await geocode(destination)

        source = ""
        try:
            if self._settings.kakao_rest_api_key and (
                is_korea(origin_loc) and is_korea(destination_loc)
            ):
                distance, duration = await route_kakao(
                    self._settings.kakao_rest_api_key, origin_loc, destination_loc
                )
                source = "Kakao Navi"
            else:
                raise SkillHTTPError("outside Korea")
        except SkillHTTPError:
            distance, duration = await route_osrm(origin_loc, destination_loc)
            source = "OSRM"

        return (
            f"Route {origin_loc['name']} -> {destination_loc['name']}: "
            f"{distance}, about {duration} by car.  [via {source}]"
        )
