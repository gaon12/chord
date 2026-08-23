"""Shared city-name -> coordinates lookup (Open-Meteo geocoding).

Used by any skill that needs a location: weather, air quality, flights.
"""

from __future__ import annotations

from typing import Any

from chord.skills._http import SkillHTTPError, get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


async def geocode(city: str) -> dict[str, Any]:
    """Resolve a city name into location details."""
    data = await get_json(GEOCODING_URL, params={"name": city, "count": 1})
    results = data.get("results") or []
    if not results:
        raise SkillHTTPError(f"Could not find a city named '{city}'.")
    first = results[0]
    return {
        "name": first.get("name", city),
        "country": first.get("country", ""),
        # First-level administrative division, e.g. 'Seoul' or 'Tokyo';
        # used to pick Korean providers such as AirKorea.
        "admin1": first.get("admin1", "") or "",
        "latitude": first["latitude"],
        "longitude": first["longitude"],
    }
