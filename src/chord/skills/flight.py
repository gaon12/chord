"""Flight-info skill via OpenSky Network and adsbdb (both key-less).

Two modes:

* **callsign** - live position of one flight (OpenSky state vectors)
  enriched with airline/route data from adsbdb when available.
* **city** - how many aircraft are flying near a city right now,
  using a rough +/-3 degree box around its coordinates.

OpenSky's anonymous tier is rate-limited (~400 requests/day); that is
plenty for a chat bot and requires no account.
"""

from __future__ import annotations

from typing import Any, ClassVar

from chord.skills._geo import geocode
from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

OPENSKY_URL = "https://opensky-network.org/api/states/all"
ADSBDB_URL = "https://api.adsbdb.com/v0/callsign/{callsign}"
AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"

#: How far from the city center to search, in degrees (~330 km).
SEARCH_RADIUS_DEGREES = 3.0

#: Human labels for Aviationstack flight statuses.
AVIATIONSTACK_STATUS = {
    "scheduled": "scheduled",
    "active": "en route",
    "landed": "landed",
    "cancelled": "cancelled",
    "incident": "incident",
    "diverted": "diverted",
}

#: Indices inside an OpenSky state vector array.
IDX_ICAO24 = 0
IDX_CALLSIGN = 1
IDX_ORIGIN_COUNTRY = 2
IDX_LONGITUDE = 5
IDX_LATITUDE = 6
IDX_BARO_ALTITUDE = 7
IDX_ON_GROUND = 8
IDX_VELOCITY = 9
IDX_HEADING = 10


class LiveFlight:
    """One airborne aircraft, parsed out of an OpenSky state vector."""

    def __init__(self, state: list) -> None:
        self.callsign = str(state[IDX_CALLSIGN] or "").strip()
        self.origin_country = state[IDX_ORIGIN_COUNTRY]
        self.longitude = state[IDX_LONGITUDE]
        self.latitude = state[IDX_LATITUDE]
        self.altitude_m = state[IDX_BARO_ALTITUDE]
        self.on_ground = bool(state[IDX_ON_GROUND])
        velocity_ms = state[IDX_VELOCITY] or 0
        self.speed_kmh = float(velocity_ms) * 3.6
        self.heading_deg = state[IDX_HEADING]

    @property
    def is_airborne(self) -> bool:
        return not self.on_ground


def parse_states(payload: dict[str, Any]) -> list[LiveFlight]:
    """Turn an OpenSky ``states`` payload into LiveFlight objects."""
    return [LiveFlight(state) for state in (payload.get("states") or [])]


async def fetch_route(callsign: str) -> str | None:
    """Best-effort route description from adsbdb; None if unavailable."""
    try:
        data = await get_json(ADSBDB_URL.format(callsign=callsign))
    except SkillHTTPError:
        return None

    flightroute = ((data.get("response") or {}).get("flightroute")) or {}
    airline = (flightroute.get("airline") or {}).get("name")
    origin = _airport_label(flightroute.get("origin"))
    destination = _airport_label(flightroute.get("destination"))

    parts = []
    if origin and destination:
        parts.append(f"{origin} -> {destination}")
    if airline:
        parts.append(f"operated by {airline}")
    return ", ".join(parts) if parts else None


def _airport_label(airport: dict | None) -> str:
    """Compact 'ICAO City' label for an adsbdb airport object."""
    if not isinstance(airport, dict):
        return ""
    code = airport.get("icao_code") or ""
    city = airport.get("municipality") or ""
    return f"{code} {city}".strip()


async def fetch_aviationstack(api_key: str, callsign: str) -> str:
    """Scheduled/real-time flight info from Aviationstack.

    Tries the IATA flight designator first (KE801) and then the ICAO
    one (KAL801), since callers may supply either form. Raises
    SkillHTTPError when no matching flight exists so the caller can
    fall back to radar data.
    """
    data = {}
    error: Exception | None = None
    for param in ("flight_iata", "flight_icao"):
        try:
            data = await get_json(
                AVIATIONSTACK_URL,
                params={"access_key": api_key, param: callsign, "limit": 1},
            )
            if data.get("data"):
                break
        except SkillHTTPError as exc:
            error = exc
    flights = data.get("data") or []
    if not flights:
        if error is not None:
            raise error
        raise SkillHTTPError(f"No flight found for '{callsign}' on Aviationstack.")

    entry = flights[0]
    status = AVIATIONSTACK_STATUS.get(str(entry.get("flight_status", "")), "unknown")
    airline = ((entry.get("airline") or {}).get("name")) or ""
    flight_iata = (entry.get("flight") or {}).get("iata") or callsign

    departure = entry.get("departure") or {}
    arrival = entry.get("arrival") or {}

    def endpoint(side: dict) -> str:
        name = side.get("airport") or "?"
        iata = side.get("iata") or ""
        code = f" ({iata})" if iata else ""
        return f"{name}{code}"

    def scheduled_time(side: dict) -> str:
        raw = side.get("scheduled") or ""
        return f", scheduled {raw[:16].replace('T', ' ')}" if raw else ""

    delay = departure.get("delay")
    delay_part = f", delayed {int(delay)} min" if delay else ""

    header = f"{airline} {flight_iata}: {status}".strip()
    route_line = (
        f"{endpoint(departure)}{scheduled_time(departure)}{delay_part}"
        f" -> {endpoint(arrival)}{scheduled_time(arrival)}."
    )
    return f"{header}\n{route_line}"


class FlightSkill(Skill):
    name = "get_flight_info"
    description = (
        "Look up live flights. With a callsign (e.g. 'KAL801', 'UAL2'): "
        "position, altitude, speed and route of that flight. "
        "With a city: how many aircraft are flying near it right now."
    )

    def __init__(self, settings) -> None:
        self._settings = settings

    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "callsign": {
                "type": "string",
                "description": ("Flight callsign like 'KAL801' or 'AAR112'. Provide this OR city."),
            },
            "city": {
                "type": "string",
                "description": "Count aircraft flying near this city. Provide this OR callsign.",
            },
        },
        "required": [],
    }

    async def run(self, callsign: str = "", city: str = "") -> str:
        if callsign.strip():
            return await self._track_callsign(callsign.strip())
        if city.strip():
            return await self._count_near_city(city.strip())
        raise SkillHTTPError("Provide either a callsign or a city.")

    # -- Modes ------------------------------------------------------------------

    async def _track_callsign(self, raw_callsign: str) -> str:
        callsign = raw_callsign.replace(" ", "").upper()

        # Preferred provider: Aviationstack (scheduled + real-time data),
        # available when the operator configured an API key.
        if self._settings.aviationstack_api_key:
            try:
                return await fetch_aviationstack(self._settings.aviationstack_api_key, callsign)
            except SkillHTTPError:
                pass  # fall through to the key-less radar sources

        payload = await get_json(OPENSKY_URL, params={"callsign": callsign})
        flights = [flight for flight in parse_states(payload) if flight.callsign == callsign]
        if not flights:
            raise SkillHTTPError(
                f"No aircraft with callsign '{callsign}' is transmitting right now."
            )
        flight = flights[0]

        if not flight.is_airborne:
            base = f"{flight.callsign} is currently on the ground."
        else:
            base = (
                f"{flight.callsign} ({flight.origin_country}) is airborne at "
                f"{_fmt_number(flight.latitude, 2)}, {_fmt_number(flight.longitude, 2)}, "
                f"altitude {_fmt_thousands(flight.altitude_m)} m, "
                f"speed {_fmt_thousands(round(flight.speed_kmh))} km/h."
            )

        route = await fetch_route(callsign)
        if route:
            return f"{base} Route: {route}."
        return base

    async def _count_near_city(self, city: str) -> str:
        location = await geocode(city)
        params = {
            "lamin": location["latitude"] - SEARCH_RADIUS_DEGREES,
            "lamax": location["latitude"] + SEARCH_RADIUS_DEGREES,
            "lomin": location["longitude"] - SEARCH_RADIUS_DEGREES,
            "lomax": location["longitude"] + SEARCH_RADIUS_DEGREES,
        }
        payload = await get_json(OPENSKY_URL, params=params)
        flights = [flight for flight in parse_states(payload) if flight.is_airborne]

        header = f"Aircraft in the air within ~330 km of {location['name']}: {len(flights)}."
        examples = [
            f"{flight.callsign or 'unknown'} ({flight.origin_country})" for flight in flights[:5]
        ]
        if examples:
            return f"{header} Examples: {', '.join(examples)}."
        return header


# -- Formatting helpers ---------------------------------------------------------


def _fmt_number(value: float | None, digits: int) -> str:
    return "?" if value is None else f"{value:.{digits}f}"


def _fmt_thousands(value: float | int | None) -> str:
    if value is None:
        return "?"
    return f"{int(value):,}"
