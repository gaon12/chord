"""Tests for the flight-info skill (OpenSky + adsbdb, mocked)."""

from __future__ import annotations

import pytest
import respx

from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.flight import FlightSkill, parse_states

OPENSKY_URL = "https://opensky-network.org/api/states/all"
AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"


def _settings(**keys):
    return Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


ADSBDB_URL = "https://api.adsbdb.com/v0/callsign/KAL801"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _state(
    callsign="KAL801",
    lon=124.22,
    lat=37.09,
    altitude=10096.5,
    on_ground=False,
    velocity=246.27,
):
    """One OpenSky state vector with realistic field positions."""
    return [
        "781c93",  # icao24
        f"{callsign:<7}",  # callsign (OpenSky pads with spaces)
        "South Korea",  # origin country
        1787483321,  # time_position
        1787483324,  # last_contact
        lon,  # longitude
        lat,  # latitude
        altitude,  # baro_altitude (m)
        on_ground,  # on_ground
        velocity,  # velocity (m/s)
        234.0,  # heading
        None,  # vertical rate
        None,
        None,
        None,
        None,
        None,
    ]


def _opensky_payload(states):
    return {"time": 1787483326, "states": states}


def _adsbdb_payload():
    return {
        "response": {
            "flightroute": {
                "callsign": "KAL801",
                "airline": {"name": "Korean Air"},
                "origin": {
                    "icao_code": "ICN",
                    "municipality": "Incheon",
                    "country_name": "South Korea",
                },
                "destination": {
                    "icao_code": "NRT",
                    "municipality": "Tokyo",
                    "country_name": "Japan",
                },
            }
        }
    }


# -- Parsing ----------------------------------------------------------------------


def test_parse_states_reads_vector_fields():
    flights = parse_states(_opensky_payload([_state()]))
    assert len(flights) == 1
    flight = flights[0]
    assert flight.callsign == "KAL801"  # padding stripped
    assert flight.origin_country == "South Korea"
    assert flight.is_airborne
    assert flight.speed_kmh == pytest.approx(886.6, rel=0.01)


def test_on_ground_flight_is_not_airborne():
    flight = parse_states(_opensky_payload([_state(on_ground=True)]))[0]
    assert not flight.is_airborne


def test_null_callsign_becomes_empty_string():
    state = _state()
    state[1] = None
    flight = parse_states(_opensky_payload([state]))[0]
    assert flight.callsign == ""


# -- Callsign mode -------------------------------------------------------------------


@respx.mock
async def test_callsign_mode_with_route():
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([_state()]))
    respx.get(ADSBDB_URL).respond(json=_adsbdb_payload())

    result = await FlightSkill(_settings()).run(callsign="kal801")

    assert "KAL801 (South Korea) is airborne" in result
    assert "10,096 m" in result
    assert "887 km/h" in result
    assert "ICN Incheon -> NRT Tokyo" in result
    assert "operated by Korean Air" in result


@respx.mock
async def test_callsign_mode_without_adsbdb_still_reports_position():
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([_state()]))
    respx.get(ADSBDB_URL).respond(status_code=404)

    result = await FlightSkill(_settings()).run(callsign="KAL801")

    assert "airborne" in result
    assert "Route:" not in result


@respx.mock
async def test_callsign_mode_ground_and_missing_cases():
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([_state(on_ground=True)]))
    respx.get(ADSBDB_URL).respond(status_code=404)

    result = await FlightSkill(_settings()).run(callsign="KAL801")
    assert "on the ground" in result

    respx.get(OPENSKY_URL).respond(json=_opensky_payload([]))
    with pytest.raises(SkillHTTPError, match="No aircraft"):
        await FlightSkill(_settings()).run(callsign="KAL801")


# -- City mode -----------------------------------------------------------------------


@respx.mock
async def test_city_mode_counts_aircraft():
    respx.get(GEO_URL).respond(
        json={
            "results": [
                {
                    "name": "Seoul",
                    "country": "South Korea",
                    "latitude": 37.566,
                    "longitude": 126.978,
                }
            ]
        }
    )
    respx.get(OPENSKY_URL).respond(
        json=_opensky_payload(
            [_state(callsign="AAR111"), _state(callsign="QDA919"), _state(on_ground=True)]
        )
    )

    result = await FlightSkill(_settings()).run(city="Seoul")

    # The grounded aircraft must not be counted.
    assert "aircraft in the air within ~330 km of seoul: 2." in result.lower()
    assert "AAR111" in result and "QDA919" in result


@respx.mock
async def test_city_mode_zero_aircraft():
    respx.get(GEO_URL).respond(
        json={"results": [{"name": "Nowhere", "country": "X", "latitude": 0.0, "longitude": 0.0}]}
    )
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([]))

    result = await FlightSkill(_settings()).run(city="Nowhere")

    assert ": 0." in result


async def test_no_arguments_raises_readable_error():
    with pytest.raises(SkillHTTPError, match="Provide either"):
        await FlightSkill(_settings()).run()


# -- Aviationstack provider -------------------------------------------------------


def _aviationstack_payload():
    return {
        "data": [
            {
                "flight_status": "active",
                "airline": {"name": "Korean Air"},
                "flight": {"iata": "KE801", "number": "801"},
                "departure": {
                    "airport": "Incheon International",
                    "iata": "ICN",
                    "scheduled": "2026-08-23T09:30:00+00:00",
                    "delay": 15,
                },
                "arrival": {
                    "airport": "Narita International",
                    "iata": "NRT",
                    "scheduled": "2026-08-23T11:45:00+00:00",
                },
            }
        ]
    }


@respx.mock
async def test_aviationstack_used_when_key_present():
    route = respx.get(
        AVIATIONSTACK_URL,
        params={"flight_iata": "KE801"},
    ).respond(json=_aviationstack_payload())

    settings = _settings(aviationstack_api_key="secret")
    result = await FlightSkill(settings).run(callsign="KE801")

    assert route.called
    assert "Korean Air KE801: en route" in result
    assert "Incheon International (ICN)" in result
    assert "-> Narita International (NRT)" in result
    assert "delayed 15 min" in result
    assert "scheduled 2026-08-23 09:30" in result


@respx.mock
async def test_aviationstack_falls_back_from_iata_to_icao():
    """IATA lookup misses -> the skill retries with the ICAO designator."""
    respx.get(AVIATIONSTACK_URL, params={"flight_iata": "KAL801"}).respond(json={"data": []})
    icao_route = respx.get(
        AVIATIONSTACK_URL,
        params={"flight_icao": "KAL801"},
    ).respond(json=_aviationstack_payload())

    settings = _settings(aviationstack_api_key="secret")
    result = await FlightSkill(settings).run(callsign="KAL801")

    assert icao_route.called
    assert "Korean Air KE801" in result


@respx.mock
async def test_aviationstack_both_misses_fall_back_to_radar():
    """No Aviationstack hit -> OpenSky radar path still answers."""
    respx.get(AVIATIONSTACK_URL, params={"flight_iata": "KAL801"}).respond(json={"data": []})
    respx.get(AVIATIONSTACK_URL, params={"flight_icao": "KAL801"}).respond(json={"data": []})
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([_state()]))
    respx.get(ADSBDB_URL).respond(status_code=404)

    settings = _settings(aviationstack_api_key="secret")
    result = await FlightSkill(settings).run(callsign="KAL801")

    assert "airborne" in result  # radar data used as fallback


@respx.mock
async def test_aviationstack_error_falls_back_to_radar():
    respx.get(AVIATIONSTACK_URL).respond(status_code=503)
    respx.get(OPENSKY_URL).respond(json=_opensky_payload([_state()]))
    respx.get(ADSBDB_URL).respond(status_code=404)

    settings = _settings(aviationstack_api_key="secret")
    result = await FlightSkill(settings).run(callsign="KAL801")

    assert "airborne" in result
