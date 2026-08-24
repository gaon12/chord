"""Tests for map and navigation skills (Kakao / Nominatim / OSRM, mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

import quota_helpers
from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.map import (
    KAKAO_KEYWORD_URL,
    KAKAO_NAVI_URL,
    NOMINATIM_URL,
    OSRM_URL,
    FindPlacesSkill,
    GetDirectionsSkill,
)

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _settings(**keys) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


def _geocode(name="Seoul", country="South Korea", lat=37.566, lon=126.978):
    return {"results": [{"name": name, "country": country, "latitude": lat, "longitude": lon}]}


def _kakao_places():
    return {
        "documents": [
            {
                "place_name": "스타벅스 강남점",
                "road_address_name": "서울 강남구 테헤란로 123",
                "phone": "02-123-4567",
                "place_url": "https://place.kakao.com/1",
            }
        ]
    }


# -- find_places -------------------------------------------------------------------


@respx.mock
async def test_find_places_korea_uses_kakao_when_key_present():
    respx.get(GEO_URL).respond(json=_geocode())
    route = respx.get(KAKAO_KEYWORD_URL).respond(json=_kakao_places())

    settings = _settings(kakao_rest_api_key="secret")
    result = await FindPlacesSkill(settings).run(query="starbucks gangnam seoul")

    assert route.called
    assert "[via Kakao]" in result
    assert "스타벅스 강남점" in result
    assert "tel 02-123-4567" in result


@respx.mock
async def test_find_places_falls_back_to_nominatim():
    respx.get(GEO_URL).respond(json=_geocode(country="France", lat=48.85, lon=2.35))
    kakao_route = respx.get(KAKAO_KEYWORD_URL).respond(json={"documents": []})
    respx.get(NOMINATIM_URL).respond(
        json=[
            {
                "name": "Cafe de Flore",
                "display_name": "Cafe de Flore, 75006 Paris, France",
            }
        ]
    )

    # Kakao key present but query is abroad -> Nominatim answers.
    settings = _settings(kakao_rest_api_key="secret")
    result = await FindPlacesSkill(settings).run(query="cafe de flore paris")

    assert not kakao_route.called
    assert "[via OpenStreetMap]" in result
    assert "Cafe de Flore" in result


@respx.mock
async def test_find_places_no_results_raises():
    respx.get(GEO_URL).respond(json=_geocode())
    respx.get(KAKAO_KEYWORD_URL).respond(json={"documents": []})
    respx.get(NOMINATIM_URL).respond(json=[])

    with pytest.raises(SkillHTTPError, match="No places found"):
        await FindPlacesSkill(_settings(kakao_rest_api_key="k")).run(query="qqq")


# -- get_directions ------------------------------------------------------------------


@respx.mock
async def test_directions_korea_uses_kakao_navi():
    respx.get(GEO_URL).respond(json=_geocode(name="Incheon", lat=37.456, lon=126.705))
    navi_route = respx.get(KAKAO_NAVI_URL).respond(
        json={
            "routes": [
                {"summary": {"distance": 30200, "duration": 2_100_000}}  # 30.2km/35min
            ]
        }
    )

    settings = _settings(kakao_rest_api_key="secret")
    result = await GetDirectionsSkill(settings).run(origin="Incheon", destination="Seoul")

    assert navi_route.called
    assert "[via Kakao Navi]" in result
    assert "30.2 km" in result
    assert "35 min" in result


@respx.mock
async def test_directions_abroad_uses_osrm():
    # Two sequential geocode calls: Paris first, then Lyon.
    paris = {
        "results": [
            {"name": "Paris", "country": "France", "latitude": 48.8566, "longitude": 2.3522}
        ]
    }
    lyon = {
        "results": [{"name": "Lyon", "country": "France", "latitude": 45.7640, "longitude": 4.8357}]
    }
    respx.get(GEO_URL).mock(
        side_effect=[
            httpx.Response(200, json=paris),
            httpx.Response(200, json=lyon),
        ]
    )
    osrm_route = respx.get(OSRM_URL.format(coords="2.3522,48.8566;4.8357,45.764")).respond(
        json={"routes": [{"distance": 465000, "duration": 18000}]}
    )

    result = await GetDirectionsSkill(_settings()).run(origin="Paris", destination="Lyon")

    assert osrm_route.called
    assert "[via OSRM]" in result
    assert "465.0 km" in result
    assert "5 h 0 min" in result


@respx.mock
async def test_directions_no_route_raises_readable_error():
    a = {"results": [{"name": "A", "country": "Japan", "latitude": 36.0, "longitude": 138.0}]}
    b = {"results": [{"name": "B", "country": "Japan", "latitude": 35.0, "longitude": 139.0}]}
    respx.get(GEO_URL).mock(
        side_effect=[
            httpx.Response(200, json=a),
            httpx.Response(200, json=b),
        ]
    )
    respx.get(OSRM_URL.format(coords="138.0,36.0;139.0,35.0")).respond(json={"routes": []})

    with pytest.raises(SkillHTTPError, match="no driving route"):
        await GetDirectionsSkill(_settings()).run(origin="A", destination="B")


@respx.mock
async def test_exhausted_kakao_quota_falls_back_to_osm():
    quota_helpers.prefill_quota("kakao_map", 300_000)
    respx.get(GEO_URL).respond(json=_geocode())
    kakao_route = respx.get(KAKAO_KEYWORD_URL).respond(json=_kakao_places())
    respx.get(NOMINATIM_URL).respond(
        json=[
            {
                "name": "Starbucks Gangnam",
                "display_name": "Starbucks, Gangnam, Seoul, South Korea",
            }
        ]
    )

    settings = _settings(kakao_rest_api_key="secret")
    result = await FindPlacesSkill(settings).run(query="starbucks gangnam seoul")

    assert not kakao_route.called  # quota checked before any request
    assert "[via OpenStreetMap]" in result
