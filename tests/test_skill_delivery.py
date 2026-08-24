"""Tests for the delivery-tracking skill (all HTTP mocked with respx).

Fixtures mirror the real response shapes observed on the carrier sites:
CJ answers JSON from an AJAX endpoint; Korea Post renders HTML tables.
"""

from __future__ import annotations

import json

import pytest
import respx

import quota_helpers
from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.delivery import (
    CJLogisticsCarrier,
    DeliverySkill,
    KoreaPostCarrier,
    TrackingEvent,
    _parse_post_table,
    format_tracking,
    latest_event,
    resolve_carrier,
)

CJ_PAGE = "https://www.cjlogistics.com/ko/tool/parcel/tracking"
SWEETTRACKER_URL = "https://tracking.sweettracker.co.kr/api/v1/trackingInfo"


def _settings(**keys):
    return Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        **keys,
    )


CJ_AJAX = "https://www.cjlogistics.com/ko/tool/parcel/tracking-detail"
POST_URL = "https://service.epost.go.kr/trace.RetrieveDomRigiTraceList.comm"


def _cj_page_html():
    return 'var GLOBAL_CSRF_NAME = "_csrf"; var GLOBAL_CSRF_VALUE = "token-123";'


def _cj_detail_json():
    # Field names match CJ's own front-end code (fncCreateResult).
    return {
        "parcelResultMap": {"resultList": [{"invcNo": "12345678901"}]},
        "parcelDetailResultMap": {
            "resultList": [
                {
                    "crgSt": "11",
                    "dTime": "2026-08-21 09:10:00",
                    "crgNm": "상품인수",
                    "regBranNm": "서울특별지사",
                },
                {
                    "crgSt": "41",
                    "dTime": "2026-08-22 08:22:11",
                    "crgNm": "상품이동중",
                    "regBranNm": "경기광명HUB",
                },
                {
                    "crgSt": "82",
                    "dTime": "2026-08-22 09:00:00",
                    "crgNm": "배송출발(배송담당: 홍길동 010-1234-5678)",
                    "regBranNm": "강남지점",
                },
            ]
        },
    }


def _post_html():
    """Korea Post trace page skeleton with two populated rows."""
    return """
    <html><body>
    <table id = "processTable" class="table_col detail_off">
      <tbody>
        <tr>
          <td>2026.08.21<br>09:30</td>
          <td><a>서울중앙우체국</a></td>
          <td><span class="evtnm">접수</span><br>상세설명</td>
        </tr>
        <tr>
          <td>2026.08.22<br>10:05</td>
          <td><a>역삼동</a></td>
          <td><span class="evtnm">배달준비</span></td>
        </tr>
      </tbody>
    </table>
    </body></html>
    """


# -- Carrier resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "carrier_type"),
    [
        ("cj", CJLogisticsCarrier),
        ("CJ대한통운", CJLogisticsCarrier),
        ("koreapost", KoreaPostCarrier),
        ("우체국", KoreaPostCarrier),
    ],
)
def test_resolve_carrier_aliases(name, carrier_type):
    assert isinstance(resolve_carrier(name), carrier_type)


def test_resolve_unknown_carrier_errors():
    with pytest.raises(SkillHTTPError, match="Unknown carrier"):
        resolve_carrier("fedex")


# -- CJ carrier -------------------------------------------------------------------


@respx.mock
async def test_cj_track_maps_status_codes():
    respx.get(CJ_PAGE).respond(text=_cj_page_html())
    respx.post(CJ_AJAX).respond(json=_cj_detail_json())

    events = await CJLogisticsCarrier().track("12345678901")

    assert [e.status for e in events] == ["picked up", "in transit", "out for delivery"]
    assert events[-1].location == "강남지점"


@respx.mock
async def test_cj_no_records_raises_readable_error():
    respx.get(CJ_PAGE).respond(text=_cj_page_html())
    empty = {
        "parcelResultMap": {"resultList": []},
        "parcelDetailResultMap": {"resultList": []},
    }
    respx.post(CJ_AJAX).respond(json=empty)

    with pytest.raises(SkillHTTPError, match="No tracking records found"):
        await CJLogisticsCarrier().track("00000000000")


# -- Korea Post carrier ------------------------------------------------------------


@respx.mock
async def test_post_track_parses_table_rows():
    respx.get(POST_URL).respond(text=_post_html())

    events = await KoreaPostCarrier().track("1234567890123")

    assert len(events) == 2
    assert events[0].status == "접수"
    assert events[0].location == "서울중앙우체국"
    assert events[1].status == "배달준비"
    assert events[1].time.startswith("2026.08.22")


def test_parse_post_table_without_data_returns_empty():
    html = '<table id = "processTable"><tbody><tr><td colspan="4">no data</td></tr></tbody></table>'
    assert _parse_post_table(html) == []


# -- Skill-level behavior -------------------------------------------------------------


@respx.mock
async def test_skill_formats_latest_status_and_link():
    respx.get(CJ_PAGE).respond(text=_cj_page_html())
    respx.post(CJ_AJAX).respond(json=_cj_detail_json())

    result = await DeliverySkill(_settings()).run(carrier="cj", tracking_number="12345678901")

    assert "CJ Logistics tracking 12345678901" in result
    assert "out for delivery" in result
    assert "Not delivered yet." in result
    assert "Details:" in result


async def test_skill_strips_non_digit_characters():
    events = [TrackingEvent(time="t", location="l", status="delivered")]
    carrier = KoreaPostCarrier()

    text = format_tracking(carrier, "123", events)

    assert "Delivered." in text


async def test_non_numeric_input_fails_fast():
    with pytest.raises(SkillHTTPError, match="does not look like"):
        await DeliverySkill(_settings()).run(carrier="cj", tracking_number="abc")


# -- Helpers ----------------------------------------------------------------------------


def test_latest_event_picks_last():
    events = [
        TrackingEvent(time="1", location="a", status="picked up"),
        TrackingEvent(time="2", location="b", status="delivered"),
    ]
    assert latest_event(events).status == "delivered"
    assert latest_event([]) is None


# -- SweetTracker provider ----------------------------------------------------------


@respx.mock
async def test_sweettracker_preferred_when_key_present():
    respx.get(SWEETTRACKER_URL).respond(
        json={
            "status": True,
            "carrier": {"id": "04", "name": "CJ대한통운"},
            "trackingDetails": [
                {
                    "time": "2026-08-21 09:10:00",
                    "status": {"id": "pickup", "text": "상품접수"},
                    "location": {"name": "서울특별지사"},
                },
                {
                    "time": "2026-08-22 15:00:00",
                    "status": {"id": "delivered", "text": "배달완료"},
                    "location": {"name": "강남지점"},
                },
            ],
        }
    )

    settings = _settings(sweettracker_api_key="secret")
    result = await DeliverySkill(settings).run(carrier="cj", tracking_number="12345678901")

    assert "CJ Logistics tracking 12345678901" in result
    assert "배달완료" in result
    assert "Delivered." in result


@respx.mock
async def test_sweettracker_api_error_surfaces_message():
    respx.get(SWEETTRACKER_URL).respond(json={"status": False, "msg": "잘못된 운송장 번호입니다."})

    settings = _settings(sweettracker_api_key="secret")
    with pytest.raises(SkillHTTPError, match="잘못된"):
        await DeliverySkill(settings).run(carrier="cj", tracking_number="12345678901")


async def test_sweettracker_only_carriers_require_key():
    settings = _settings()  # no key
    with pytest.raises(SkillHTTPError, match="SWEETTRACKER_API_KEY"):
        resolve_carrier("hanjin", settings.sweettracker_api_key)


def test_is_delivered_covers_languages():
    from chord.skills.delivery import is_delivered

    assert is_delivered("delivered")
    assert is_delivered("배달완료")
    assert is_delivered("도착완료")
    assert not is_delivered("in transit")
    assert not is_delivered("상품이동중")


# -- Quota enforcement ---------------------------------------------------------------


def _bump_monthly(bucket: str, count: int) -> None:
    from datetime import datetime

    path = quota_helpers.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    monthly = data.setdefault("monthly", {})
    entry = monthly.setdefault(bucket, {"period": "", "count": 0})
    entry["period"] = datetime.now().strftime("%Y-%m")
    entry["count"] = count
    path.write_text(json.dumps(data), encoding="utf-8")


@respx.mock
async def test_sweettracker_monthly_quota_blocks_lookup():
    _bump_monthly("sweettracker", 100)

    settings = _settings(sweettracker_api_key="secret")
    with pytest.raises(SkillHTTPError, match="monthly limit of 100"):
        await DeliverySkill(settings).run(carrier="cj", tracking_number="12345678901")


@respx.mock
async def test_same_waybill_over_daily_cap_serves_cache():
    respx.get(SWEETTRACKER_URL).respond(
        json={
            "status": True,
            "trackingDetails": [
                {
                    "time": "2026-08-22 15:00:00",
                    "status": {"id": "delivered", "text": "배달완료"},
                    "location": {"name": "강남지점"},
                }
            ],
        }
    )
    settings = _settings(sweettracker_api_key="secret")

    first = await DeliverySkill(settings).run(carrier="cj", tracking_number="77777777777")

    # Burn through the per-number daily budget (10 lookups per day).
    from chord.skills._quota import get_quota_store

    store = get_quota_store(quota_helpers.store_path())
    key = "sweettracker#77777777777"
    store.bump_daily(key, 10 - store.daily_count(key))

    route = respx.get(SWEETTRACKER_URL)
    calls_before = route.call_count
    second = await DeliverySkill(settings).run(carrier="cj", tracking_number="77777777777")

    assert route.call_count == calls_before  # served from cache, no API call
    assert second == first
