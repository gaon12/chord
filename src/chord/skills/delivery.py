"""Package-delivery tracking skill for Korean carriers.

Two carriers are supported out of the box, both key-less:

* **CJ Logistics** (CJ대한통운) - the tracking page is JavaScript-
  rendered, so we call its internal AJAX endpoint instead:
  GET the page to obtain a CSRF token, then POST the waybill number to
  ``/ko/tool/parcel/tracking-detail`` which answers clean JSON.
* **Korea Post** (우체국) - server-rendered HTML; we parse the
  ``processTable`` rows from the trace page.

Each carrier turns its raw data into normalized TrackingEvents so the
skill output stays identical no matter the source. Adding a carrier =
subclassing Carrier, then wiring it into resolve_carrier() and the
alias table.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import httpx

from chord.skills._http import (
    DEFAULT_HEADERS,
    TIMEOUT_SECONDS,
    SkillHTTPError,
    get_json,
    get_text,
)
from chord.skills._quota import get_quota_store
from chord.skills.base import Skill

# ---------------------------------------------------------------------------
# Normalized model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingEvent:
    """One scan event in a parcel's journey, already normalized."""

    time: str
    location: str
    status: str


def latest_event(events: list[TrackingEvent]) -> TrackingEvent | None:
    """The most recent event (carriers return oldest-first)."""
    return events[-1] if events else None


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------

_CJ_PAGE_URL = "https://www.cjlogistics.com/ko/tool/parcel/tracking"
_CJ_AJAX_URL = "https://www.cjlogistics.com/ko/tool/parcel/tracking-detail"

#: CJ 'crgSt' codes -> short English status labels.
CJ_STATUS_CODES = {
    "11": "picked up",
    "21": "in transit",
    "41": "in transit",
    "44": "in transit",
    "RMN": "hub arrival",
    "42": "arrived at local hub",
    "82": "out for delivery",
    "91": "delivered",
}

_POST_PAGE_URL = "https://service.epost.go.kr/trace.RetrieveDomRigiTraceList.comm"

#: SweetTracker (tracking.sweettracker.co.kr) company codes for the
#: carriers this skill knows by id.
SWEETTRACKER_COMPANY_CODES = {
    "post": "01",  # 우체국택배
    "cj": "04",  # CJ대한통운
    "hanjin": "05",  # 한진택배
    "logen": "06",  # 로젠택배
    "lotte": "08",  # 롯데택배
}
SWEETTRACKER_NAMES = {
    "post": "Korea Post",
    "cj": "CJ Logistics",
    "hanjin": "Hanjin",
    "logen": "Logen",
    "lotte": "Lotte",
}
SWEETTRACKER_URL = "https://tracking.sweettracker.co.kr/api/v1/trackingInfo"

logger = logging.getLogger(__name__)

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_CSRF_VALUE_RE = re.compile(r"GLOBAL_CSRF_VALUE\s*=\s*[\"']([^\"']+)[\"']")
_CSRF_NAME_RE = re.compile(r"GLOBAL_CSRF_NAME\s*=\s*[\"']([^\"']+)[\"']")


class Carrier(ABC):
    """One delivery company and how to query it."""

    #: Short id used in tool arguments ('cj', 'post').
    id: str = ""
    #: Human-readable name used in replies.
    name: str = ""

    @abstractmethod
    async def track(self, number: str) -> list[TrackingEvent]:
        """Return the parcel's events, oldest first."""

    @abstractmethod
    def tracking_url(self, number: str) -> str:
        """Public page users can open for details."""


class CJLogisticsCarrier(Carrier):
    """CJ대한통운 - queries the site's own AJAX endpoint."""

    id = "cj"
    name = "CJ Logistics"

    def tracking_url(self, number: str) -> str:
        return f"{_CJ_PAGE_URL}?paramInvcNo={number}"

    async def track(self, number: str) -> list[TrackingEvent]:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=DEFAULT_HEADERS) as client:
            # Both calls must share one client: the POST is only accepted
            # when it carries the cookies set by the initial page load.
            csrf_name, csrf_value = await self._fetch_csrf(client)
            data = await self._post_detail(client, csrf_name, csrf_value, number)

        detail_list = (data.get("parcelDetailResultMap") or {}).get("resultList") or []
        events = [
            TrackingEvent(
                time=str(item.get("dTime", "")).strip(),
                location=str(item.get("regBranNm", "")).strip(),
                status=CJ_STATUS_CODES.get(str(item.get("crgSt", "")), str(item.get("crgSt", ""))),
            )
            for item in detail_list
        ]
        if not events:
            raise SkillHTTPError(
                f"No tracking records found at {self.name} for '{number}'. "
                "Check the number or try again after it is scanned."
            )
        return events

    async def _fetch_csrf(self, client: httpx.AsyncClient) -> tuple[str, str]:
        """Load the tracking page and extract its CSRF token pair."""
        try:
            response = await client.get(_CJ_PAGE_URL)
            response.raise_for_status()
            page_html = response.text
        except httpx.HTTPStatusError as exc:
            raise SkillHTTPError(f"{self.name} answered HTTP {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            raise SkillHTTPError(f"Could not reach {self.name}.") from exc

        name_match = _CSRF_NAME_RE.search(page_html)
        value_match = _CSRF_VALUE_RE.search(page_html)
        if not value_match:
            raise SkillHTTPError("Could not obtain a session token from CJ Logistics.")
        return (name_match.group(1) if name_match else "_csrf"), value_match.group(1)

    async def _post_detail(
        self,
        client: httpx.AsyncClient,
        csrf_name: str,
        csrf_value: str,
        number: str,
    ):
        """POST the waybill with the CSRF token pair."""
        headers = {
            "Referer": _CJ_PAGE_URL,
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            response = await client.post(
                _CJ_AJAX_URL,
                data={csrf_name: csrf_value, "paramInvcNo": number},
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise SkillHTTPError(f"{self.name} answered HTTP {exc.response.status_code}.") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise SkillHTTPError(f"Could not query {self.name}.") from exc


class KoreaPostCarrier(Carrier):
    """우체국 - parses the server-rendered trace table."""

    id = "post"
    name = "Korea Post"

    def tracking_url(self, number: str) -> str:
        return f"{_POST_PAGE_URL}?sid1={number}"

    async def track(self, number: str) -> list[TrackingEvent]:
        html = await get_text(
            _POST_PAGE_URL,
            params={"sid1": number},
        )
        events = _parse_post_table(html)
        if not events:
            raise SkillHTTPError(
                f"No tracking records found at Korea Post for '{number}'. Check the number."
            )
        return events


class SweetTrackerCarrier(Carrier):
    """Aggregated tracking via SweetTracker (스마트택배), needs an API key.

    Preferred over the site scrapers because it covers many carriers
    (CJ, 우체국, 한진, 로젠, 롯데, ...) behind one normalized API.

    Quota rules enforced here:
    * 100 lookups per month (shared bucket ``sweettracker``).
    * The SAME waybill number may only be queried 10 times per day;
      repeat requests are answered from the cached last result instead
      of spending another paid lookup.
    """

    id = "sweettracker"

    #: SweetTracker rejects more than 10 queries for one waybill/day.
    PER_NUMBER_DAILY_LIMIT = 10

    def __init__(
        self,
        api_key: str,
        company_code: str,
        display_name: str,
        settings=None,
    ) -> None:
        self._api_key = api_key
        self.company_code = company_code
        self.name = display_name
        self._settings = settings

    def tracking_url(self, number: str) -> str:
        return f"https://tracking.sweettracker.co.kr/?t_code={self.company_code}&t_invoice={number}"

    async def track(self, number: str) -> list[TrackingEvent]:
        store = (
            get_quota_store(self._settings.quota_store_path) if self._settings is not None else None
        )
        cache_key = f"sweettracker#{number}"

        # Same-number daily rule: serve repeats from today's cache.
        if store is not None and store.daily_count(cache_key) >= self.PER_NUMBER_DAILY_LIMIT:
            cached = store.get_cached(cache_key)
            if cached is not None:
                logger.info("Serving cached tracking for %s (daily limit reached).", number)
                return [TrackingEvent(**item) for item in cached]
            raise SkillHTTPError(
                f"Waybill {number} was already looked up "
                f"{self.PER_NUMBER_DAILY_LIMIT} times today. Try again tomorrow."
            )

        if store is not None:
            store.require("sweettracker")
            store.bump_daily(cache_key)

        data = await get_json(
            SWEETTRACKER_URL,
            params={
                "t_key": self._api_key,
                "t_code": self.company_code,
                "t_invoice": number,
            },
        )
        # The API signals failures with {"status": false, "msg": "..."}.
        if data.get("status") is False:
            raise SkillHTTPError(f"SweetTracker: {data.get('msg', 'lookup failed')}.")

        details = data.get("trackingDetails") or []
        events = [
            TrackingEvent(
                time=str(item.get("time", "")).strip(),
                location=str((item.get("location") or {}).get("name", "")).strip(),
                status=str((item.get("status") or {}).get("text", "")).strip(),
            )
            for item in details
        ]
        if not events:
            raise SkillHTTPError(
                f"No tracking records found at {self.name} for '{number}'. Check the number."
            )

        if store is not None:
            store.record("sweettracker")
            store.put_cached(cache_key, [dataclasses.asdict(event) for event in events])
        return events


def _parse_post_table(html: str) -> list[TrackingEvent]:
    """Extract events from the Korea Post ``processTable`` markup.

    Each populated row has three cells: date/time, location, and the
    status (a span with class ``evtnm`` plus optional detail text).
    Parsing is deliberately tolerant - cells may contain nested spans.
    """
    events: list[TrackingEvent] = []

    table_match = re.search(r'<table[^>]*id\s*=\s*"processTable"[^>]*>(.*?)</table>', html, re.S)
    section = table_match.group(1) if table_match else html

    for row_match in _ROW_RE.finditer(section):
        row = row_match.group(1)
        evtnm = re.search(r'class="evtnm"[^>]*>([^<]+)<', row)
        if not evtnm:
            continue
        cells = [_TAG_RE.sub(" ", cell.group(1)).strip() for cell in _CELL_RE.finditer(row)]
        cells = [re.sub(r"\s+", " ", cell) for cell in cells]
        if len(cells) < 3:
            continue
        events.append(
            TrackingEvent(
                time=cells[0],
                location=cells[1],
                status=evtnm.group(1).strip(),
            )
        )
    return events


# Alias tables so the LLM can pass several natural spellings.
CARRIER_ALIASES: dict[str, str] = {
    "cj": "cj",
    "cjlogistics": "cj",
    "cj대한통운": "cj",
    "대한통운": "cj",
    "post": "post",
    "koreapost": "post",
    "epost": "post",
    "우체국": "post",
    "hanjin": "hanjin",
    "한진": "hanjin",
    "logen": "logen",
    "로젠": "logen",
    "lotte": "lotte",
    "롯데": "lotte",
}


def resolve_carrier(name: str, settings=None) -> Carrier:
    """Look up a carrier by id or alias.

    When a SweetTracker API key is configured and the carrier is one
    SweetTracker covers, it is preferred over the site scrapers
    because one stable JSON API beats per-site HTML parsing.
    """
    key = CARRIER_ALIASES.get(name.strip().lower())
    if key is None:
        supported = ", ".join(sorted(set(CARRIER_ALIASES)))
        raise SkillHTTPError(f"Unknown carrier '{name}'. Supported: {supported}.")

    sweettracker_api_key = getattr(settings, "sweettracker_api_key", "") or ""
    company_code = SWEETTRACKER_COMPANY_CODES.get(key)
    if sweettracker_api_key and company_code:
        return SweetTrackerCarrier(
            api_key=sweettracker_api_key,
            company_code=company_code,
            display_name=SWEETTRACKER_NAMES[key],
            settings=settings,
        )

    scraper_classes = {"cj": CJLogisticsCarrier, "post": KoreaPostCarrier}
    scraper_class = scraper_classes.get(key)
    if scraper_class is None:
        raise SkillHTTPError(
            f"'{name}' requires a SweetTracker API key (set SWEETTRACKER_API_KEY)."
        )
    return scraper_class()


def is_delivered(status: str) -> bool:
    """Delivered detection across carriers (English + Korean labels)."""
    lowered = status.lower()
    return "delivered" in lowered or "배달완료" in status or "도착완료" in status


def format_tracking(carrier: Carrier, number: str, events: list[TrackingEvent]) -> str:
    """Render events into one compact chat-friendly summary."""
    last = latest_event(events)
    lines = [f"{carrier.name} tracking {number}: {last.status} ({last.time}, {last.location})."]
    lines.append("Delivered." if is_delivered(last.status) else "Not delivered yet.")
    history = " | ".join(f"{event.status} @ {event.location} {event.time}" for event in events[-3:])
    lines.append(f"Recent scans: {history}")
    lines.append(f"Details: {carrier.tracking_url(number)}")
    return " ".join(lines)


class DeliverySkill(Skill):
    name = "track_parcel"
    description = (
        "Track a Korean parcel delivery. Supported carriers: 'cj' "
        "(CJ Logistics / CJ대한통운), 'post' (Korea Post / 우체국), plus "
        "'hanjin', 'logen' and 'lotte' when SWEETTRACKER_API_KEY is set. "
        "Pass the carrier id and the tracking number."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "carrier": {
                "type": "string",
                "description": (
                    "Carrier id: 'cj' (CJ Logistics), 'post' (Korea Post), "
                    "'hanjin', 'logen' or 'lotte'."
                ),
            },
            "tracking_number": {
                "type": "string",
                "description": "Waybill number (digits only), e.g. '12345678901'.",
            },
        },
        "required": ["carrier", "tracking_number"],
    }

    def __init__(self, settings) -> None:
        self._settings = settings

    async def run(self, carrier: str, tracking_number: str) -> str:
        number = re.sub(r"\D", "", tracking_number)
        if not number:
            raise SkillHTTPError(f"'{tracking_number}' does not look like a tracking number.")

        resolved = resolve_carrier(carrier, self._settings)
        events = await resolved.track(number)
        return format_tracking(resolved, number, events)
