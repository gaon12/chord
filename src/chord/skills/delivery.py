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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import httpx

from chord.skills._http import (
    DEFAULT_HEADERS,
    TIMEOUT_SECONDS,
    SkillHTTPError,
    get_text,
)
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
}


def resolve_carrier(name: str) -> Carrier:
    """Look up a carrier by id or alias."""
    key = CARRIER_ALIASES.get(name.strip().lower())
    if key is None:
        supported = ", ".join(sorted({a for a in CARRIER_ALIASES}))
        raise SkillHTTPError(f"Unknown carrier '{name}'. Supported: {supported}.")
    carrier_class = {"cj": CJLogisticsCarrier, "post": KoreaPostCarrier}[key]
    return carrier_class()


def format_tracking(carrier: Carrier, number: str, events: list[TrackingEvent]) -> str:
    """Render events into one compact chat-friendly summary."""
    last = latest_event(events)
    lines = [f"{carrier.name} tracking {number}: {last.status} ({last.time}, {last.location})."]
    delivered = last.status == "delivered"
    lines.append("Delivered." if delivered else "Not delivered yet.")
    history = " | ".join(f"{event.status} @ {event.location} {event.time}" for event in events[-3:])
    lines.append(f"Recent scans: {history}")
    lines.append(f"Details: {carrier.tracking_url(number)}")
    return " ".join(lines)


class DeliverySkill(Skill):
    name = "track_parcel"
    description = (
        "Track a Korean parcel delivery. Supported carriers: "
        "'cj' (CJ Logistics / CJ대한통운) and 'post' (Korea Post / 우체국). "
        "Pass the carrier id and the tracking number."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "carrier": {
                "type": "string",
                "description": "Carrier id: 'cj' for CJ Logistics, 'post' for Korea Post.",
            },
            "tracking_number": {
                "type": "string",
                "description": "Waybill number (digits only), e.g. '12345678901'.",
            },
        },
        "required": ["carrier", "tracking_number"],
    }

    async def run(self, carrier: str, tracking_number: str) -> str:
        number = re.sub(r"\D", "", tracking_number)
        if not number:
            raise SkillHTTPError(f"'{tracking_number}' does not look like a tracking number.")

        resolved = resolve_carrier(carrier)
        events = await resolved.track(number)
        return format_tracking(resolved, number, events)
