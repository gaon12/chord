"""Date/time and timezone skills (pure Python, no network needed).

Two tools:

* ``get_current_datetime`` - now() for an IANA timezone, including the
  abbreviation and UTC offset.
* ``convert_timezone``     - move a wall-clock time between zones;
  accepts ISO 8601 input plus a few common fallbacks.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chord.skills.base import Skill

#: Sane default for the expected audience of this bot.
DEFAULT_TIMEZONE = "Asia/Seoul"

#: Extra parse attempts after datetime.fromisoformat fails.
FALLBACK_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y.%m.%d %H:%M:%S",
    "%d %b %Y %H:%M",
]


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve an IANA zone name, raising a readable error otherwise."""
    candidate = (name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    # Accept common short forms people type.
    aliases = {
        "kst": "Asia/Seoul",
        "jst": "Asia/Tokyo",
        "utc": "UTC",
        "korea": "Asia/Seoul",
        "seoul": "Asia/Seoul",
        "tokyo": "Asia/Tokyo",
        "newyork": "America/New_York",
        "london": "Europe/London",
    }
    candidate = aliases.get(candidate.lower().replace(" ", ""), candidate)
    try:
        return ZoneInfo(candidate)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SkillValueError(
            f"Unknown timezone '{name}'. Use IANA names like 'Asia/Seoul'."
        ) from exc


class SkillValueError(ValueError):
    """Raised for bad user input; converted to text by the registry."""


def parse_datetime(text: str, zone: ZoneInfo) -> datetime:
    """Parse flexible datetime text and attach the given timezone."""
    cleaned = text.strip()
    naive = None
    try:
        naive = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in FALLBACK_FORMATS:
            try:
                naive = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
    if naive is None:
        raise SkillValueError(f"Could not parse '{text}'. Try ISO format like 2026-08-23 14:00.")
    if naive.tzinfo is None:
        return naive.replace(tzinfo=zone)
    return naive.astimezone(zone)


class CurrentDatetimeSkill(Skill):
    name = "get_current_datetime"
    description = (
        "Get today's date and current time for a timezone, with the "
        "UTC offset. Defaults to Asia/Seoul."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone such as 'Asia/Seoul', 'UTC', "
                    "'America/New_York'. Default 'Asia/Seoul'."
                ),
            }
        },
        "required": [],
    }

    async def run(self, timezone: str = "") -> str:
        zone = resolve_timezone(timezone)
        now = datetime.now(zone)
        offset = now.strftime("%z") or "+0000"
        offset_pretty = f"UTC{offset[:3]}:{offset[3:]}"
        weekday = now.strftime("%A")
        return (
            f"{zone.key}: {now.strftime('%Y-%m-%d')} ({weekday}) "
            f"{now.strftime('%H:%M:%S')} ({offset_pretty})"
        )


class ConvertTimezoneSkill(Skill):
    name = "convert_timezone"
    description = (
        "Convert a specific time from one timezone to another, e.g. "
        "'2026-08-23 14:00' from 'Asia/Seoul' to 'America/New_York'."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "datetime_text": {
                "type": "string",
                "description": "The moment to convert, e.g. '2026-08-23 14:00'.",
            },
            "from_timezone": {
                "type": "string",
                "description": "Timezone of the input time.",
            },
            "to_timezone": {
                "type": "string",
                "description": "Target timezone.",
            },
        },
        "required": ["datetime_text", "to_timezone"],
    }

    async def run(
        self,
        datetime_text: str,
        to_timezone: str,
        from_timezone: str = "",
    ) -> str:
        source_zone = resolve_timezone(from_timezone or DEFAULT_TIMEZONE)
        target_zone = resolve_timezone(to_timezone)
        moment = parse_datetime(datetime_text, source_zone)
        converted = moment.astimezone(target_zone)

        return (
            f"{moment.strftime('%Y-%m-%d %H:%M')} ({source_zone.key}) = "
            f"{converted.strftime('%Y-%m-%d %H:%M')} ({target_zone.key})"
        )
