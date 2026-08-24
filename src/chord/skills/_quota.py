"""API-quota tracking and enforcement for keyed providers.

Providers declare their known limits in :data:`LIMITS`; every call goes
through :class:`QuotaStore`, which persists counters in a small JSON
file (default ``usage.json``, git-ignored) so usage survives restarts.

* Monthly buckets reset automatically when the calendar month changes.
* Daily buckets reset when the calendar day changes (local time).
* ``QuotaExceededError`` extends ``SkillHTTPError``, so skills that
  already fall back to secondary providers degrade gracefully instead
  of failing - e.g. an exhausted WeatherAPI key silently switches the
  weather answer to Open-Meteo.

SweetTracker additionally caps *the same waybill number* at 10 lookups
per day; repeat questions are served from a cached copy of the last
successful result instead of burning another paid call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chord.skills._http import SkillHTTPError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuotaLimit:
    """Known upstream limits for one provider bucket."""

    #: Maximum calls per calendar month (None = unlimited/not tracked).
    monthly: int | None = None
    #: Maximum calls per calendar day (None = unlimited/not tracked).
    daily: int | None = None
    #: Human-readable name used in error messages.
    display: str = ""


#: Known limits gathered from each provider's documentation.
LIMITS: dict[str, QuotaLimit] = {
    # Keyed providers (user-provided numbers).
    "sweettracker": QuotaLimit(monthly=100, display="SweetTracker"),
    "kakao_map": QuotaLimit(monthly=300_000, display="Kakao Map"),
    "aviationstack": QuotaLimit(monthly=100, display="Aviationstack"),
    "weatherapi": QuotaLimit(monthly=100_000, display="WeatherAPI.com"),
    # data.go.kr services share one credential but meter separately;
    # limits below are the published daily defaults.
    "kma": QuotaLimit(daily=1_000, display="KMA 기상청"),
    "airkorea": QuotaLimit(daily=500, display="AirKorea 에어코리아"),
    # Key-less providers (documented public limits; enforced defensively).
    "open_meteo": QuotaLimit(daily=10_000, display="Open-Meteo"),
    "opensky": QuotaLimit(daily=100, display="OpenSky"),  # 400 credits/day, ~4/call
}


class QuotaExceededError(SkillHTTPError):
    """A provider bucket is out of budget until its next reset."""


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _month() -> str:
    return datetime.now().strftime("%Y-%m")


class QuotaStore:
    """Persisted usage counters with automatic calendar resets."""

    def __init__(self, path: Path, limits: dict[str, QuotaLimit] | None = None) -> None:
        self.path = Path(path)
        self.limits = limits if limits is not None else LIMITS
        self._monthly: dict[str, dict[str, Any]] = {}
        self._daily: dict[str, int] = {}
        self._cache: dict[str, Any] = {}
        self._load()

    # -- Persistence -----------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self._monthly = data.get("monthly") or {}
        today = _today()
        # Drop daily counters from previous days; they have reset anyway.
        self._daily = {
            key: count for key, count in (data.get("daily") or {}).items() if key.endswith(today)
        }
        self._cache = data.get("cache") or {}

    def _save(self) -> None:
        payload = {
            "monthly": self._monthly,
            "daily": self._daily,
            "cache": self._cache,
        }
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("Could not persist quota state %s: %s", self.path, exc)

    # -- Counters -----------------------------------------------------------------

    def _month_entry(self, name: str) -> dict[str, Any]:
        entry = self._monthly.setdefault(name, {"period": "", "count": 0})
        if entry.get("period") != _month():  # new month -> fresh budget
            entry["period"] = _month()
            entry["count"] = 0
        return entry

    def month_used(self, name: str) -> int:
        """Calls already made this calendar month for one bucket."""
        return int(self._month_entry(name)["count"])

    def daily_count(self, bucket_key: str) -> int:
        """Current count for one daily bucket (any caller-defined key)."""
        return int(self._daily.get(f"{bucket_key}#{_today()}", 0))

    def bump_daily(self, bucket_key: str, n: int = 1) -> None:
        self._daily[f"{bucket_key}#{_today()}"] = self.daily_count(bucket_key) + n

    # -- Enforcement ---------------------------------------------------------------

    def require(self, name: str) -> None:
        """Raise :class:`QuotaExceededError` if any cap for ``name`` is spent."""
        limit = self.limits.get(name)
        if limit is None:
            return  # untracked bucket - allow freely

        if limit.monthly is not None and self.month_used(name) >= limit.monthly:
            first_of_next = f"{datetime.now().year}-{int(datetime.now().month) + 1:02d}-01"
            raise QuotaExceededError(
                f"{limit.display or name}: monthly limit of {limit.monthly} calls "
                f"is used up ({self.month_used(name)}/{limit.monthly}). "
                f"It resets on {first_of_next}."
            )

        if limit.daily is not None and self.daily_count(name) >= limit.daily:
            raise QuotaExceededError(
                f"{limit.display or name}: daily limit of {limit.daily} calls "
                f"is used up ({limit.daily}/{limit.daily}). "
                f"It resets after {_today()}."
            )

    def record(self, name: str, n: int = 1) -> None:
        """Count successful calls against monthly and daily buckets."""
        entry = self._month_entry(name)
        entry["count"] = int(entry["count"]) + n
        self.bump_daily(name, n)
        self._save()

    # -- Small result cache (used for SweetTracker per-waybill dedupe) ---------------

    def get_cached(self, key: str) -> Any:
        return self._cache.get(key)

    def put_cached(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._save()


#: Process-wide stores keyed by resolved file path, so every skill shares
#: one counter set without passing the store around explicitly.
_STORES: dict[str, QuotaStore] = {}


def get_quota_store(path: Path) -> QuotaStore:
    """Return the shared store for one file path."""
    key = str(Path(path).resolve())
    if key not in _STORES:
        _STORES[key] = QuotaStore(Path(path))
    return _STORES[key]


def render_usage(store: QuotaStore) -> str:
    """Render current usage of every tracked bucket as clean text."""
    lines = ["API usage:"]
    any_tracked = False
    for name, limit in LIMITS.items():
        parts: list[str] = []
        if limit.monthly is not None:
            used = store.month_used(name)
            parts.append(f"this month {used:,}/{limit.monthly:,}")
        if limit.daily is not None:
            used = store.daily_count(name)
            parts.append(f"today {used:,}/{limit.daily:,}")
        if not parts:
            continue
        any_tracked = True
        lines.append(f"- {limit.display or name}: " + " | ".join(parts))
    if not any_tracked:
        lines.append("- all providers untouched today")
    return "\n".join(lines)
