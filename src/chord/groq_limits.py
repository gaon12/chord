"""Groq rate-limit headers, with their documented units kept explicit.

Request headers describe requests per day; token headers describe tokens per
minute. They do not report every possible quota. Missing values stay unknown,
and expired observations are never presented as fresh remaining capacity.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_SECONDS = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}


def duration_seconds(raw: str | None) -> float | None:
    """Read seconds or Groq durations such as 2m59.56s; reject invalid values."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        value = float(raw)
    except ValueError:
        parts = list(_DURATION_PART.finditer(raw))
        if not parts or "".join(part[0] for part in parts) != raw:
            return None
        value = sum(float(part[1]) * _SECONDS[part[2]] for part in parts)
    return value if math.isfinite(value) and value >= 0 else None


def _count(raw: str | None) -> int | None:
    if raw is None or not raw.strip().isdigit():
        return None
    return int(raw)


@dataclass
class GroqLimits:
    request_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset_at: float | None = None
    token_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_reset_at: float | None = None
    retry_at: float = 0.0

    def update(self, headers: Mapping[str, str], *, now: float | None = None) -> None:
        """Record one response. Header names are case insensitive."""
        now = time.monotonic() if now is None else now
        headers = {key.lower(): value for key, value in headers.items()}
        # A missing counter must not inherit an old observation: another
        # process can consume the organization's quota between our requests.
        self.request_limit = _count(headers.get("x-ratelimit-limit-requests"))
        self.requests_remaining = _count(headers.get("x-ratelimit-remaining-requests"))
        self.token_limit = _count(headers.get("x-ratelimit-limit-tokens"))
        self.tokens_remaining = _count(headers.get("x-ratelimit-remaining-tokens"))
        request_delay = duration_seconds(headers.get("x-ratelimit-reset-requests"))
        token_delay = duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        self.requests_reset_at = now + request_delay if request_delay is not None else None
        self.tokens_reset_at = now + token_delay if token_delay is not None else None

    def mark_rate_limited(self, headers: Mapping[str, str], *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.update(headers, now=now)
        headers = {key.lower(): value for key, value in headers.items()}
        retry = duration_seconds(headers.get("retry-after"))
        # An unreported limit may be RPM or TPD. Do not invent its bucket or
        # assume it resets tomorrow; use a short retry cooldown instead.
        self.retry_at = now + (retry if retry is not None else 60.0)

    def ready(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now < self.retry_at:
            return False
        for remaining, reset in (
            (self.requests_remaining, self.requests_reset_at),
            (self.tokens_remaining, self.tokens_reset_at),
        ):
            if remaining == 0 and reset is not None and now < reset:
                return False
        return True

    def describe(self, *, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now

        def bucket(remaining, limit, reset) -> str:
            if remaining is None:
                return "unknown"
            if reset is None or now >= reset:
                return "unknown (last observation expired or reset time missing)"
            total = str(limit) if limit is not None else "?"
            return f"{remaining}/{total} remaining; reset in {math.ceil(reset - now)}s"

        lines = [
            "RPD: " + bucket(self.requests_remaining, self.request_limit, self.requests_reset_at),
            "TPM: " + bucket(self.tokens_remaining, self.token_limit, self.tokens_reset_at),
        ]
        if now < self.retry_at:
            lines.append(f"Retry cooldown: {math.ceil(self.retry_at - now)}s")
        return " · ".join(lines)
