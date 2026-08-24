"""Tests for the quota store - counters, resets, persistence, cache."""

from __future__ import annotations

import json

import pytest

from chord.skills._http import SkillHTTPError
from chord.skills._quota import LIMITS, QuotaExceededError, QuotaStore


def _limits(**overrides) -> dict:
    limits = {
        "sweettracker": LIMITS["sweettracker"],
        "weatherapi": LIMITS["weatherapi"],
    }
    limits.update(overrides)
    return limits


def test_monthly_cap_blocks_and_reports_reset(tmp_path):
    store = QuotaStore(tmp_path / "usage.json", _limits())
    for _ in range(100):
        store.require("sweettracker")
        store.record("sweettracker")

    with pytest.raises(QuotaExceededError) as excinfo:
        store.require("sweettracker")

    message = str(excinfo.value)
    assert "SweetTracker" in message
    assert "100" in message
    assert "resets on" in message.lower()


def test_monthly_counter_resets_on_new_period(tmp_path):
    path = tmp_path / "usage.json"
    store = QuotaStore(path, _limits())
    store.record("sweettracker")
    # Simulate a stored counter from last month.
    entry = store._month_entry("sweettracker")
    entry["period"] = "2000-01"
    entry["count"] = 999

    store.require("sweettracker")  # must not raise: new period means fresh
    assert store.month_used("sweettracker") == 0


def test_daily_cap_blocks_until_next_day(tmp_path):
    store = QuotaStore(tmp_path / "usage.json", {"kma": LIMITS["kma"]})
    for _ in range(1_000):
        store.bump_daily("kma")
    store._save()

    with pytest.raises(QuotaExceededError, match="daily limit"):
        store.require("kma")


def test_unknown_bucket_is_untracked_but_countable(tmp_path):
    store = QuotaStore(tmp_path / "usage.json", {})
    store.require("whatever")  # no cap -> never raises
    store.record("whatever", n=3)
    assert store.month_used("whatever") == 3


def test_persistence_roundtrip_across_instances(tmp_path):
    path = tmp_path / "usage.json"
    first = QuotaStore(path, _limits())
    first.record("weatherapi", n=7)

    second = QuotaStore(path, _limits())
    assert second.month_used("weatherapi") == 7


def test_stale_daily_buckets_are_pruned_on_load(tmp_path):
    path = tmp_path / "usage.json"
    payload = {
        "monthly": {},
        "daily": {"opensky#1999-01-01": 55},
        "cache": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    store = QuotaStore(path)
    assert all(not key.startswith("opensky#1999") for key in store._daily)


def test_quota_error_is_skill_http_error():
    """Skills' existing fallback paths catch SkillHTTPError."""
    assert issubclass(QuotaExceededError, SkillHTTPError)


def test_cache_roundtrip(tmp_path):
    store = QuotaStore(tmp_path / "usage.json")
    store.put_cached("sweettracker#123", [{"status": "delivered"}])

    other = QuotaStore(tmp_path / "usage.json")
    assert other.get_cached("sweettracker#123") == [{"status": "delivered"}]
