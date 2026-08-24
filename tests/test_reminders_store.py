"""Tests for chord.reminders - SQLite-backed scheduling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chord.reminders import ReminderStore


def _now():
    return datetime.now(UTC)


def test_add_and_due(tmp_path):
    store = ReminderStore(tmp_path / "test.db")
    past = _now() - timedelta(minutes=1)

    store.add(channel_id=42, due=past, text="check oven")
    due = store.due()

    assert len(due) == 1
    assert due[0].channel_id == 42
    assert due[0].text == "check oven"


def test_future_reminder_not_due(tmp_path):
    store = ReminderStore(tmp_path / "test.db")
    future = _now() + timedelta(hours=3)

    store.add(channel_id=42, due=future, text="meeting")
    assert store.due() == []


def test_mark_done_excludes_from_due(tmp_path):
    store = ReminderStore(tmp_path / "test.db")
    rid = store.add(channel_id=42, due=_now(), text="done deal")
    store.mark_done(rid)
    assert store.due() == []


def test_pending_for_channel_filters_and_sorts(tmp_path):
    store = ReminderStore(tmp_path / "test.db")
    soon = _now() + timedelta(minutes=5)
    later = _now() + timedelta(hours=1)

    store.add(channel_id=99, due=later, text="second")
    store.add(channel_id=42, due=soon, text="first-42")
    store.add(channel_id=42, due=_now() + timedelta(minutes=30), text="third-42")

    pending = store.pending_for_channel(42)
    assert [r.text for r in pending] == ["first-42", "third-42"]
