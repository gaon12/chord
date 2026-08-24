"""Tests for reminder skills - context binding, time parsing, listing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chord.context import reset_current_channel, set_current_channel
from chord.skills._http import SkillHTTPError
from chord.skills.reminders import (
    ListRemindersSkill,
    SetReminderSkill,
    format_reminder_row,
)


def _settings(tmp_path):
    from chord.config import Settings

    return Settings(
        discord_token="t",
        openai_api_key="k",
        reminder_db_path=str(tmp_path / "chord.db"),
        _env_file=None,
    )


@pytest.fixture
def channel_ctx():
    token = set_current_channel(42)
    yield 42
    reset_current_channel(token)


# -- set_reminder -------------------------------------------------------------------


async def test_set_with_in_minutes(tmp_path, channel_ctx):
    skill = SetReminderSkill(_settings(tmp_path))
    result = await skill.run(text="check oven", in_minutes=30)

    assert "Reminder #1" in result
    assert "check oven" in result


async def test_set_with_absolute_time(tmp_path, channel_ctx):
    skill = SetReminderSkill(_settings(tmp_path))
    result = await skill.run(text="standup", at="2026-08-24 18:00")

    assert "Reminder #" in result
    assert "standup" in result


async def test_both_or_neither_time_source_raises(tmp_path, channel_ctx):
    skill = SetReminderSkill(_settings(tmp_path))

    with pytest.raises(SkillHTTPError, match="exactly one"):
        await skill.run(text="x")  # neither

    with pytest.raises(SkillHTTPError, match="exactly one"):
        await skill.run(text="x", in_minutes=5, at="2026-08-25")  # both


async def test_no_channel_context_raises(tmp_path):
    skill = SetReminderSkill(_settings(tmp_path))

    with pytest.raises(SkillHTTPError, match="channel"):
        await skill.run(text="hi", in_minutes=5)


async def test_empty_text_raises(tmp_path, channel_ctx):
    skill = SetReminderSkill(_settings(tmp_path))

    with pytest.raises(SkillHTTPError, match="empty"):
        await skill.run(text="", in_minutes=10)


# -- list_reminders -------------------------------------------------------------------


async def test_list_shows_pending(tmp_path, channel_ctx):
    setter = SetReminderSkill(_settings(tmp_path))
    lister = ListRemindersSkill(_settings(tmp_path))

    await setter.run(text="buy milk", in_minutes=60)
    result = await lister.run()

    assert "1 pending" in result
    assert "buy milk" in result


async def test_list_empty_returns_message(tmp_path, channel_ctx):
    lister = ListRemindersSkill(_settings(tmp_path))
    result = await lister.run()
    assert "No pending" in result


# -- format_reminder_row ----------------------------------------------------------------


def test_format_row_includes_id_and_text():
    from chord.reminders import Reminder

    due = datetime(2026, 8, 24, 15, 0, tzinfo=UTC)
    row = Reminder(id=3, channel_id=42, due=due, text="call mom")
    formatted = format_reminder_row(row)
    assert "#3" in formatted
    assert "call mom" in formatted
