"""Reminder skills - schedule and list personal nudges.

* ``set_reminder``      - store a message to be delivered later in the
  same channel. Time can be given as relative minutes or an absolute
  datetime (ISO 8601 preferred, common formats accepted).
* ``list_reminders``    - show this channel's pending reminders.

Both read the current Discord channel from ``chord.context``, which the
bot binds around every chat turn.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from chord.context import current_channel
from chord.reminders import ReminderStore, utc_now
from chord.skills._http import SkillHTTPError
from chord.skills.base import Skill
from chord.skills.datetime_info import parse_datetime, resolve_timezone

DEFAULT_TIMEZONE = "Asia/Seoul"


class _ReminderBase(Skill):
    """Shared plumbing: settings access + channel binding + store."""

    def __init__(self, settings) -> None:
        self._settings = settings

    def _store(self) -> ReminderStore:
        return ReminderStore(self._settings.reminder_db_path)

    @staticmethod
    def _channel() -> int:
        try:
            return current_channel()
        except LookupError:
            raise SkillHTTPError(
                "Reminders need a channel - ask inside the chat you want to be reminded in."
            ) from None


def format_reminder_row(row) -> str:
    due = row.due.astimezone()
    return f"#{row.id} {due.strftime('%m-%d %H:%M')} - {row.text}"


class SetReminderSkill(_ReminderBase):
    name = "set_reminder"
    description = (
        "Schedule a reminder that the bot posts later into this same "
        "channel. Give EITHER 'in_minutes' OR 'at' (a date-time)."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What should the reminder say?",
            },
            "in_minutes": {
                "type": "integer",
                "description": "Minutes from now, e.g. 30.",
            },
            "at": {
                "type": "string",
                "description": ("Absolute time, e.g. '2026-08-24 18:00' (KST by default)."),
            },
        },
        "required": ["text"],
    }

    async def run(
        self,
        text: str,
        in_minutes: int | None = None,
        at: str = "",
    ) -> str:
        if not text.strip():
            raise SkillHTTPError("The reminder text is empty.")
        if (in_minutes is None) == (not at.strip()):
            raise SkillHTTPError("Give exactly one of 'in_minutes' or 'at'.")

        channel_id = self._channel()

        if in_minutes is not None:
            minutes = max(int(in_minutes), 1)
            due = utc_now() + timedelta(minutes=minutes)
        else:
            zone = resolve_timezone(DEFAULT_TIMEZONE)
            due = parse_datetime(at.strip(), zone)

        store = self._store()
        reminder_id = store.add(channel_id, due, text.strip())
        local_due = due.astimezone()
        return (
            f"Reminder #{reminder_id} set for "
            f'{local_due.strftime("%m-%d %H:%M")} - "{text.strip()}".'
        )


class ListRemindersSkill(_ReminderBase):
    name = "list_reminders"
    description = "List the pending reminders in this channel."

    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    async def run(self) -> str:
        channel_id = self._channel()
        rows = self._store().pending_for_channel(channel_id)
        if not rows:
            return "No pending reminders in this channel."
        lines = [f"{len(rows)} pending reminder(s):"]
        lines += [format_reminder_row(row) for row in rows]
        return "\n".join(lines)
