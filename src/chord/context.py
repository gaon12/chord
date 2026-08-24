"""Per-invocation Discord context for skills.

Skills are context-free by design (the engine only passes tool
arguments), but a few of them - reminders especially - need to know
*where* the conversation is happening. The bot sets this contextvar
around each chat turn; skills read it through :func:`current_channel`.

If it is unset (e.g. a skill invoked outside the message flow), readers
raise a friendly error instead of guessing.
"""

from __future__ import annotations

from contextvars import ContextVar

_current_channel: ContextVar[int | None] = ContextVar("current_channel_id", default=None)


def set_current_channel(channel_id: int):
    """Set the channel for this invocation; returns a reset token."""
    return _current_channel.set(int(channel_id))


def reset_current_channel(token) -> None:
    _current_channel.reset(token)


def current_channel() -> int:
    """The channel id of the ongoing conversation.

    Raises:
        LookupError: when no channel is bound (skill used out of band).
    """
    channel_id = _current_channel.get()
    if channel_id is None:
        raise LookupError("No channel context - this action needs to run inside a chat.")
    return int(channel_id)
