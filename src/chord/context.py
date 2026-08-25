"""Per-invocation Discord context for skills.

Skills are context-free by design (the engine only passes tool
arguments), but a few of them need to know *where* the conversation is
happening: reminders, to know which channel to post back into, and
anything age-restricted, to know whether this is a channel where that is
allowed. The bot sets this contextvar around each chat turn.

If it is unset (e.g. a skill invoked outside the message flow), readers
raise a friendly error or answer conservatively rather than guessing.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelContext:
    """What a skill may need to know about where it is running."""

    id: int

    #: Discord's own age-restriction flag on the channel.
    nsfw: bool = False

    #: DMs are one-to-one with a verified account, which is why Discord
    #: treats them as age-restricted by nature.
    is_dm: bool = False

    @property
    def age_restricted(self) -> bool:
        return self.nsfw or self.is_dm


_current_channel: ContextVar[ChannelContext | None] = ContextVar(
    "current_channel",
    default=None,
)


def set_current_channel(channel_id: int, *, nsfw: bool = False, is_dm: bool = False):
    """Set the channel for this invocation; returns a reset token."""
    return _current_channel.set(
        ChannelContext(id=int(channel_id), nsfw=bool(nsfw), is_dm=bool(is_dm))
    )


def reset_current_channel(token) -> None:
    _current_channel.reset(token)


def current_channel_context() -> ChannelContext:
    """Everything known about the ongoing conversation's channel.

    Raises:
        LookupError: when no channel is bound (skill used out of band).
    """
    context = _current_channel.get()
    if context is None:
        raise LookupError("No channel context - this action needs to run inside a chat.")
    return context


def current_channel() -> int:
    """The channel id of the ongoing conversation.

    Raises:
        LookupError: when no channel is bound (skill used out of band).
    """
    return current_channel_context().id


def channel_allows_age_restricted() -> bool:
    """Whether adult content may be posted where this turn is running.

    Discord requires age-restricted content to stay in channels marked
    as such (or DMs); posting it into an ordinary channel is what gets a
    server reported, not a matter of taste. Out of band - no channel
    bound at all - the answer is False, because the safe default when
    you cannot tell where you are is "not here".
    """
    context = _current_channel.get()
    return bool(context and context.age_restricted)
