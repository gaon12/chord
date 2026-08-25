"""Files a skill produced while answering one message.

A skill's return value is text, and that is right: it goes to the model,
which reads it and writes the reply. It is the wrong channel for a 50 KB
PNG - base64 in a tool result would be tokens spent on something the
model cannot see anyway, and most providers would refuse the request
long before that.

So images travel beside the conversation rather than through it. The bot
opens a collection around each turn, a skill drops files into it while it
runs, and the bot attaches whatever it finds to the reply it sends. The
model only ever learns "a chart is attached", in the skill's own text.

Same contextvar idiom as :mod:`chord.context`: each message is handled in
its own task, so the collection is per-turn without any locking, and a
skill called out of band (a test, a script) simply finds nothing
collecting and is told its file was not taken.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Upload ceiling. Discord's own limit is 10 MB on an unboosted server;
#: staying under it means a rejected upload is our bug, not a surprise.
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024

#: Per turn. More than a handful of images in one answer is a tool loop
#: gone wrong, not something anybody asked for.
MAX_ATTACHMENTS = 4


@dataclass(frozen=True)
class Attachment:
    """One file to send along with the reply."""

    filename: str
    data: bytes


_attachments: ContextVar[list[Attachment] | None] = ContextVar(
    "turn_attachments",
    default=None,
)


def start_collecting():
    """Begin collecting files for this turn; returns a reset token."""
    return _attachments.set([])


def reset_attachments(token) -> None:
    """End the collection opened by :func:`start_collecting`."""
    _attachments.reset(token)


def attach(filename: str, data: bytes) -> bool:
    """Offer a file to the reply being written.

    Returns:
        True when the file will be sent. False is not an error - it
        means nothing is collecting (the skill ran outside a chat turn)
        or a limit was reached - and callers should say so in their text
        rather than promising an image that will not arrive.
    """
    files = _attachments.get()
    if files is None:
        logger.debug("No turn is collecting; dropping attachment %s.", filename)
        return False
    if len(files) >= MAX_ATTACHMENTS:
        logger.warning(
            "Already holding %d attachments this turn; dropping %s.",
            MAX_ATTACHMENTS,
            filename,
        )
        return False
    if len(data) > MAX_ATTACHMENT_BYTES:
        logger.warning(
            "Attachment %s is %d bytes, over the %d-byte limit; dropping it.",
            filename,
            len(data),
            MAX_ATTACHMENT_BYTES,
        )
        return False

    files.append(Attachment(filename, data))
    return True


def collected() -> list[Attachment]:
    """Everything attached during this turn, oldest first."""
    return list(_attachments.get() or ())
