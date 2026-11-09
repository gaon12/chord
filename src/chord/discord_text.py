"""Clean incoming Discord text and format model replies."""

from __future__ import annotations

import logging
import re
from typing import Any

import discord

logger = logging.getLogger(__name__)

#: Matches user (<@123>, <@!123>) and role (<@&123>) mention tokens.
_MENTION_RE = re.compile(r"<@[!&]?\d+>")


def clean_message_text(content: str) -> str:
    """Strip mention tokens and tidy whitespace from a raw message."""
    cleaned = _MENTION_RE.sub(" ", content)
    return cleaned.strip()


#: Discord caps display names at 32 characters; reusing that cap keeps a
#: joke nickname from crowding out the actual question.
MAX_SPEAKER_NAME = 32


def speaker_name(author: Any) -> str:
    """Readable name for whoever sent a message.

    Prefers the per-server nickname (``display_name``) because that is
    the name the other people in the channel actually see; falls back to
    the account name, then to a placeholder.
    """
    raw = getattr(author, "display_name", None) or getattr(author, "name", None)
    cleaned = " ".join(str(raw or "").replace("[", "(").replace("]", ")").split())
    return cleaned[:MAX_SPEAKER_NAME] or "unknown"


def label_speaker(name: str, text: str) -> str:
    """Tag a message with its author so the model can tell people apart.

    A channel is a group chat, but every participant's words arrive as
    the same anonymous ``user`` role - so without a label the model
    reads a whole room as one person and answers accordingly ("you said
    earlier..." to someone who never said it). The OpenAI ``name`` field
    would be the tidier home for this, but plenty of compatible
    providers drop or reject it, so the label rides inside the content
    where nothing can strip it.

    It is a hint, not authentication: anyone can type the same shape
    into a message. Brackets inside nicknames are folded to parentheses
    so a forged label at least cannot look identical to a real one.
    """
    return f"[{name}]: {text}"


def build_reply_context(message: discord.Message) -> str:
    """Extract replied-to message content for LLM context."""
    ref = getattr(message, "reference", None)
    resolved = getattr(ref, "resolved", None)
    content = getattr(resolved, "content", None)
    author = getattr(resolved, "author", None)

    if not content or not content.strip() or not hasattr(author, "display_name"):
        return ""

    display_name = author.display_name
    text = clean_message_text(content)
    if len(text) > 500:
        text = text[:497] + "..."
    return f'[replying to {display_name}: "{text}"]\n'


def channel_is_nsfw(channel: Any) -> bool:
    """Whether Discord has this channel marked age-restricted.

    ``is_nsfw()`` exists on guild channels and threads but not on DMs,
    and a stray implementation could raise, so a missing or unhappy flag
    reads as "not marked" - the answer that keeps adult content out of
    an ordinary channel rather than into it.
    """
    check = getattr(channel, "is_nsfw", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception:  # noqa: BLE001 - an unreadable flag is not a marked channel
        logger.debug("Could not read the NSFW flag for channel %s", getattr(channel, "id", "?"))
        return False


#: Files listed per message. Someone dumping twenty screenshots is not
#: asking about all twenty, and each line costs prompt.
MAX_LISTED_ATTACHMENTS = 4


def build_attachment_context(message: discord.Message) -> str:
    """List what was uploaded, so the model knows it can reach it.

    Discord delivers uploads out of band: ``message.content`` for a
    posted screenshot is an empty string, so a model reading only the
    text has no idea an image exists, let alone where. Naming them puts
    the URL where a tool can be pointed at it - read_qr for a code,
    read_url for a text file - and turns "이거 읽어줘" from an impossible
    request into a fetchable one.
    """
    lines = []
    for item in (getattr(message, "attachments", None) or [])[:MAX_LISTED_ATTACHMENTS]:
        url = getattr(item, "url", None)
        if not url:
            continue
        kind = "image" if (getattr(item, "content_type", "") or "").startswith("image/") else "file"
        name = " ".join(str(getattr(item, "filename", "") or "file").split())[:80]
        lines.append(f"[attached {kind}: {name} {url}]")
    return "\n".join(lines) + "\n" if lines else ""


#: Tag names open models use to think out loud inside the answer body.
_REASONING_TAG = r"thought|thinking|think|reasoning|reflection"

#: A complete <thought>...</thought> block, tag-matched so a stray
#: </think> cannot close a <thought>.
_REASONING_BLOCK_RE = re.compile(
    rf"<(?P<tag>{_REASONING_TAG})\b[^>]*>.*?</(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)

#: An opening tag that never closes: the rest is scratch work.
_UNCLOSED_REASONING_RE = re.compile(
    rf"<(?:{_REASONING_TAG})\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)

#: A closing tag with no opener: everything before it is scratch work.
_ORPHAN_REASONING_END_RE = re.compile(
    rf"\A.*</(?:{_REASONING_TAG})\s*>",
    re.DOTALL | re.IGNORECASE,
)

#: Fenced code blocks are quoted verbatim - a snippet about <think> tags
#: is content, not reasoning.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def strip_reasoning_blocks(text: str) -> str:
    """Remove chain-of-thought the model leaked into its answer.

    Models like gemma-4-31b-it narrate their reasoning inline as
    ``<thought>...</thought>`` before the real reply, which Discord
    renders as a wall of first-person deliberation. Discord users want
    the answer; the thinking is controlled by ``REASONING_LEVEL``, not
    displayed.

    Text inside fenced code blocks is preserved untouched, and if
    stripping would leave nothing at all the original text is returned -
    a leaked thought still beats an empty message.
    """
    stripped = _strip_outside_code_fences(text)
    return stripped if stripped.strip() else text


def _strip_outside_code_fences(text: str) -> str:
    """Apply reasoning removal to everything except fenced code."""
    parts: list[str] = []
    cursor = 0
    for fence in _CODE_FENCE_RE.finditer(text):
        parts.append(_strip_reasoning(text[cursor : fence.start()]))
        parts.append(fence.group(0))
        cursor = fence.end()
    parts.append(_strip_reasoning(text[cursor:]))
    return "".join(parts)


def _strip_reasoning(text: str) -> str:
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _ORPHAN_REASONING_END_RE.sub("", text)
    return _UNCLOSED_REASONING_RE.sub("", text)


def format_reply(answer: str) -> str:
    """Post-process LLM output for cleaner Discord rendering.

    The LLM already produces markdown, so this only removes reasoning
    the model was not supposed to show.
    """
    return strip_reasoning_blocks(answer).strip()
