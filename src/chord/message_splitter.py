"""Fit replies into Discord messages while keeping code fences balanced."""

from __future__ import annotations

import re

DISCORD_MESSAGE_LIMIT = 2000
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split paragraphs and reopen long code blocks in each message.

    A message must be readable on its own. Long blocks keep their language
    label and get a closing fence in every chunk. Plain text prefers newline
    and space boundaries, but an unusually long word is split when necessary.
    """
    if limit < 16:
        raise ValueError("The message limit must be at least 16 characters.")
    text = text.strip()
    if not text:
        return ["(empty reply)"]
    chunks: list[str] = []
    for segment in _segments(text):
        fence = _FENCE.match(segment.split("\n", 1)[0])
        pieces = _split_code(segment, fence, limit) if fence else _split_plain(segment, limit)
        for piece in pieces:
            if chunks and len(chunks[-1]) + 2 + len(piece) <= limit:
                chunks[-1] += "\n\n" + piece
            else:
                chunks.append(piece)
    return chunks


def _segments(text: str) -> list[str]:
    segments: list[str] = []
    lines: list[str] = []
    marker: str | None = None

    def flush() -> None:
        if lines:
            segments.append("\n".join(lines))
            lines.clear()

    for line in text.split("\n"):
        fence = _FENCE.match(line)
        if marker is not None:
            lines.append(line)
            if (
                fence
                and fence[1][0] == marker[0]
                and len(fence[1]) >= len(marker)
                and not fence[2].strip()
            ):
                flush()
                marker = None
        elif fence:
            flush()
            marker = fence[1]
            lines.append(line)
        elif line.strip():
            lines.append(line)
        else:
            flush()
    flush()
    return segments


def _split_plain(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def _split_code(text: str, fence: re.Match, limit: int) -> list[str]:
    opening, _, body = text.partition("\n")
    marker = fence[1]
    last_line = body.rsplit("\n", 1)[-1]
    closing = _FENCE.fullmatch(last_line)
    if (
        closing
        and closing[1][0] == marker[0]
        and len(closing[1]) >= len(marker)
        and not closing[2].strip()
    ):
        body = body[: -len(last_line)].removesuffix("\n")

    # Extremely long fence markers or language labels must not consume the
    # whole message. Such malformed input is safer to show as plain text.
    room = limit - len(opening) - len(marker) - 2
    if room < 1:
        return _split_plain(text, limit)
    if len(body) <= room:
        return [f"{opening}\n{body}\n{marker}"]

    pieces: list[str] = []
    while body:
        cut = min(len(body), room)
        if cut < len(body):
            newline = body.rfind("\n", 0, cut)
            if newline >= 0:
                cut = newline + 1
        pieces.append(f"{opening}\n{body[:cut]}\n{marker}")
        body = body[cut:]
    return pieces
