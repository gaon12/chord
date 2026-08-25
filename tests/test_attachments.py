"""Tests for chord.attachments - files that ride beside the conversation."""

from __future__ import annotations

import asyncio

from chord.attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    attach,
    collected,
    reset_attachments,
    start_collecting,
)


def test_nothing_is_collected_outside_a_turn():
    """A skill run from a script or a test has nowhere to put a file."""
    assert attach("chart.png", b"data") is False
    assert collected() == []


def test_attached_files_come_back_in_order():
    token = start_collecting()
    try:
        assert attach("first.png", b"a") is True
        assert attach("second.png", b"b") is True

        names = [item.filename for item in collected()]
        assert names == ["first.png", "second.png"]
    finally:
        reset_attachments(token)


def test_the_collection_ends_with_the_turn():
    token = start_collecting()
    attach("chart.png", b"data")
    reset_attachments(token)

    assert collected() == []


def test_collected_hands_out_a_copy():
    """A caller emptying its list must not empty the collection."""
    token = start_collecting()
    try:
        attach("chart.png", b"data")
        collected().clear()

        assert len(collected()) == 1
    finally:
        reset_attachments(token)


def test_a_runaway_tool_loop_cannot_flood_the_channel():
    token = start_collecting()
    try:
        results = [attach(f"{index}.png", b"x") for index in range(MAX_ATTACHMENTS + 3)]

        assert results[:MAX_ATTACHMENTS] == [True] * MAX_ATTACHMENTS
        assert not any(results[MAX_ATTACHMENTS:])
        assert len(collected()) == MAX_ATTACHMENTS
    finally:
        reset_attachments(token)


def test_an_oversized_file_is_refused_rather_than_sent(caplog):
    """Discord would reject it; better to say so than to lose the reply."""
    token = start_collecting()
    try:
        with caplog.at_level("WARNING"):
            taken = attach("huge.png", b"x" * (MAX_ATTACHMENT_BYTES + 1))

        assert taken is False
        assert collected() == []
        assert "over the" in caplog.text
    finally:
        reset_attachments(token)


async def test_concurrent_turns_do_not_see_each_others_files():
    """Two channels answering at once must not swap charts."""

    async def turn(name: str) -> list[str]:
        token = start_collecting()
        try:
            attach(f"{name}.png", b"data")
            await asyncio.sleep(0)  # let the other turn interleave
            return [item.filename for item in collected()]
        finally:
            reset_attachments(token)

    first, second = await asyncio.gather(turn("alice"), turn("bob"))

    assert first == ["alice.png"]
    assert second == ["bob.png"]
