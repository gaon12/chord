"""Tests for chord.context - what a skill knows about where it runs."""

from __future__ import annotations

import asyncio

import pytest

from chord.context import (
    channel_allows_age_restricted,
    current_channel,
    current_channel_context,
    reset_current_channel,
    set_current_channel,
)


def test_the_channel_id_is_readable_inside_a_turn():
    token = set_current_channel(42)
    try:
        assert current_channel() == 42
    finally:
        reset_current_channel(token)


def test_reading_a_channel_out_of_band_raises_rather_than_guessing():
    with pytest.raises(LookupError, match="No channel context"):
        current_channel()


def test_an_ordinary_channel_is_not_age_restricted():
    token = set_current_channel(42)
    try:
        assert channel_allows_age_restricted() is False
    finally:
        reset_current_channel(token)


def test_a_marked_channel_is_age_restricted():
    token = set_current_channel(42, nsfw=True)
    try:
        assert channel_allows_age_restricted() is True
        assert current_channel_context().nsfw is True
    finally:
        reset_current_channel(token)


def test_a_dm_counts_as_age_restricted():
    """Discord treats a one-to-one DM as age-gated by its nature."""
    token = set_current_channel(42, is_dm=True)
    try:
        assert channel_allows_age_restricted() is True
    finally:
        reset_current_channel(token)


def test_out_of_band_is_not_age_restricted():
    """Not knowing where you are is a reason to say no, not yes."""
    assert channel_allows_age_restricted() is False


def test_the_context_ends_with_the_turn():
    token = set_current_channel(42, nsfw=True)
    reset_current_channel(token)

    assert channel_allows_age_restricted() is False


async def test_concurrent_turns_do_not_share_a_channel():
    """One NSFW channel must not unlock an ordinary one answering beside it."""

    async def turn(channel_id: int, nsfw: bool) -> tuple[int, bool]:
        token = set_current_channel(channel_id, nsfw=nsfw)
        try:
            await asyncio.sleep(0)  # let the other turn interleave
            return current_channel(), channel_allows_age_restricted()
        finally:
            reset_current_channel(token)

    first, second = await asyncio.gather(turn(1, True), turn(2, False))

    assert first == (1, True)
    assert second == (2, False)
