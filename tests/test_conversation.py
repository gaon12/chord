"""Tests for chord.conversation - per-channel chat history."""

from __future__ import annotations

from chord.conversation import ConversationStore


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_history_starts_empty_per_channel():
    store = ConversationStore()
    assert store.history(1) == []
    assert store.history(2) == []  # channels are independent


def test_append_and_read_back():
    store = ConversationStore()
    store.append(1, _msg("user", "hi"), _msg("assistant", "hello"))
    assert store.history(1) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_old_messages_are_trimmed():
    store = ConversationStore(max_messages=4)
    for i in range(10):
        store.append(1, _msg("user", f"m{i}"))

    history = store.history(1)
    assert len(history) <= 4
    assert history[-1]["content"] == "m9"  # newest always survives


def test_reset_clears_only_that_channel():
    store = ConversationStore()
    store.append(1, _msg("user", "one"))
    store.append(2, _msg("user", "two"))

    store.reset(1)

    assert store.history(1) == []
    assert store.history(2) == [{"role": "user", "content": "two"}]
