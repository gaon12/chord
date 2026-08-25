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


# -- Trimming on turn boundaries --------------------------------------------------


def _turn(n: int) -> list[dict]:
    """One tool-calling turn: user -> assistant(tool_calls) -> tool -> assistant."""
    return [
        {"role": "user", "content": f"q{n}"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{n}"}]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": "result"},
        {"role": "assistant", "content": f"a{n}"},
    ]


def test_trim_never_leaves_a_tool_message_first():
    """An orphaned tool result is a hard 400 from the provider."""
    store = ConversationStore(max_messages=6)

    for n in range(5):
        store.append(1, *_turn(n))
        assert store.history(1)[0]["role"] == "user"


def test_trim_drops_whole_turns():
    store = ConversationStore(max_messages=6)

    store.append(1, *_turn(0))
    store.append(1, *_turn(1))

    history = store.history(1)
    assert [m.get("content") for m in history if m["role"] == "user"] == ["q1"]
    assert len(history) == 4  # the older turn went entirely


def test_history_below_the_cap_is_untouched():
    store = ConversationStore(max_messages=10)
    store.append(1, *_turn(0))

    assert len(store.history(1)) == 4


def test_a_single_oversized_turn_is_kept_rather_than_broken():
    """Better to exceed the cap briefly than to send an invalid history."""
    store = ConversationStore(max_messages=2)

    store.append(1, *_turn(0))

    assert len(store.history(1)) == 4
    assert store.history(1)[0]["role"] == "user"


def test_next_turn_start_finds_the_boundary():
    from chord.conversation import next_turn_start

    history = _turn(0) + _turn(1)

    assert next_turn_start(history, 1) == 4
    assert next_turn_start(history, 4) == 4
    assert next_turn_start(history, 5) == 0  # no boundary left -> keep all


# -- Compaction hand-off ----------------------------------------------------------


def test_replace_prefix_swaps_the_oldest_messages():
    store = ConversationStore()
    first, second = _msg("user", "old"), _msg("user", "new")
    store.append(1, first, second)

    applied = store.replace_prefix(1, [first], [_msg("user", "summary")])

    assert applied is True
    assert store.history(1) == [{"role": "user", "content": "summary"}, second]


def test_replace_prefix_declines_when_the_channel_moved_on():
    """Compaction runs after the reply, so the history can change under it."""
    store = ConversationStore()
    first = _msg("user", "old")
    store.append(1, first)
    summarized = [first]
    store.reset(1)  # e.g. someone ran /reset while the digest was written
    store.append(1, _msg("user", "fresh start"))

    applied = store.replace_prefix(1, summarized, [_msg("user", "summary")])

    assert applied is False
    assert store.history(1) == [{"role": "user", "content": "fresh start"}]


def test_replace_prefix_on_an_unknown_channel_is_a_no_op():
    store = ConversationStore()
    assert store.replace_prefix(42, [_msg("user", "x")], []) is False


def test_replace_prefix_keeps_messages_appended_meanwhile():
    store = ConversationStore()
    first = _msg("user", "old")
    store.append(1, first)
    store.append(1, _msg("user", "arrived during the summary"))

    store.replace_prefix(1, [first], [_msg("user", "summary")])

    assert [m["content"] for m in store.history(1)] == [
        "summary",
        "arrived during the summary",
    ]
