"""Tests for chord.compaction - folding old turns into a digest."""

from __future__ import annotations

from chord.compaction import (
    SUMMARY_ACK,
    SUMMARY_PREFIX,
    HistoryCompactor,
    estimate_history_tokens,
    estimate_message_tokens,
    estimate_tokens,
    render_transcript,
    split_for_compaction,
    summary_messages,
)
from fakes import FakeLLM


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _turn(n: int, size: int = 200) -> list[dict]:
    """One plain question/answer turn of roughly `size` characters."""
    return [_user(f"[alice]: q{n} " + "x" * size), _assistant(f"a{n} " + "y" * size)]


# -- Token estimation ---------------------------------------------------------------


def test_ascii_text_is_about_four_characters_per_token():
    assert estimate_tokens("x" * 400) == 100


def test_korean_text_costs_far_more_than_its_character_count_suggests():
    """A chars/token ratio tuned on ASCII under-reads Korean threefold."""
    assert estimate_tokens("안녕하세요" * 20) == 100


def test_message_estimate_covers_tool_calls_not_just_content():
    """A tool-call message has no content at all, but is not free."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}}
        ],
    }
    assert estimate_message_tokens(message) > 10


def test_history_estimate_adds_its_messages_up():
    history = [_user("hi"), _assistant("hello")]
    assert estimate_history_tokens(history) == sum(estimate_message_tokens(m) for m in history)


# -- Choosing what to summarize -------------------------------------------------------


def test_split_cuts_on_a_turn_boundary():
    history = _turn(0) + _turn(1) + _turn(2)

    older, recent = split_for_compaction(history, keep_tokens=80)

    assert recent[0]["role"] == "user"
    assert older + recent == history


def test_split_keeps_the_newest_turn_even_when_it_busts_the_budget():
    """Answering the question just asked beats respecting the budget."""
    history = _turn(0) + _turn(1)

    older, recent = split_for_compaction(history, keep_tokens=1)

    assert recent == _turn(1)
    assert older == _turn(0)


def test_split_never_orphans_a_tool_result():
    history = [
        _user("[alice]: q0"),
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0"}]},
        {"role": "tool", "tool_call_id": "c0", "content": "result"},
        _assistant("a0"),
        _user("[alice]: q1"),
        _assistant("a1"),
    ]

    older, recent = split_for_compaction(history, keep_tokens=1)

    assert recent[0]["role"] == "user"
    assert not any(m["role"] == "tool" for m in older) or older[0]["role"] == "user"


def test_split_declines_when_the_whole_history_is_one_turn():
    history = _turn(0)

    older, recent = split_for_compaction(history, keep_tokens=1)

    assert older == []
    assert recent == history


# -- Transcript rendering ------------------------------------------------------------


def test_transcript_keeps_speaker_labels():
    text = render_transcript([_user("[alice]: 안녕"), _user("[bob]: 나도")])
    assert "[alice]: 안녕" in text
    assert "[bob]: 나도" in text


def test_transcript_truncates_bulky_tool_results():
    history = [{"role": "tool", "tool_call_id": "c", "content": "z" * 5000}]

    text = render_transcript(history)

    assert len(text) < 500
    assert text.endswith("...")


def test_transcript_records_which_tool_was_called():
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "function": {"name": "get_weather", "arguments": "{}"}}],
        }
    ]
    assert "get_weather" in render_transcript(history)


# -- Compaction ---------------------------------------------------------------------


def _compactor(reply: str = "- alice asked about the weather", budget: int = 100):
    llm = FakeLLM(reply)
    return HistoryCompactor(llm, token_budget=budget), llm


async def test_short_history_is_left_alone():
    compactor, llm = _compactor()

    assert await compactor.compact([_user("[alice]: hi")]) is None
    assert not llm.calls  # and it costs nothing to find that out


async def test_empty_history_is_left_alone():
    compactor, _ = _compactor()
    assert await compactor.compact([]) is None


async def test_zero_budget_disables_compaction():
    compactor = HistoryCompactor(FakeLLM(), token_budget=0)

    assert compactor.enabled is False
    assert await compactor.compact(_turn(0) + _turn(1)) is None


async def test_no_llm_disables_compaction():
    compactor = HistoryCompactor(None, token_budget=100)

    assert compactor.enabled is False
    assert await compactor.compact(_turn(0) + _turn(1)) is None


async def test_long_history_is_folded_into_a_digest():
    compactor, llm = _compactor()
    history = _turn(0) + _turn(1) + _turn(2)

    result = await compactor.compact(history)

    assert result is not None
    consumed, replacement = result
    # The oldest turns went in, the newest stayed out of the digest.
    assert consumed == history[: len(consumed)]
    assert consumed != history
    assert replacement[0]["content"].startswith(SUMMARY_PREFIX)
    assert "alice asked about the weather" in replacement[0]["content"]
    assert replacement[1] == {"role": "assistant", "content": SUMMARY_ACK}


async def test_the_digest_is_written_without_tools():
    """The summarizer only reads; letting it call tools would be a bug."""
    compactor, llm = _compactor()

    await compactor.compact(_turn(0) + _turn(1) + _turn(2))

    assert llm.calls[0]["tools"] is None


async def test_the_transcript_sent_for_summarizing_holds_the_old_turns():
    compactor, llm = _compactor()

    await compactor.compact(_turn(0) + _turn(1) + _turn(2))

    transcript = llm.calls[0]["messages"][-1]["content"]
    assert "q0" in transcript
    assert "q2" not in transcript  # the newest turn is still in history


async def test_an_earlier_digest_is_folded_in_rather_than_nested():
    compactor, llm = _compactor()
    history = summary_messages("- 이전 요약") + _turn(1) + _turn(2)

    await compactor.compact(history)

    transcript = llm.calls[0]["messages"][-1]["content"]
    assert "이전 요약" in transcript


async def test_an_empty_summary_leaves_the_history_untouched():
    """A digest that says nothing would erase the conversation."""
    compactor, _ = _compactor(reply="   ")

    assert await compactor.compact(_turn(0) + _turn(1) + _turn(2)) is None


async def test_compaction_actually_shrinks_the_history():
    compactor, _ = _compactor()
    history = _turn(0) + _turn(1) + _turn(2)

    consumed, replacement = await compactor.compact(history)
    compacted = replacement + history[len(consumed) :]

    assert estimate_history_tokens(compacted) < estimate_history_tokens(history)
    assert compacted[0]["role"] == "user"  # still a valid history to send
