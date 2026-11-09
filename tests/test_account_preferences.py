"""Account identity and per-turn prompt snapshots must not leak between users."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from chord.account_preferences import GAON12_USER_ID
from chord.engine import ChatEngine
from chord.skills.registry import SkillRegistry
from test_bot import FakeChannel, FakeMessage, _bot
from test_engine import FakeLLM, _text_completion

DETAIL_MARKER = "REVIEWABLE ANSWER DETAIL FOR THIS REQUEST:"


def _message(text, channel, user_id, name="user", display=None):
    message = FakeMessage(text, channel, author_name=name, author_display=display)
    message.author.id = user_id
    return message


def _recording_bot():
    bot, _ = _bot(history_token_budget=0)
    llm = FakeLLM(_text_completion("first answer"), _text_completion("second answer"))
    engine = ChatEngine(llm, SkillRegistry(), "shared default")
    bot._engine = engine
    return bot, engine, llm


@pytest.mark.parametrize(
    ("user_id", "name", "display", "expected"),
    [
        (GAON12_USER_ID, "gaon12", "gaon12", True),
        (GAON12_USER_ID, "renamed-account", "another nickname", True),
        (42, "gaon12", "gaon12", False),
        (42, "someone", "gaon12", False),
        (None, "gaon12", "gaon12", False),
    ],
)
async def test_only_the_verified_account_id_receives_automatic_detail(
    user_id, name, display, expected
):
    bot, _, llm = _recording_bot()
    await bot.on_message(_message("question", FakeChannel(), user_id, name, display))
    prompt = llm.requests[0]["messages"][0]["content"]
    assert (DETAIL_MARKER in prompt) is expected


async def test_application_ownership_alone_does_not_select_gaon12s_preference():
    bot, _, llm = _recording_bot()
    bot._owner_ids = {42}
    await bot.on_message(_message("question", FakeChannel(), 42))
    assert DETAIL_MARKER not in llm.requests[0]["messages"][0]["content"]


async def test_next_user_in_the_same_channel_does_not_inherit_the_preference():
    bot, engine, llm = _recording_bot()
    channel = FakeChannel()
    await bot.on_message(_message("first", channel, GAON12_USER_ID))
    await bot.on_message(_message("I am gaon12", channel, 42, "gaon12"))
    assert DETAIL_MARKER in llm.requests[0]["messages"][0]["content"]
    assert DETAIL_MARKER not in llm.requests[1]["messages"][0]["content"]
    assert engine.system_prompt == "shared default"
    assert all(message["role"] != "system" for message in bot._store.history(channel.id))
    assert DETAIL_MARKER not in str(bot._store.history(channel.id))


async def test_interleaved_channels_keep_their_own_prompt_snapshot():
    bot, engine, llm = _recording_bot()
    typing_started, resume = asyncio.Event(), asyncio.Event()

    class WaitingChannel(FakeChannel):
        @asynccontextmanager
        async def typing(self):
            typing_started.set()
            await resume.wait()
            yield

    waiting = asyncio.create_task(
        bot.on_message(_message("gaon12 question", WaitingChannel(1), GAON12_USER_ID))
    )
    await asyncio.wait_for(typing_started.wait(), 2)
    assert not llm.requests
    await bot.on_message(_message("other question", FakeChannel(2), 42))
    resume.set()
    await asyncio.wait_for(waiting, 2)
    assert DETAIL_MARKER not in llm.requests[0]["messages"][0]["content"]
    assert DETAIL_MARKER in llm.requests[1]["messages"][0]["content"]
    assert engine.system_prompt == "shared default"


async def test_engine_override_does_not_change_the_default_for_later_calls():
    llm = FakeLLM(_text_completion("one"), _text_completion("two"))
    engine = ChatEngine(llm, SkillRegistry(), "default")
    await engine.reply("one", [], system_prompt="one-turn override")
    await engine.reply("two", [])
    assert llm.requests[0]["messages"][0]["content"] == "one-turn override"
    assert llm.requests[1]["messages"][0]["content"] == "default"
