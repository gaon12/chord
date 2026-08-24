"""Tests for chord.bot - helpers plus on_message flow with fake objects."""

from __future__ import annotations

import contextlib

import quota_helpers  # noqa: F401  (conftest already isolates quota store)
from chord.bot import (
    ChordBot,
    build_reply_context,
    clean_message_text,
    format_reply,
    split_message,
)
from chord.config import Settings

# -- Stub classes -------------------------------------------------------------------


class StubEngine:
    """Records calls and returns a canned answer."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []
        self.system_prompt_value = ""

    @property
    def system_prompt(self):
        return self.system_prompt_value

    @system_prompt.setter
    def system_prompt(self, value):
        self.system_prompt_value = value

    async def reply(self, user_text, history):
        self.calls.append((user_text, history))
        return ("answer!", [{"role": "user", "content": user_text}])


class FakeChannel:
    """Just enough of discord.TextChannel: typing() and send()."""

    def __init__(self, channel_id: int = 100):
        self.id = channel_id
        self.sent: list[str] = []

    @contextlib.asynccontextmanager
    async def typing(self):
        yield None

    async def send(self, content: str):
        self.sent.append(content)


class FakeAuthor:
    def __init__(self, name="user", is_bot=False, display_name=None):
        self.name = name
        self.display_name = display_name or name
        self.bot = is_bot


class FakeMessage:
    """Minimal discord.Message stand-in for on_message testing."""

    def __init__(
        self,
        content: str,
        channel,
        *,
        author_bot: bool = False,
        author_name="user",
        mentions=(),
        guild=None,
        reference=None,
    ):
        self.content = content
        self.channel = channel
        self.author = FakeAuthor(name=author_name, is_bot=author_bot)
        self.mentions = list(mentions)
        self.guild = guild
        self.reference = reference


class FakeUser:
    """Stands in for a Discord user (bot or human)."""

    def __init__(self, user_id: int = 999):
        self.id = user_id


def _bot() -> tuple[ChordBot, StubEngine]:
    settings = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
    )
    engine = StubEngine()
    bot = ChordBot(settings=settings, engine=engine)
    bot._me_id = 999
    return bot, engine


# -- clean_message_text -------------------------------------------------------------


def test_clean_message_text_removes_mentions():
    assert clean_message_text("<@123456789> what time is it?") == "what time is it?"


def test_clean_message_text_handles_nicknames_and_extra_space():
    assert clean_message_text("<@!42>  hi   there ") == "hi there"


def test_clean_message_text_keeps_regular_text():
    assert clean_message_text("plain text") == "plain text"


# -- split_message --------------------------------------------------------------------


def test_split_short_text_unchanged():
    assert split_message("hello") == ["hello"]


def test_split_empty_text_returns_placeholder():
    assert split_message("   ") == ["(empty reply)"]


def test_split_breaks_at_paragraph_boundaries():
    paragraphs = ["a" * 900, "b" * 900, "c" * 900]
    chunks = split_message("\n\n".join(paragraphs))
    assert len(chunks) == 2
    assert all(len(c) <= 2000 for c in chunks)


def test_split_hard_slices_giant_single_word():
    giant = "x" * 5000
    chunks = split_message(giant)
    assert len(chunks) >= 3
    assert "".join(chunks) == giant


def test_split_preserves_markdown_blocks():
    text = "```\ncode line\n```"
    chunks = split_message(text)
    assert len(chunks) == 1
    assert "```" in chunks[0]


# -- build_reply_context ---------------------------------------------------------------


class FakeRefMessage:
    def __init__(self, content, author_name="someone"):
        self.content = content
        self.author = FakeAuthor(name=author_name, display_name=author_name)


def test_build_reply_context_with_content():
    msg = FakeMessage("<@999> what does this mean?", FakeChannel())
    msg.reference = type("Ref", (), {})()
    msg.reference.resolved = FakeRefMessage("original question about quantum physics")

    result = build_reply_context(msg)

    assert "[replying to someone:" in result
    assert "quantum physics" in result


def test_build_reply_context_no_reference():
    assert build_reply_context(FakeMessage("hi", FakeChannel())) == ""


def test_build_reply_context_truncates_long_messages():
    long_content = "y" * 1000
    msg = FakeMessage("<@999> explain", FakeChannel())
    msg.reference = type("Ref", (), {})()
    msg.reference.resolved = FakeRefMessage(long_content)

    result = build_reply_context(msg)
    assert len(result) < 600  # truncated


# -- format_reply ------------------------------------------------------------------------


def test_format_reply_is_passthrough_for_simple_text():
    assert format_reply("hello") == "hello"


# -- on_message flow ----------------------------------------------------------------------


async def test_replies_when_mentioned_in_guild():
    bot, engine = _bot()
    channel = FakeChannel()
    msg = FakeMessage("<@999> hello", channel, mentions=[FakeUser(999)], guild="g")

    await bot.on_message(msg)

    assert engine.calls[0][0] == "hello"
    assert channel.sent == ["answer!"]


async def test_ignores_messages_without_mention():
    bot, engine = _bot()
    channel = FakeChannel()
    await bot.on_message(FakeMessage("just chatting", channel, guild="g"))
    assert not engine.calls
    assert not channel.sent


async def test_answers_direct_messages_without_mention():
    bot, engine = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("dm text", channel))  # no guild -> DM

    assert engine.calls[0][0] == "dm text"


async def test_ignores_other_bots():
    bot, engine = _bot()
    channel = FakeChannel()
    other = FakeUser()
    msg = FakeMessage("<@1> hi", channel, author_bot=True, mentions=[other], guild="g")

    await bot.on_message(msg)

    assert not engine.calls


async def test_mention_only_message_shows_help():
    bot, _ = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999>", channel, mentions=[FakeUser(999)]))

    assert "chat with me" in channel.sent[0]


async def test_engine_failure_sends_friendly_error():
    bot, engine = _bot()
    channel = FakeChannel()

    async def boom(user_text, history):
        raise RuntimeError("llm down")

    engine.reply = boom

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert channel.sent == ["Sorry — something went wrong on my side."]


async def test_history_is_passed_to_engine_and_updated():
    bot, engine = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> one", channel, mentions=[FakeUser(999)]))
    await bot.on_message(FakeMessage("<@999> two", channel, mentions=[FakeUser(999)]))

    _, history_on_second = engine.calls[1]
    assert any(m.get("content") == "one" for m in history_on_second)


# -- Reply-to-bot messages -----------------------------------------------------------


async def test_reply_to_bot_message_triggers_response():
    """Replying to one of the bot's own messages triggers a response."""
    bot, engine = _bot()
    channel = FakeChannel()

    bot_author = FakeUser()
    bot_author.display_name = "chord"
    ref_resolved = type(
        "ResolvedMsg",
        (),
        {"content": "Here's the weather data.", "author": bot_author},
    )()
    reference = type("Ref", (), {"resolved": ref_resolved})()

    msg = FakeMessage(
        "<@999> explain this",
        channel,
        mentions=[FakeUser(999)],
        guild="g",
        reference=reference,
    )

    await bot.on_message(msg)

    assert len(engine.calls) == 1
    assert "explain this" in engine.calls[0][0]
    assert "Here's the weather data" in engine.calls[0][0]  # reply context


# -- Persona integration --------------------------------------------------------------


async def test_persona_refreshes_on_each_message():
    bot, engine = _bot()

    captured: list[str] = []

    def capture_set(self, value):
        captured.append(value)

    def capture_get(self):
        return captured[-1] if captured else ""

    StubEngine.system_prompt = property(capture_get, capture_set)
    try:
        await bot.on_message(
            FakeMessage("<@999> hi", FakeChannel(), mentions=[FakeUser(999)], guild="g")
        )
        assert len(captured) >= 1  # system prompt was set from persona
    finally:
        del StubEngine.system_prompt
