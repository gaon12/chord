"""Tests for the Discord layer - pure helpers plus on_message flow.

Real Discord connections are never opened; small fakes stand in for
messages and channels so the whole reply path runs in-memory.
"""

from __future__ import annotations

import contextlib

from chord.bot import ChordBot, clean_message_text, split_message
from chord.config import Settings

# -- clean_message_text -----------------------------------------------------------


def test_clean_message_text_removes_mentions():
    assert clean_message_text("<@123456789> what time is it?") == "what time is it?"


def test_clean_message_text_handles_nicknames_and_extra_space():
    assert clean_message_text("<@!42>  hi   there ") == "hi there"


def test_clean_message_text_keeps_regular_text():
    assert clean_message_text("plain text") == "plain text"


# -- split_message ------------------------------------------------------------------


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


# -- Fakes for on_message ------------------------------------------------------------


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
    def __init__(self, is_bot: bool = False):
        self.bot = is_bot


class FakeMessage:
    def __init__(
        self,
        content: str,
        channel: FakeChannel,
        *,
        author_bot: bool = False,
        mentions=(),
        guild=None,
    ):
        self.content = content
        self.channel = channel
        self.author = FakeAuthor(author_bot)
        self.mentions = list(mentions)
        self.guild = guild


class FakeUser:
    """Stands in for the bot's own client user."""


class StubEngine:
    """Records calls and returns a canned answer."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    async def reply(self, user_text, history):
        self.calls.append((user_text, history))
        return ("answer!", [{"role": "user", "content": user_text}])


def _bot() -> tuple[ChordBot, StubEngine]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
    )
    engine = StubEngine()
    bot = ChordBot(settings=settings, engine=engine)  # type: ignore[arg-type]
    bot.me = FakeUser()
    return bot, engine


async def test_replies_when_mentioned_in_guild():
    bot, engine = _bot()
    channel = FakeChannel()
    msg = FakeMessage("<@999> hello", channel, mentions=[bot.me], guild="g")

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
    await bot.on_message(FakeMessage("<@999>", channel, mentions=[bot.me]))

    assert "chat with me" in channel.sent[0]


async def test_engine_failure_sends_friendly_error():
    bot, engine = _bot()
    channel = FakeChannel()

    async def boom(user_text, history):
        raise RuntimeError("llm down")

    engine.reply = boom

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[bot.me]))

    assert channel.sent == ["Sorry - something went wrong on my side."]


async def test_reset_command_clears_history():
    bot, engine = _bot()
    channel = FakeChannel()

    # Simulate a prior turn remembered for this channel.
    bot._store.append(channel.id, {"role": "user", "content": "old"})
    await bot.on_message(FakeMessage("!reset", channel))

    assert bot._store.history(channel.id) == []
    assert channel.sent == ["Conversation cleared."]


async def test_help_command():
    bot, _ = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("!help", channel))

    assert "!reset" in channel.sent[0]


async def test_history_is_passed_to_engine_and_updated():
    bot, engine = _bot()
    channel = FakeChannel()

    first = FakeMessage("<@999> one", channel, mentions=[bot.me])
    await bot.on_message(first)
    second = FakeMessage("<@999> two", channel, mentions=[bot.me])
    await bot.on_message(second)

    # The second call must include the messages stored by the first turn.
    _, history_on_second = engine.calls[1]
    assert [m["content"] for m in history_on_second] == ["one"]
