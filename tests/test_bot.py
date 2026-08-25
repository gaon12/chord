"""Tests for chord.bot - helpers plus on_message flow with fake objects."""

from __future__ import annotations

import contextlib

import discord

import quota_helpers  # noqa: F401  (conftest already isolates quota store)
from chord.attachments import attach
from chord.bot import (
    HELP_TEXT,
    ChordBot,
    build_reply_context,
    clean_message_text,
    estimate_tool_prompt_tokens,
    format_reply,
    label_speaker,
    speaker_name,
    split_message,
    warn_if_tool_catalog_is_large,
)
from chord.compaction import SUMMARY_PREFIX, HistoryCompactor
from chord.config import REASONING_LEVELS, Settings
from chord.skills.registry import SkillRegistry
from fakes import FakeLLM

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

    def __init__(self, channel_id: int = 100, raises: Exception | None = None):
        self.id = channel_id
        self.sent: list[str] = []
        #: One entry per send(): the files it carried, if any.
        self.uploads: list[list] = []
        #: When set, every send() raises it (permission / HTTP failures).
        self.raises = raises
        #: When set, only sends carrying files raise it.
        self.rejects_files: Exception | None = None

    @contextlib.asynccontextmanager
    async def typing(self):
        yield None

    async def send(self, content: str, files=None):
        if self.raises is not None:
            raise self.raises
        if files and self.rejects_files is not None:
            raise self.rejects_files
        self.sent.append(content)
        self.uploads.append(list(files or []))


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
        author_display=None,
        mentions=(),
        guild=None,
        reference=None,
        role_mentions=(),
    ):
        self.content = content
        self.channel = channel
        self.author = FakeAuthor(name=author_name, is_bot=author_bot, display_name=author_display)
        self.mentions = list(mentions)
        self.role_mentions = list(role_mentions)
        self.guild = guild
        self.reference = reference


class _FakeResponse:
    """Just enough of an aiohttp response for discord.HTTPException."""

    status = 413
    reason = "Payload Too Large"


class FakeUser:
    """Stands in for a Discord user (bot or human)."""

    def __init__(self, user_id: int = 999):
        self.id = user_id


def _bot(**overrides) -> tuple[ChordBot, StubEngine]:
    settings = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        **overrides,
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

    assert engine.calls[0][0] == "[user]: hello"
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

    assert engine.calls[0][0] == "[user]: dm text"


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
    assert any(m.get("content") == "[user]: one" for m in history_on_second)


# -- Speaker identity ------------------------------------------------------------------


def test_speaker_name_prefers_the_server_nickname():
    assert speaker_name(FakeAuthor(name="account", display_name="Nickname")) == "Nickname"


def test_speaker_name_falls_back_to_the_account_name():
    author = FakeAuthor(name="account")
    del author.display_name
    assert speaker_name(author) == "account"


def test_speaker_name_folds_brackets_so_labels_cannot_be_forged():
    assert speaker_name(FakeAuthor(display_name="[admin]")) == "(admin)"


def test_speaker_name_collapses_whitespace_and_newlines():
    assert speaker_name(FakeAuthor(display_name="a\n b  c")) == "a b c"


def test_speaker_name_is_truncated_to_discords_own_limit():
    assert len(speaker_name(FakeAuthor(display_name="n" * 80))) == 32


def test_speaker_name_survives_an_author_with_no_name_at_all():
    assert speaker_name(object()) == "unknown"


def test_label_speaker_tags_the_message():
    assert label_speaker("Alice", "hi") == "[Alice]: hi"


async def test_each_author_reaches_the_engine_under_their_own_name():
    """Two people in one channel must not read as one person."""
    bot, engine = _bot()
    channel = FakeChannel()

    await bot.on_message(
        FakeMessage(
            "<@999> 내 이름 기억해",
            channel,
            mentions=[FakeUser(999)],
            guild="g",
            author_display="Alice",
        )
    )
    await bot.on_message(
        FakeMessage(
            "<@999> 나는?",
            channel,
            mentions=[FakeUser(999)],
            guild="g",
            author_display="Bob",
        )
    )

    assert engine.calls[0][0] == "[Alice]: 내 이름 기억해"
    assert engine.calls[1][0] == "[Bob]: 나는?"
    # ...and the earlier speaker is still named in the stored history.
    assert any(m.get("content") == "[Alice]: 내 이름 기억해" for m in engine.calls[1][1])


async def test_reply_context_stays_in_front_of_the_speaker_label():
    bot, engine = _bot()
    channel = FakeChannel()

    quoted_author = FakeUser()
    quoted_author.display_name = "Carol"
    resolved = type("ResolvedMsg", (), {"content": "원본 메시지", "author": quoted_author})()
    reference = type("Ref", (), {"resolved": resolved})()

    await bot.on_message(
        FakeMessage(
            "<@999> 이거 무슨 뜻이야?",
            channel,
            mentions=[FakeUser(999)],
            guild="g",
            reference=reference,
            author_display="Dave",
        )
    )

    prompt = engine.calls[0][0]
    assert prompt.startswith("[replying to Carol:")
    assert prompt.endswith("[Dave]: 이거 무슨 뜻이야?")


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


# -- Skill-produced attachments ---------------------------------------------------------


class AttachingEngine(StubEngine):
    """An engine whose skills drop a file in, the way a chart skill does."""

    def __init__(self, answer: str = "answer!", count: int = 1):
        super().__init__()
        self.answer = answer
        self.count = count

    async def reply(self, user_text, history):
        for index in range(self.count):
            attach(f"chart{index}.png", b"\x89PNG fake")
        self.calls.append((user_text, history))
        return (self.answer, [{"role": "user", "content": user_text}])


async def test_a_chart_produced_by_a_skill_reaches_the_channel():
    bot, _ = _bot()
    bot._engine = AttachingEngine()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> 환율 그래프", channel, mentions=[FakeUser(999)]))

    assert channel.sent == ["answer!"]
    assert [f.filename for f in channel.uploads[0]] == ["chart0.png"]


async def test_files_ride_on_the_last_chunk_of_a_long_answer():
    """The image belongs under the whole answer, not halfway through it."""
    bot, _ = _bot()
    bot._engine = AttachingEngine(answer="\n\n".join(["x" * 1500, "y" * 1500]))
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert len(channel.sent) == 2
    assert channel.uploads[0] == []
    assert [f.filename for f in channel.uploads[1]] == ["chart0.png"]


async def test_a_rejected_upload_does_not_cost_the_user_the_answer():
    """The numbers are the answer; the chart is the nice-to-have."""
    bot, _ = _bot()
    bot._engine = AttachingEngine()
    channel = FakeChannel()
    channel.rejects_files = discord.HTTPException(_FakeResponse(), "payload too large")

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert channel.sent == ["answer!"]
    assert channel.uploads == [[]]  # retried without the file


async def test_files_do_not_leak_into_the_next_turn():
    bot, _ = _bot()
    bot._engine = AttachingEngine()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> one", channel, mentions=[FakeUser(999)]))
    bot._engine = StubEngine()  # a turn whose skills attach nothing
    await bot.on_message(FakeMessage("<@999> two", channel, mentions=[FakeUser(999)]))

    assert channel.uploads[1] == []


async def test_a_plain_answer_sends_no_upload_argument():
    """Every provider path stays exactly as it was when there is no file."""
    bot, _ = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert channel.uploads == [[]]


# -- Tool index in the system prompt ---------------------------------------------------


async def test_the_system_prompt_lists_the_registered_tools():
    """A 25-entry JSON catalog is easy for a small model to overlook."""
    from chord.skills.base import Skill

    class WeatherSkill(Skill):
        name = "get_weather"
        description = "Current weather for a city."
        parameters = {"type": "object", "properties": {}}

        async def run(self, **kwargs):
            return "sunny"

    registry = SkillRegistry()
    registry.register(WeatherSkill())
    bot, engine = _bot()
    bot._registry = registry
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert "- get_weather: Current weather for a city" in engine.system_prompt_value


# -- History compaction ----------------------------------------------------------------


async def test_long_channel_history_is_compacted_after_the_reply():
    """The digest is paid for after answering, not before."""
    bot, engine = _bot()
    channel = FakeChannel()
    bot._compactor = HistoryCompactor(FakeLLM("- 앨리스가 날씨를 물어봄"), token_budget=1)

    await bot.on_message(FakeMessage("<@999> one", channel, mentions=[FakeUser(999)]))
    await bot.on_message(FakeMessage("<@999> two", channel, mentions=[FakeUser(999)]))
    await bot.on_message(FakeMessage("<@999> three", channel, mentions=[FakeUser(999)]))

    history = engine.calls[-1][1]
    assert history[0]["content"].startswith(SUMMARY_PREFIX)
    assert "앨리스가 날씨를 물어봄" in history[0]["content"]
    # The oldest turn now lives in the digest instead of verbatim.
    assert not any(m.get("content") == "[user]: one" for m in history)
    # ...and every answer still went out.
    assert channel.sent == ["answer!", "answer!", "answer!"]


async def test_compaction_is_off_when_the_budget_is_zero():
    bot, engine = _bot(history_token_budget=0)
    channel = FakeChannel()

    for text in ("one", "two", "three"):
        await bot.on_message(FakeMessage(f"<@999> {text}", channel, mentions=[FakeUser(999)]))

    assert any(m.get("content") == "[user]: one" for m in engine.calls[-1][1])


async def test_a_failing_summarizer_never_costs_the_user_their_answer():
    bot, _ = _bot()
    channel = FakeChannel()

    class BoomCompactor:
        async def compact(self, history):
            raise RuntimeError("summarizer down")

    bot._compactor = BoomCompactor()

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)]))

    assert channel.sent == ["answer!"]


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


# -- Why a message does (not) get answered -------------------------------------------


class FakeRole:
    def __init__(self, role_id: int):
        self.id = role_id


class FakeGuild:
    """Guild exposing the bot member's roles, like discord.Guild.me."""

    def __init__(self, my_role_ids=(), default_role_id: int = 1):
        self.me = type("Member", (), {"roles": [FakeRole(r) for r in my_role_ids]})()
        self.default_role = FakeRole(default_role_id)


def test_role_mention_of_the_bots_own_role_is_answered():
    """'@chord' usually autocompletes to the bot's integration role."""
    bot, _ = _bot()
    guild = FakeGuild(my_role_ids=[555])
    msg = FakeMessage("<@&555> 안녕?", FakeChannel(), guild=guild, role_mentions=[FakeRole(555)])

    assert bot._reply_reason(msg) == "role-mention"


def test_everyone_ping_is_not_treated_as_a_mention():
    bot, _ = _bot()
    guild = FakeGuild(my_role_ids=[1], default_role_id=1)
    msg = FakeMessage("@everyone", FakeChannel(), guild=guild, role_mentions=[FakeRole(1)])

    assert bot._reply_reason(msg) is None


def test_role_mention_of_someone_elses_role_is_ignored():
    bot, _ = _bot()
    guild = FakeGuild(my_role_ids=[555])
    msg = FakeMessage("<@&777> hey", FakeChannel(), guild=guild, role_mentions=[FakeRole(777)])

    assert bot._reply_reason(msg) is None


def test_raw_mention_token_is_answered_when_mentions_are_unresolved():
    """Guards against gateway payloads that don't resolve `mentions`."""
    bot, _ = _bot()
    msg = FakeMessage("<@999> 안녕?", FakeChannel(), guild="g", mentions=[])

    assert bot._reply_reason(msg) == "mention-token"


def test_nickname_mention_token_is_answered():
    bot, _ = _bot()
    msg = FakeMessage("<@!999> 안녕?", FakeChannel(), guild="g", mentions=[])

    assert bot._reply_reason(msg) == "mention-token"


def test_reply_reason_is_dm_outside_guilds():
    bot, _ = _bot()
    assert bot._reply_reason(FakeMessage("hi", FakeChannel())) == "dm"


def test_role_mention_token_is_stripped_from_the_prompt():
    assert clean_message_text("<@&555> 안녕?") == "안녕?"


async def test_role_mention_reaches_the_engine():
    bot, engine = _bot()
    channel = FakeChannel()
    guild = FakeGuild(my_role_ids=[555])

    await bot.on_message(
        FakeMessage("<@&555> 안녕?", channel, guild=guild, role_mentions=[FakeRole(555)])
    )

    assert engine.calls[0][0] == "[user]: 안녕?"
    assert channel.sent == ["answer!"]


# -- Send failures and unexpected errors ----------------------------------------------


async def test_missing_send_permission_does_not_crash_the_handler():
    """403 on send must be logged, not raised into discord.py's void."""
    import discord

    bot, engine = _bot()
    forbidden = discord.Forbidden.__new__(discord.Forbidden)
    Exception.__init__(forbidden, "missing permissions")
    channel = FakeChannel(raises=forbidden)

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)], guild="g"))

    assert engine.calls  # the turn ran to completion
    assert channel.sent == []


async def test_unexpected_handler_error_is_swallowed_and_logged(caplog):
    bot, _ = _bot()

    async def boom(message):
        raise RuntimeError("kaboom")

    bot._handle_message = boom

    await bot.on_message(FakeMessage("<@999> hi", FakeChannel(), mentions=[FakeUser(999)]))

    assert "Unhandled error" in caplog.text


# -- /reasoning ------------------------------------------------------------------------


class StubLLM:
    """Minimal LLMService stand-in for the /reasoning command."""

    def __init__(self, effort: str | None = "minimal", enabled: bool = True):
        self.reasoning_effort = effort
        self._enabled = enabled

    @property
    def reasoning_enabled(self) -> bool:
        return self._enabled and self.reasoning_effort is not None

    def set_reasoning_effort(self, effort: str | None) -> None:
        self.reasoning_effort = effort
        self._enabled = True


def _bot_with_llm(level: str = "none") -> tuple[ChordBot, StubLLM]:
    settings = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="k",
        reasoning_level=level,
    )
    engine = StubEngine()
    llm = StubLLM()
    engine.llm = llm
    bot = ChordBot(settings=settings, engine=engine)
    bot._me_id = 999
    return bot, llm


def test_reasoning_starts_at_the_configured_level():
    bot, _ = _bot_with_llm("light")
    assert "**light**" in bot._describe_reasoning()


def test_describe_reasoning_lists_every_level():
    bot, _ = _bot_with_llm()
    described = bot._describe_reasoning()
    assert all(level in described for level in REASONING_LEVELS)


def test_setting_a_level_retunes_the_live_llm():
    bot, llm = _bot_with_llm("none")

    message = bot._set_reasoning_level("heavy")

    assert llm.reasoning_effort == "high"
    assert bot._reasoning_level == "heavy"
    assert "**none**" in message and "**heavy**" in message


def test_setting_auto_drops_the_parameter_entirely():
    bot, llm = _bot_with_llm("medium")

    bot._set_reasoning_level("auto")

    assert llm.reasoning_effort is None
    assert llm.reasoning_enabled is False


def test_unknown_level_is_rejected_without_touching_the_llm():
    bot, llm = _bot_with_llm("none")

    message = bot._set_reasoning_level("galaxy-brain")

    assert "Unknown level" in message
    assert llm.reasoning_effort == "minimal"  # unchanged
    assert bot._reasoning_level == "none"


def test_describe_warns_when_the_model_rejected_the_parameter():
    """Silently ignored settings are worse than none at all."""
    bot, llm = _bot_with_llm("heavy")
    llm._enabled = False

    assert "no effect" in bot._describe_reasoning()


def test_no_warning_when_reasoning_is_deliberately_off():
    bot, _ = _bot_with_llm("auto")
    assert "no effect" not in bot._describe_reasoning()


def test_reasoning_is_advertised_in_help():
    assert "/reasoning" in HELP_TEXT


# -- Leaked chain-of-thought -------------------------------------------------------


def test_thought_block_is_removed_from_the_reply():
    """gemma-4-31b-it narrates its reasoning inline before answering."""
    raw = "<thought>The user greeted me in Korean.\nI should be brief.</thought>안녕하세요!"
    assert format_reply(raw) == "안녕하세요!"


def test_think_tag_variants_and_casing_are_handled():
    assert format_reply("<THINK>hmm</THINK>42") == "42"
    assert format_reply("<Reasoning>x</Reasoning>ok") == "ok"
    assert format_reply('<think id="1">x</think>ok') == "ok"


def test_unclosed_thought_tag_drops_the_rest():
    """A cut-off answer must not dump half a monologue into the channel."""
    assert format_reply("Here you go.\n<thought>wait, actually") == "Here you go."


def test_orphan_closing_tag_drops_what_came_before():
    assert format_reply("rambling on and on</thought>The answer is 42.") == "The answer is 42."


def test_code_fences_keep_literal_reasoning_tags():
    """Explaining <think> tags in a snippet is content, not reasoning."""
    raw = "Use this:\n```html\n<think>example</think>\n```"
    result = format_reply(raw)
    assert "<think>example</think>" in result


def test_thought_only_answer_is_kept_rather_than_blanked():
    """A leaked thought still beats sending nothing."""
    raw = "<thought>I have no idea what to say.</thought>"
    assert "no idea" in format_reply(raw)


def test_ordinary_answers_are_untouched():
    assert format_reply("  Seoul is 23°C right now.  ") == "Seoul is 23°C right now."


def test_mismatched_tags_do_not_close_each_other():
    result = format_reply("<thought>a</think>b")
    assert "b" in result


async def test_leaked_reasoning_never_reaches_the_channel():
    bot, engine = _bot()
    channel = FakeChannel()

    async def reply(user_text, history):
        return ("<thought>thinking…</thought>Hi!", [{"role": "user", "content": user_text}])

    engine.reply = reply

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)], guild="g"))

    assert channel.sent == ["Hi!"]


async def test_llm_timeout_gets_its_own_message():
    """A timeout is not a crash - tell the user to just ask again."""
    from openai import APITimeoutError

    bot, engine = _bot()
    channel = FakeChannel()

    async def stall(user_text, history):
        raise APITimeoutError(request=None)

    engine.reply = stall

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)], guild="g"))

    assert len(channel.sent) == 1
    assert "다시 물어봐" in channel.sent[0]


# -- Tool catalog size ------------------------------------------------------------


def _tools(count: int, schema_chars: int = 400) -> list[dict]:
    """Tool definitions of a realistic size, for the size estimator."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "d" * schema_chars,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(count)
    ]


def test_tool_token_estimate_grows_with_the_catalog():
    small = estimate_tool_prompt_tokens(_tools(10))
    large = estimate_tool_prompt_tokens(_tools(100))

    assert 0 < small < large


def test_empty_catalog_costs_nothing():
    assert estimate_tool_prompt_tokens([]) == 0


def test_large_tool_catalog_warns_about_the_input_token_budget(caplog):
    """145 tools measured at 15 303 prompt tokens - 96% of a 16k/min quota."""
    with caplog.at_level("WARNING"):
        warn_if_tool_catalog_is_large(_tools(120))

    assert "mcp.json" in caplog.text
    assert "429" in caplog.text


def test_normal_tool_catalog_stays_quiet(caplog):
    with caplog.at_level("WARNING"):
        warn_if_tool_catalog_is_large(_tools(5))

    assert caplog.text == ""


def test_warning_threshold_is_measured_in_tokens_not_tool_count(caplog):
    """Many tiny tools are cheap; a few enormous schemas are not."""
    with caplog.at_level("WARNING"):
        warn_if_tool_catalog_is_large(_tools(200, schema_chars=10))
    assert caplog.text == ""

    with caplog.at_level("WARNING"):
        warn_if_tool_catalog_is_large(_tools(3, schema_chars=20_000))
    assert "mcp.json" in caplog.text


async def test_rate_limit_gets_its_own_message():
    """429 is a quota problem the user can act on, not an internal error."""
    from openai import RateLimitError

    bot, engine = _bot()
    channel = FakeChannel()

    async def throttled(user_text, history):
        exc = RateLimitError.__new__(RateLimitError)
        Exception.__init__(exc, "429 quota exceeded")
        raise exc

    engine.reply = throttled

    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)], guild="g"))

    assert len(channel.sent) == 1
    assert "사용량 한도" in channel.sent[0]


# -- Recovering from a rejected history -------------------------------------------


def _bad_request(message: str):
    from openai import BadRequestError

    exc = BadRequestError.__new__(BadRequestError)
    Exception.__init__(exc, message)
    return exc


HISTORY_400 = (
    "400 - GenerateContentRequest.contents[0].parts[0]"
    ".function_response.name: Name cannot be empty."
)


async def test_rejected_history_is_cleared_so_the_channel_recovers():
    """Otherwise the same bad messages are re-sent on every later turn."""
    bot, engine = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> one", channel, mentions=[FakeUser(999)], guild="g"))
    assert bot._store.history(channel.id)  # history exists now

    async def rejected(user_text, history):
        raise _bad_request(HISTORY_400)

    engine.reply = rejected
    await bot.on_message(FakeMessage("<@999> two", channel, mentions=[FakeUser(999)], guild="g"))

    assert bot._store.history(channel.id) == []
    assert "초기화" in channel.sent[-1]


async def test_the_next_message_works_again_after_recovery():
    bot, engine = _bot()
    channel = FakeChannel()

    await bot.on_message(FakeMessage("<@999> one", channel, mentions=[FakeUser(999)], guild="g"))

    failed_once = []

    async def rejected_once(user_text, history):
        if not failed_once:
            failed_once.append(True)
            raise _bad_request(HISTORY_400)
        return ("recovered", [{"role": "user", "content": user_text}])

    engine.reply = rejected_once
    await bot.on_message(FakeMessage("<@999> two", channel, mentions=[FakeUser(999)], guild="g"))
    await bot.on_message(FakeMessage("<@999> three", channel, mentions=[FakeUser(999)], guild="g"))

    assert channel.sent[-1] == "recovered"


async def test_bad_request_without_history_is_not_blamed_on_history():
    """An empty history cannot be the cause - a bad tool schema can."""
    bot, engine = _bot()
    channel = FakeChannel()

    async def rejected(user_text, history):
        raise _bad_request("400 - Invalid JSON schema for function 'add'.")

    engine.reply = rejected
    await bot.on_message(FakeMessage("<@999> hi", channel, mentions=[FakeUser(999)], guild="g"))

    assert channel.sent == ["Sorry — something went wrong on my side."]
