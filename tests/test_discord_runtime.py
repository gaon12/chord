"""Concurrency and authorization at the Discord event and command boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from chord.channel_turns import ChannelTurns
from chord.discord_text import clean_message_text
from test_bot import FakeChannel, FakeMessage, StubEngine, _bot


async def test_same_channel_waits_but_other_channels_can_continue():
    bot, _ = _bot(history_token_budget=0)
    started, release = asyncio.Event(), asyncio.Event()

    class BlockingEngine(StubEngine):
        async def reply(self, text, history, *, system_prompt=None):
            self.calls.append((text, history))
            if text.endswith("first"):
                started.set()
                await release.wait()
            return text, [{"role": "user", "content": text}]

    engine = BlockingEngine()
    bot._engine = engine
    channel = FakeChannel(1)
    first = asyncio.create_task(bot.on_message(FakeMessage("first", channel)))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(bot.on_message(FakeMessage("second", channel)))
    await asyncio.sleep(0)
    await bot.on_message(FakeMessage("other", FakeChannel(2)))
    assert [call[0] for call in engine.calls] == ["[user]: first", "[user]: other"]
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert engine.calls[-1][1] == [{"role": "user", "content": "[user]: first"}]
    assert channel.sent == ["[user]: first", "[user]: second"]


async def test_cancelled_waiter_does_not_break_channel_lock():
    turns = ChannelTurns()
    async with turns.hold(1):
        waiter = asyncio.create_task(_enter(turns))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    await asyncio.wait_for(_enter(turns), 2)


async def _enter(turns):
    async with turns.hold(1):
        pass


def _interaction(user_id=7, guild=True, manage_messages=False):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id),
        guild=object() if guild else None,
        channel_id=1,
        permissions=SimpleNamespace(manage_messages=manage_messages),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


async def test_only_application_owner_can_change_global_reasoning():
    bot, _ = _bot()
    bot._owner_ids = {42}
    bot._register_slash_commands()
    change = bot.tree.get_command("reasoning").callback
    stranger = _interaction()
    await change(stranger, "heavy")
    assert bot._reasoning_level == "none"
    assert "Only the application owner" in stranger.response.send_message.call_args.args[0]


async def test_persona_is_not_disclosed_to_other_users():
    bot, _ = _bot()
    bot._register_slash_commands()
    interaction = _interaction(guild=False)
    await bot.tree.get_command("persona").callback(interaction, "view")
    assert "Current persona" not in interaction.response.send_message.call_args.args[0]


async def test_guild_reset_requires_permission_and_dm_reset_is_serialized():
    bot, _ = _bot()
    bot._register_slash_commands()
    bot._store.append(1, {"role": "user", "content": "old"})
    reset = bot.tree.get_command("reset").callback
    await reset(_interaction())
    assert bot._store.history(1)
    await reset(_interaction(guild=False))
    assert bot._store.history(1) == []


def test_model_replies_cannot_ping_users_or_everyone():
    bot, _ = _bot()
    mentions = bot.allowed_mentions
    assert not any((mentions.everyone, mentions.users, mentions.roles, mentions.replied_user))


def test_incoming_code_keeps_its_indentation_and_newlines():
    assert (
        clean_message_text("<@999>\n```py\nif True:\n    pass\n```")
        == "```py\nif True:\n    pass\n```"
    )
