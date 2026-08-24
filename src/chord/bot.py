"""Discord layer - wires the chat engine to Discord events.

Responsibilities kept here (and nowhere else):

* deciding when to answer (mentioned or DM),
* cleaning the raw message into user text,
* sending replies within Discord's length limits,
* tiny convenience commands (!help / !reset).

All conversation logic lives in :mod:`chord.engine`.
"""

from __future__ import annotations

import logging
import re

import discord
from discord.ext import tasks

from chord.config import Settings
from chord.context import reset_current_channel, set_current_channel
from chord.conversation import ConversationStore
from chord.engine import ChatEngine
from chord.llm import LLMService
from chord.mcp_client import McpManager
from chord.persona import PersonaProvider
from chord.reminders import ReminderStore
from chord.skills import create_default_registry
from chord.skills._quota import get_quota_store, render_usage
from chord.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

#: Discord hard-limits every message to 2000 characters.
DISCORD_MESSAGE_LIMIT = 2000

#: Matches <@123456> and <@!123456> mention tokens.
_MENTION_RE = re.compile(r"<@!?\d+>")

HELP_TEXT = (
    "**chord** - chat with me by mentioning me, e.g. `@chord how's the weather "
    "in Seoul?`\n"
    "`!help`  - show this message\n"
    "`!usage` - show remaining API quotas\n"
    "`!reminders - list pending reminders for this channel\n"
    "`!reset` - forget this channel's conversation"
)


def clean_message_text(content: str) -> str:
    """Strip mention tokens and tidy whitespace from a raw message."""
    cleaned = _MENTION_RE.sub(" ", content)
    return " ".join(cleaned.split())


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split long text into sendable chunks.

    Prefers breaking at blank lines, then at newlines, then hard-slices.
    Returns at least one chunk even for empty input.
    """
    text = text.strip()
    if not text:
        return ["(empty reply)"]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    for paragraph in text.split("\n\n"):
        # A single paragraph can itself exceed the limit -> split harder.
        while len(paragraph) > limit:
            head, paragraph = _hard_slice(paragraph, limit)
            chunks.append(head)
        if paragraph:
            chunks.append(paragraph)

    # Merge small neighbours so we do not spam many tiny messages.
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) + 2 + len(chunk) <= limit:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def _hard_slice(text: str, limit: int) -> tuple[str, str]:
    """Cut ``text`` at ``limit``, preferring the last newline inside."""
    cut = text.rfind("\n", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip(), text[cut:].lstrip()


class ChordBot(discord.Client):
    """Discord client that answers mentions using the chat engine."""

    def __init__(
        self,
        settings: Settings,
        engine: ChatEngine,
        registry: SkillRegistry | None = None,
    ) -> None:
        # message_content intent is required to read non-command messages;
        # it must also be enabled in the developer portal.
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        # Cached in on_ready because Client.user is read-only and only
        # populated after login; keeping our own reference also makes the
        # mention check easy to exercise in tests.
        self.me: discord.ClientUser | None = None

        self._settings = settings
        self._engine = engine
        self._persona = PersonaProvider(settings.persona_path)
        self._reminders = ReminderStore(settings.reminder_db_path)
        self._store = ConversationStore()
        # The registry is kept so setup_hook can add MCP tools later;
        # the engine reads it live, so additions are picked up
        # immediately without rebuilding the engine.
        self._registry = registry or SkillRegistry()
        self._mcp = McpManager()

    async def setup_hook(self) -> None:
        """Called once after login but before the gateway connects.

        MCP servers are started here so their tools are registered
        before the first message arrives, and a periodic loop keeps
        them in sync with mcp.json edits at runtime.
        """
        registered = await self._mcp.start(self._settings, self._registry.register)
        if registered:
            logger.info("Registered %d MCP tool(s).", registered)
        self._mcp_reload_loop.start()
        self._reminder_loop.start()

    @tasks.loop(minutes=30)
    async def _mcp_reload_loop(self) -> None:
        """Periodically re-read mcp.json and hot-reload on changes."""
        try:
            changed = await self._mcp.reload_if_changed(
                self._settings, self._registry, self._registry.register
            )
            if changed:
                logger.info("MCP tools refreshed from mcp.json.")
        except Exception:  # noqa: BLE001 - a bad config must not kill the loop
            logger.exception("MCP reload failed; keeping previous servers.")

    @_mcp_reload_loop.before_loop
    async def _wait_until_gateway_ready(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(seconds=30)
    async def _reminder_loop(self) -> None:
        """Deliver reminders whose time has come."""
        delivered = await self.deliver_due_reminders()
        if delivered:
            logger.info("Delivered %d reminder(s).", delivered)

    @_reminder_loop.before_loop
    async def _wait_until_gateway_ready_for_reminders(self) -> None:
        await self.wait_until_ready()

    def _resolve_channel(self, channel_id: int):
        """Find a channel by id (cached guild channels first, then fetch)."""
        channel = self.get_channel(channel_id)
        return channel

    async def deliver_due_reminders(self) -> int:
        """Send every due reminder; returns how many were delivered."""
        count = 0
        for reminder in self._reminders.due():
            try:
                channel = self._resolve_channel(reminder.channel_id)
                if channel is None:
                    channel = await self.fetch_channel(reminder.channel_id)
                local_due = reminder.due.astimezone()
                await channel.send(
                    f"⏰ Reminder: {reminder.text} (scheduled {local_due.strftime('%m-%d %H:%M')})"
                )
                self._reminders.mark_done(reminder.id)
                count += 1
            except Exception:  # noqa: BLE001 - one bad channel != stop all
                logger.exception(
                    "Could not deliver reminder #%s to channel %s",
                    reminder.id,
                    reminder.channel_id,
                )
        return count

    @_mcp_reload_loop.before_loop
    async def _wait_until_gateway_ready(self) -> None:
        await self.wait_until_ready()

    async def close(self) -> None:
        self._reminder_loop.cancel()
        self._mcp_reload_loop.cancel()
        await self._mcp.stop()
        await super().close()

    async def on_ready(self) -> None:
        self.me = self.user
        logger.info("Logged in as %s (id=%s)", self.user, self.user and self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        # Ignore other bots and ourselves.
        if message.author.bot:
            return

        # Plain-text convenience commands start with '!'.
        if message.content.startswith("!"):
            await self._handle_command(message)
            return
        if not self._should_reply(message):
            return

        user_text = clean_message_text(message.content)
        if not user_text:
            await message.channel.send(HELP_TEXT)
            return

        channel_id = message.channel.id
        # Persona edits land on the very next message - no restart.
        self._engine.system_prompt = self._persona.get()
        # Skills like set_reminder need to know where we are.
        token = set_current_channel(channel_id)
        async with message.channel.typing():
            try:
                answer, new_messages = await self._engine.reply(
                    user_text, self._store.history(channel_id)
                )
            except Exception:
                logger.exception("Chat failed in channel %s", channel_id)
                await message.channel.send("Sorry - something went wrong on my side.")
                return
            finally:
                reset_current_channel(token)

        self._store.append(channel_id, *new_messages)
        for chunk in split_message(answer):
            await message.channel.send(chunk)

    # -- Internals -------------------------------------------------------------

    def _should_reply(self, message: discord.Message) -> bool:
        """Answer direct messages and server messages mentioning us."""
        if message.guild is None:
            return True
        return self.me in (message.mentions or [])

    async def _handle_command(self, message: discord.Message) -> None:
        """Handle the plain-text commands the bot understands."""
        command = message.content.strip().lower()
        if command == "!reminders":
            rows = self._reminders.pending_for_channel(message.channel.id)
            if not rows:
                await message.channel.send("No pending reminders in this channel.")
                return
            lines = [f"{len(rows)} pending reminder(s):"]
            for row in rows:
                local = row.due.astimezone().strftime("%m-%d %H:%M")
                lines.append(f"#{row.id} {local} - {row.text}")
            await message.channel.send("\n".join(lines))
        elif command == "!reset":
            self._store.reset(message.channel.id)
            await message.channel.send("Conversation cleared.")
        elif command == "!usage":
            store = get_quota_store(self._settings.quota_store_path)
            await message.channel.send(render_usage(store))
        elif command == "!help":
            await message.channel.send(HELP_TEXT)


def build_bot(settings: Settings) -> ChordBot:
    """Create a fully wired bot instance from settings.

    This is the composition root: everything (LLM client, skills, MCP,
    engine) gets connected here, keeping individual modules decoupled.
    """
    registry = create_default_registry(settings)
    engine = ChatEngine(
        llm=LLMService(settings),
        registry=registry,
        system_prompt=settings.system_prompt,
    )
    logger.info(
        "Bot ready: %d skill(s), model=%s, base_url=%s",
        len(registry),
        settings.openai_model,
        settings.openai_base_url,
    )
    return ChordBot(settings=settings, engine=engine, registry=registry)
