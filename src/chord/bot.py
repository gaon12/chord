"""Discord layer - wires the chat engine to Discord events.

Responsibilities kept here (and nowhere else):

* deciding when to answer (mentioned, DM'd, or replied-to),
* cleaning raw messages into LLM-ready text,
* registering and serving slash commands,
* sending replies within Discord's length limits,
* extracting reply-to-message context for better answers.

All conversation logic lives in :mod:`chord.engine`.
"""

from __future__ import annotations

import logging
import re

import discord
from discord import app_commands
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
        while len(paragraph) > limit:
            head, paragraph = _hard_slice(paragraph, limit)
            chunks.append(head)
        if paragraph:
            chunks.append(paragraph)

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


def build_reply_context(message: discord.Message) -> str:
    """Extract replied-to message content for LLM context."""
    ref = getattr(message, "reference", None)
    resolved = getattr(ref, "resolved", None)
    content = getattr(resolved, "content", None)
    author = getattr(resolved, "author", None)

    if not content or not content.strip() or not hasattr(author, "display_name"):
        return ""

    display_name = author.display_name
    text = clean_message_text(content)
    if len(text) > 500:
        text = text[:497] + "..."
    return f'[replying to {display_name}: "{text}"]\n'


def format_reply(answer: str) -> str:
    """Post-process LLM output for cleaner Discord rendering.

    Currently a light pass-through; the LLM already produces markdown.
    Kept as a seam for future formatting rules.
    """
    return answer.strip()


HELP_TEXT = (
    "**chord** — chat with me by mentioning me (`@chord 서울 날씨 어때?`) "
    "or reply to any message.\n"
    "`/help` — show this message\n"
    "`/usage` — show remaining API quotas\n"
    "`/reminders` — list pending reminders\n"
    "`/reset` — forget this channel's conversation\n"
    "`/persona` — view or reload the character definition"
)


class ChordBot(discord.Client):
    """Discord client that answers mentions and slash commands."""

    def __init__(
        self,
        settings: Settings,
        engine: ChatEngine,
        registry: SkillRegistry | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.me: discord.ClientUser | None = None
        self.tree = app_commands.CommandTree(self)

        self._settings = settings
        self._engine = engine
        self._persona = PersonaProvider(settings.persona_path)
        self._reminders = ReminderStore(settings.reminder_db_path)
        self._store = ConversationStore()
        self._registry = registry or SkillRegistry()
        self._mcp = McpManager()

    async def setup_hook(self) -> None:
        """Called once after login but before the gateway connects."""
        self._register_slash_commands()
        registered = await self._mcp.start(self._settings, self._registry.register)
        if registered:
            logger.info("Registered %d MCP tool(s).", registered)
        await self.tree.sync()
        self._mcp_reload_loop.start()
        self._reminder_loop.start()

    def _register_slash_commands(self) -> None:
        """Register all slash commands on the command tree."""

        @self.tree.command(name="help", description="Show what chord can do")
        async def help_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(HELP_TEXT, ephemeral=True)

        @self.tree.command(name="usage", description="Show remaining API quotas per provider")
        async def usage_cmd(interaction: discord.Interaction) -> None:
            store = get_quota_store(self._settings.quota_store_path)
            await interaction.response.send_message(render_usage(store), ephemeral=True)

        @self.tree.command(
            name="reminders",
            description="List pending reminders in this channel",
        )
        async def reminders_cmd(interaction: discord.Interaction) -> None:
            rows = self._reminders.pending_for_channel(interaction.channel_id)
            if not rows:
                await interaction.response.send_message(
                    "No pending reminders in this channel.", ephemeral=True
                )
                return
            lines = [f"**{len(rows)} pending reminder(s)**"]
            for row in rows:
                local = row.due.astimezone().strftime("%m-%d %H:%M")
                lines.append(f"> `#{row.id}` {local} — {row.text}")
            await interaction.response.send_message("\n".join(lines))

        @self.tree.command(
            name="reset",
            description="Clear this channel's conversation memory",
        )
        async def reset_cmd(interaction: discord.Interaction) -> None:
            self._store.reset(interaction.channel_id)
            await interaction.response.send_message("🧹 Conversation cleared.", ephemeral=True)

        @self.tree.command(
            name="persona",
            description="View or reload the character definition",
        )
        @app_commands.describe(action="'view' shows current, 'reload' refreshes from file")
        @app_commands.choices(
            action=[
                app_commands.Choice(name="view", value="view"),
                app_commands.Choice(name="reload", value="reload"),
            ]
        )
        async def persona_cmd(interaction: discord.Interaction, action: str = "reload") -> None:
            if action == "view":
                prompt = self._engine.system_prompt
                preview = prompt[:300] + ("..." if len(prompt) > 300 else "")
                await interaction.response.send_message(
                    f"Current persona:\n```\n{preview}\n```", ephemeral=True
                )
            else:
                changed = self._persona.refresh()
                self._engine.system_prompt = self._persona.get()
                msg = "🔄 Persona reloaded from file." if changed else "✅ Persona unchanged."
                await interaction.response.send_message(msg, ephemeral=True)

    # -- Background loops -------------------------------------------------------

    @tasks.loop(minutes=30)
    async def _mcp_reload_loop(self) -> None:
        """Periodically re-read mcp.json and hot-reload on changes."""
        try:
            changed = await self._mcp.reload_if_changed(
                self._settings, self._registry, self._registry.register
            )
            if changed:
                logger.info("MCP tools refreshed from mcp.json.")
        except Exception:  # noqa: BLE001
            logger.exception("MCP reload failed; keeping previous servers.")

    @_mcp_reload_loop.before_loop
    async def _wait_gateway_mcp(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(seconds=30)
    async def _reminder_loop(self) -> None:
        """Deliver reminders whose time has come."""
        delivered = await self.deliver_due_reminders()
        if delivered:
            logger.info("Delivered %d reminder(s).", delivered)

    @_reminder_loop.before_loop
    async def _wait_gateway_reminders(self) -> None:
        await self.wait_until_ready()

    def _resolve_channel(self, channel_id: int):
        """Find a channel by id (cached guild channels first)."""
        return self.get_channel(channel_id)

    async def deliver_due_reminders(self) -> int:
        """Send every due reminder; returns count delivered."""
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
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Could not deliver reminder #%s to channel %s",
                    reminder.id,
                    reminder.channel_id,
                )
        return count

    async def close(self) -> None:
        self._reminder_loop.cancel()
        self._mcp_reload_loop.cancel()
        await self._mcp.stop()
        await super().close()

    # -- Events -----------------------------------------------------------------

    async def on_ready(self) -> None:
        self.me = self.user
        logger.info("Logged in as %s (id=%s)", self.user, self.user and self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not self._should_reply(message):
            return

        user_text = clean_message_text(message.content)
        reply_context = build_reply_context(message)

        if not user_text and not reply_context:
            await message.channel.send(HELP_TEXT)
            return

        # Build the full prompt including reply context.
        prompt_text = reply_context + user_text if reply_context else user_text
        if not prompt_text.strip():
            await message.channel.send(HELP_TEXT)
            return

        channel_id = message.channel.id
        self._engine.system_prompt = self._persona.get()
        token = set_current_channel(channel_id)

        async with message.channel.typing():
            try:
                answer, new_messages = await self._engine.reply(
                    prompt_text, self._store.history(channel_id)
                )
            except Exception:
                logger.exception("Chat failed in channel %s", channel_id)
                await message.channel.send("Sorry — something went wrong on my side.")
                return
            finally:
                reset_current_channel(token)

        self._store.append(channel_id, *new_messages)
        answer = format_reply(answer)
        for chunk in split_message(answer):
            await message.channel.send(chunk)

    # -- Internals ---------------------------------------------------------------

    def _should_reply(self, message: discord.Message) -> bool:
        """Answer DMs, server mentions, and replies to bot messages."""
        if message.guild is None:
            return True
        if self.me in (message.mentions or []):
            return True
        ref = getattr(message, "reference", None)
        resolved = getattr(ref, "resolved", None)
        return resolved is not None and getattr(resolved, "author", None) == self.me


def build_bot(settings: Settings) -> ChordBot:
    """Create a fully wired bot instance from settings."""
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
