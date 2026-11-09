"""Discord layer - wires the chat engine to Discord events.

This module coordinates events, channel turns, reminders, and lifecycle.
Text helpers live in discord_text, slash commands in discord_commands, and
message splitting in message_splitter. The engine handles LLM/tool rounds.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import discord
from discord import app_commands
from discord.ext import tasks
from openai import APITimeoutError, BadRequestError, RateLimitError

from chord.attachments import Attachment, collected, reset_attachments, start_collecting
from chord.channel_turns import ChannelTurns
from chord.compaction import HistoryCompactor
from chord.config import Settings
from chord.context import reset_current_channel, set_current_channel
from chord.conversation import ConversationStore
from chord.discord_commands import HELP_TEXT as HELP_TEXT
from chord.discord_commands import describe_reasoning, register_commands, set_reasoning_level
from chord.discord_text import (
    build_attachment_context as build_attachment_context,
)
from chord.discord_text import (
    build_reply_context as build_reply_context,
)
from chord.discord_text import (
    channel_is_nsfw as channel_is_nsfw,
)
from chord.discord_text import (
    clean_message_text as clean_message_text,
)
from chord.discord_text import (
    format_reply as format_reply,
)
from chord.discord_text import (
    label_speaker as label_speaker,
)
from chord.discord_text import (
    speaker_name as speaker_name,
)
from chord.discord_text import (
    strip_reasoning_blocks as strip_reasoning_blocks,
)
from chord.engine import ChatEngine
from chord.llm import LLMService
from chord.mcp_client import McpManager
from chord.message_splitter import split_message as split_message
from chord.persona import PersonaProvider, with_tool_index
from chord.reminders import ReminderStore
from chord.skills import create_default_registry
from chord.skills._http import close_shared_client
from chord.skills.registry import SkillRegistry, estimate_tool_prompt_tokens

logger = logging.getLogger(__name__)

#: Discord hard-limits every message to 2000 characters.
DISCORD_MESSAGE_LIMIT = 2000


#: Estimated prompt tokens of tool schemas above which the catalog is
#: worth complaining about. Every tool definition is re-sent with every
#: request, so this is charged per message and counts against the
#: provider's input-token rate limit. The Gemini free tier allows 16 000
#: input tokens per minute; past ~8 000 a single tool-calling turn (two
#: requests) already exceeds it and the provider starts answering 429.
LARGE_TOOL_PROMPT_TOKENS = 8000


def warn_if_tool_catalog_is_large(tools: list[dict]) -> None:
    """Point at mcp.json when the catalog eats the input-token budget."""
    estimate = estimate_tool_prompt_tokens(tools)
    if estimate < LARGE_TOOL_PROMPT_TOKENS:
        return
    logger.warning(
        "%d tools add roughly %d prompt tokens to EVERY request (they are "
        "re-sent each time, and a tool-calling turn sends them several "
        "times). This is what exhausts input-token rate limits and turns "
        "replies into 429 retries. Trim mcp.json or set MCP_ENABLED=false.",
        len(tools),
        estimate,
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
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())

        # Set in on_ready from self.user; used for mention matching.
        self._me_id: int | None = None
        self._owner_ids: set[int] = set()
        self.tree = app_commands.CommandTree(self)

        self._settings = settings
        self._engine = engine
        self._persona = PersonaProvider(settings.persona_path)
        self._reminders = ReminderStore(settings.reminder_db_path)
        self._store = ConversationStore(settings.history_max_messages)
        self._turns = ChannelTurns()
        self._compactor = HistoryCompactor(
            getattr(engine, "llm", None), settings.history_token_budget
        )
        self._registry = registry or SkillRegistry()
        self._mcp = McpManager()
        # Mirrors the LLM service so /reasoning can report a level name
        # instead of the raw provider-side effort value.
        self._reasoning_level = settings.reasoning_level

    async def setup_hook(self) -> None:
        """Called once after login but before the gateway connects."""
        application = await self.application_info()
        owner_id = getattr(application.team, "owner_id", None) or application.owner.id
        self._owner_ids = {owner_id}
        self._register_slash_commands()
        registered = await self._mcp.start(self._settings, self._registry.register)
        if registered:
            logger.info("Registered %d MCP tool(s).", registered)
        if not self._settings.on_demand_tools:
            warn_if_tool_catalog_is_large(self._registry.to_openai_tools())
        await self.tree.sync()
        self._mcp_reload_loop.start()
        self._reminder_loop.start()

    def _register_slash_commands(self) -> None:
        register_commands(self)

    def _describe_reasoning(self) -> str:
        return describe_reasoning(self)

    def _set_reasoning_level(self, level: str) -> str:
        return set_reasoning_level(self, level)

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
        await close_shared_client()
        llm = getattr(self._engine, "llm", None)
        if llm is not None and hasattr(llm, "aclose"):
            await llm.aclose()
        await super().close()

    # -- Events -----------------------------------------------------------------

    async def on_ready(self) -> None:
        self._me_id = getattr(self.user, "id", None)
        logger.info(
            "Logged in as %s (id=%s) in %d guild(s)",
            self.user,
            self._me_id,
            len(getattr(self, "guilds", ()) or ()),
        )
        if not self.intents.message_content:
            logger.warning(
                "Message Content Intent is off - mentions will arrive empty. "
                "Enable it in the Developer Portal (Bot -> Privileged Gateway Intents)."
            )

    async def on_message(self, message: discord.Message) -> None:
        """Entry point for every message; never lets an error escape.

        discord.py swallows unhandled listener errors into a traceback
        nobody reads, which looks exactly like "the bot ignored me".
        """
        try:
            await self._handle_message(message)
        except Exception:
            logger.exception(
                "Unhandled error while handling message in channel %s",
                getattr(getattr(message, "channel", None), "id", "?"),
            )

    async def _handle_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        reason = self._reply_reason(message)
        if reason is None:
            return

        # History reads, tool effects, replies, and compaction share one turn
        # boundary. Other channels can still run while this channel is busy.
        async with self._turns.hold(message.channel.id):
            await self._reply_to_message(message, reason)

    async def _reply_to_message(self, message: discord.Message, reason: str) -> None:
        logger.debug(
            "Answering message from %s in channel %s (reason=%s): %r",
            message.author,
            message.channel.id,
            reason,
            message.content[:100],
        )

        user_text = clean_message_text(message.content)
        reply_context = build_reply_context(message)
        attachment_context = build_attachment_context(message)

        if not user_text.strip() and not reply_context.strip() and not attachment_context:
            # Bot was mentioned but content is empty - almost certainly
            # means Message Content Intent is not enabled in the portal.
            if message.guild is not None:
                logger.warning(
                    "Mention received but content is empty. "
                    "Enable 'Message Content Intent' in the Developer Portal "
                    "(Bot -> Privileged Gateway Intents)."
                )
                await self._send(
                    message.channel,
                    "멘션은 받았는데 메시지 내용을 읽을 수 없어요. "
                    "Developer Portal에서 **Message Content Intent**를 활성화해 주세요.",
                )
            else:
                await self._send(message.channel, HELP_TEXT)
            return

        # Build the full prompt: context first - what was replied to and
        # what was uploaded - then the message itself, tagged with who
        # sent it.
        prompt_text = (
            f"{reply_context}{attachment_context}"
            f"{label_speaker(speaker_name(message.author), user_text)}"
        )

        channel_id = message.channel.id
        self._engine.system_prompt = self._system_prompt()
        token = set_current_channel(
            channel_id,
            nsfw=channel_is_nsfw(message.channel),
            is_dm=message.guild is None,
        )
        # Skills can produce images (charts); they land here rather than
        # in the tool result the model reads. See chord.attachments.
        attachment_token = start_collecting()
        files: list[Attachment] = []

        async with message.channel.typing():
            try:
                answer, new_messages = await self._engine.reply(
                    prompt_text, self._store.history(channel_id)
                )
            except RateLimitError:
                logger.warning(
                    "Provider rate limit hit in channel %s. Large tool catalogs "
                    "are the usual cause - see the startup warning.",
                    channel_id,
                )
                await self._send(
                    message.channel,
                    "지금 API 사용량 한도에 걸렸어요. 잠시 뒤에 다시 물어봐 주세요.",
                )
                return
            except BadRequestError:
                # A history the provider refuses poisons every later
                # message in the channel: the same rejected messages get
                # re-sent forever. Drop it and let the user carry on
                # rather than making them discover /reset.
                if self._store.history(channel_id):
                    logger.exception(
                        "Provider rejected the request in channel %s; clearing history",
                        channel_id,
                    )
                    self._store.reset(channel_id)
                    await self._send(
                        message.channel,
                        "대화 기록이 꼬여서 초기화했어요. 다시 물어봐 주세요.",
                    )
                else:
                    logger.exception("Chat failed in channel %s with no history", channel_id)
                    await self._send(message.channel, "Sorry — something went wrong on my side.")
                return
            except APITimeoutError:
                logger.warning(
                    "LLM timed out after %ss in channel %s",
                    self._settings.llm_timeout_seconds,
                    channel_id,
                )
                await self._send(
                    message.channel,
                    "생각이 너무 길어져서 중간에 멈췄어요. 다시 물어봐 주세요.",
                )
                return
            except Exception:
                logger.exception("Chat failed in channel %s", channel_id)
                await self._send(message.channel, "Sorry — something went wrong on my side.")
                return
            finally:
                files = collected()
                reset_current_channel(token)
                reset_attachments(attachment_token)

        self._store.append(channel_id, *new_messages)
        answer = format_reply(answer)
        chunks = split_message(answer)
        for index, chunk in enumerate(chunks):
            # Images ride on the last chunk so they appear below the
            # whole answer rather than interrupting it.
            attached = files if index == len(chunks) - 1 else None
            if not await self._send(message.channel, chunk, attached):
                break
        await self._compact_history(channel_id)

    # -- Internals ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Persona, rules, and a menu of the tools that exist right now.

        Built per message rather than cached: persona.md hot-reloads, and
        the MCP catalog changes under a running bot.
        """
        return with_tool_index(self._persona.get(), self._registry.to_openai_tools())

    async def _compact_history(self, channel_id: int) -> None:
        """Summarize old turns once a channel outgrows its token budget.

        Deliberately runs *after* the answer has gone out: the digest
        costs an extra LLM round-trip, and paying for it on the critical
        path would make one unlucky message visibly slower for no gain.
        The budget sits below the real context window precisely so the
        turn that has to fit is the next one, not this one.

        Failure here is never worth an error message to the channel -
        the history simply stays long and the message-count cap takes
        over.
        """
        try:
            result = await self._compactor.compact(self._store.history(channel_id))
        except Exception:  # noqa: BLE001 - housekeeping must not break chat
            logger.exception("History compaction failed in channel %s", channel_id)
            return
        if result is None:
            return

        consumed, replacement = result
        if self._store.replace_prefix(channel_id, consumed, replacement):
            logger.info(
                "Compacted %d message(s) into a summary in channel %s",
                len(consumed),
                channel_id,
            )
        else:
            logger.debug("Channel %s moved on mid-summary; dropping the digest", channel_id)

    async def _send(
        self,
        channel: Any,
        content: str,
        files: list[Attachment] | None = None,
    ) -> bool:
        """Send one message, turning Discord refusals into clear logs.

        A missing "Send Messages" permission is otherwise invisible: the
        bot does all the work and the channel stays empty.

        Returns:
            True when the message actually went out.
        """
        extra = {"files": _discord_files(files)} if files else {}
        try:
            await channel.send(content, **extra)
            return True
        except discord.Forbidden:
            logger.error(
                "Not allowed to send in channel %s. Give the bot 'View Channel' + "
                "'Send Messages' (plus 'Send Messages in Threads' for threads) there.",
                getattr(channel, "id", "?"),
            )
        except discord.HTTPException:
            logger.exception(
                "Discord rejected a message in channel %s", getattr(channel, "id", "?")
            )
            if files:
                # The text is the answer and the chart is the bonus, so a
                # rejected upload (too large, bad type, attachment
                # permission missing) must not take the words with it.
                logger.warning("Retrying without the %d attachment(s).", len(files))
                return await self._send(channel, content)
        return False

    def _is_me(self, user: Any) -> bool:
        """Check if a user object is this bot (race-safe: ID-based)."""
        me_id = self._me_id or getattr(self.user, "id", None)
        if me_id is None or user is None:
            return False
        return getattr(user, "id", None) == me_id

    def _should_reply(self, message: discord.Message) -> bool:
        """Answer DMs, server mentions, and replies to bot messages."""
        return self._reply_reason(message) is not None

    def _reply_reason(self, message: discord.Message) -> str | None:
        """Return *why* this message deserves an answer, else ``None``.

        Split out from :meth:`_should_reply` so the reason can be logged.
        A bot that silently ignores a mention is the hardest possible
        thing to debug from the outside, so every decision is traceable.
        """
        if message.guild is None:
            return "dm"

        my_id = self._me_id or getattr(self.user, "id", None)
        if my_id is None:
            return None

        if my_id in {getattr(u, "id", None) for u in (message.mentions or [])}:
            return "user-mention"

        # Fallback: some gateway payloads (and cache misses) leave
        # `mentions` unresolved while the raw token is still in the text.
        content = message.content or ""
        if f"<@{my_id}>" in content or f"<@!{my_id}>" in content:
            return "mention-token"

        if self._mentions_one_of_my_roles(message):
            return "role-mention"

        ref = getattr(message, "reference", None)
        resolved = getattr(ref, "resolved", None)
        if getattr(getattr(resolved, "author", None), "id", None) == my_id:
            return "reply-to-bot"

        return None

    def _mentions_one_of_my_roles(self, message: discord.Message) -> bool:
        """True when the message pings a role this bot actually holds.

        Typing ``@chord`` in a server very often autocompletes to the
        bot's *integration role* instead of the bot user. Discord then
        delivers the ping in ``role_mentions`` and leaves ``mentions``
        empty, so an ID-only check never fires and the bot looks dead.

        ``@everyone`` (the guild default role) is excluded on purpose -
        answering every mass ping would be spam.
        """
        role_mentions = getattr(message, "role_mentions", None) or []
        if not role_mentions:
            return False

        guild = message.guild
        me = getattr(guild, "me", None)
        my_role_ids = {getattr(role, "id", None) for role in (getattr(me, "roles", None) or [])}
        my_role_ids.discard(None)
        my_role_ids.discard(getattr(getattr(guild, "default_role", None), "id", None))
        if not my_role_ids:
            return False

        return any(getattr(role, "id", None) in my_role_ids for role in role_mentions)


def _discord_files(files: list[Attachment] | None) -> list[discord.File]:
    """Wrap collected bytes as uploads discord.py understands."""
    return [discord.File(io.BytesIO(item.data), filename=item.filename) for item in files or ()]


def build_bot(settings: Settings) -> ChordBot:
    """Create a fully wired bot instance from settings."""
    llm = LLMService(settings)
    registry = create_default_registry(settings, llm=llm)
    engine = ChatEngine(
        llm=llm,
        registry=registry,
        system_prompt=settings.system_prompt,
        on_demand_tools=settings.on_demand_tools,
    )
    logger.info(
        "Bot ready: %d skill(s), model=%s, base_url=%s",
        len(registry),
        llm.model,
        settings.llm_base_url,
    )
    return ChordBot(settings=settings, engine=engine, registry=registry)
