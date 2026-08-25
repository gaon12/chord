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

import json
import logging
import re
from typing import Any

import discord
from discord import app_commands
from discord.ext import tasks
from openai import APITimeoutError, BadRequestError, RateLimitError

from chord.compaction import HistoryCompactor
from chord.config import REASONING_EFFORT_BY_LEVEL, REASONING_LEVELS, Settings
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

#: Matches user (<@123>, <@!123>) and role (<@&123>) mention tokens.
_MENTION_RE = re.compile(r"<@[!&]?\d+>")


def clean_message_text(content: str) -> str:
    """Strip mention tokens and tidy whitespace from a raw message."""
    cleaned = _MENTION_RE.sub(" ", content)
    return " ".join(cleaned.split())


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split long text into sendable chunks without breaking markdown fences.

    Code blocks (```` ``` ````) are treated as atomic units — never split mid-fence.
    Other text splits at blank lines or newlines.
    """
    text = text.strip()
    if not text:
        return ["(empty reply)"]
    if len(text) <= limit:
        return [text]

    segments = _split_segments(text)
    chunks = _merge_chunks(segments, limit)
    return chunks


def _split_segments(text: str) -> list[str]:
    """Split text into paragraphs and intact code-fence blocks."""
    segments: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            segments.append(content)
        buffer.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if in_fence:
                buffer.append(line)
                flush()  # close fence -> emit whole block
                in_fence = False
            else:
                flush()
                buffer.append(line)
                in_fence = True
        elif in_fence:
            buffer.append(line)
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    return [s for s in segments if s]


def _merge_chunks(segments: list[str], limit: int) -> list[str]:
    """Merge adjacent segments; hard-split any that exceed ``limit``."""
    chunks: list[str] = []
    for seg in segments:
        if chunks and len(chunks[-1]) + 2 + len(seg) <= limit:
            chunks[-1] += "\n\n" + seg
            continue

        if len(seg) > limit:
            # Hard-split oversized segments at newline or space boundaries.
            while len(seg) > limit:
                cut = seg.rfind("\n", 0, limit)
                if cut < limit // 2:
                    cut = seg.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                chunks.append(seg[:cut].rstrip())
                seg = seg[cut:].lstrip()
            if seg:
                if chunks and len(chunks[-1]) + 2 + len(seg) <= limit:
                    chunks[-1] += "\n\n" + seg
                else:
                    chunks.append(seg)
        else:
            chunks.append(seg)
    return [c for c in chunks if c]


#: Discord caps display names at 32 characters; reusing that cap keeps a
#: joke nickname from crowding out the actual question.
MAX_SPEAKER_NAME = 32


def speaker_name(author: Any) -> str:
    """Readable name for whoever sent a message.

    Prefers the per-server nickname (``display_name``) because that is
    the name the other people in the channel actually see; falls back to
    the account name, then to a placeholder.
    """
    raw = getattr(author, "display_name", None) or getattr(author, "name", None)
    cleaned = " ".join(str(raw or "").replace("[", "(").replace("]", ")").split())
    return cleaned[:MAX_SPEAKER_NAME] or "unknown"


def label_speaker(name: str, text: str) -> str:
    """Tag a message with its author so the model can tell people apart.

    A channel is a group chat, but every participant's words arrive as
    the same anonymous ``user`` role - so without a label the model
    reads a whole room as one person and answers accordingly ("you said
    earlier..." to someone who never said it). The OpenAI ``name`` field
    would be the tidier home for this, but plenty of compatible
    providers drop or reject it, so the label rides inside the content
    where nothing can strip it.

    It is a hint, not authentication: anyone can type the same shape
    into a message. Brackets inside nicknames are folded to parentheses
    so a forged label at least cannot look identical to a real one.
    """
    return f"[{name}]: {text}"


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


#: Tag names open models use to think out loud inside the answer body.
_REASONING_TAG = r"thought|thinking|think|reasoning|reflection"

#: A complete <thought>...</thought> block, tag-matched so a stray
#: </think> cannot close a <thought>.
_REASONING_BLOCK_RE = re.compile(
    rf"<(?P<tag>{_REASONING_TAG})\b[^>]*>.*?</(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)

#: An opening tag that never closes: the rest is scratch work.
_UNCLOSED_REASONING_RE = re.compile(
    rf"<(?:{_REASONING_TAG})\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)

#: A closing tag with no opener: everything before it is scratch work.
_ORPHAN_REASONING_END_RE = re.compile(
    rf"\A.*</(?:{_REASONING_TAG})\s*>",
    re.DOTALL | re.IGNORECASE,
)

#: Fenced code blocks are quoted verbatim - a snippet about <think> tags
#: is content, not reasoning.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def strip_reasoning_blocks(text: str) -> str:
    """Remove chain-of-thought the model leaked into its answer.

    Models like gemma-4-31b-it narrate their reasoning inline as
    ``<thought>...</thought>`` before the real reply, which Discord
    renders as a wall of first-person deliberation. Discord users want
    the answer; the thinking is controlled by ``REASONING_LEVEL``, not
    displayed.

    Text inside fenced code blocks is preserved untouched, and if
    stripping would leave nothing at all the original text is returned -
    a leaked thought still beats an empty message.
    """
    stripped = _strip_outside_code_fences(text)
    return stripped if stripped.strip() else text


def _strip_outside_code_fences(text: str) -> str:
    """Apply reasoning removal to everything except fenced code."""
    parts: list[str] = []
    cursor = 0
    for fence in _CODE_FENCE_RE.finditer(text):
        parts.append(_strip_reasoning(text[cursor : fence.start()]))
        parts.append(fence.group(0))
        cursor = fence.end()
    parts.append(_strip_reasoning(text[cursor:]))
    return "".join(parts)


def _strip_reasoning(text: str) -> str:
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _ORPHAN_REASONING_END_RE.sub("", text)
    return _UNCLOSED_REASONING_RE.sub("", text)


def format_reply(answer: str) -> str:
    """Post-process LLM output for cleaner Discord rendering.

    The LLM already produces markdown, so this only removes reasoning
    the model was not supposed to show.
    """
    return strip_reasoning_blocks(answer).strip()


HELP_TEXT = (
    "**chord** — chat with me by mentioning me (`@chord 서울 날씨 어때?`) "
    "or reply to any message.\n"
    "`/help` — show this message\n"
    "`/usage` — show remaining API quotas\n"
    "`/reminders` — list pending reminders\n"
    "`/reset` — forget this channel's conversation\n"
    "`/persona` — view or reload the character definition\n"
    "`/reasoning` — view or set how hard I think before answering"
)

#: One line per level for /reasoning, so the trade-off is visible right
#: where the choice is actually made.
REASONING_LEVEL_HELP: dict[str, str] = {
    "auto": "provider default (parameter not sent)",
    "none": "answer immediately — fastest, best for chat",
    "light": "a little thinking on hard questions",
    "medium": "balanced",
    "heavy": "think it through — slowest",
}


#: Rough JSON-schema characters per token. Calibrated against the
#: provider's own ``usage.prompt_tokens`` on this project: growing the
#: catalog by 11 086 schema characters cost 2 866 tokens, and by 25 250
#: characters cost 8 947 - i.e. 2.8-3.9 chars/token. Only ever used to
#: decide whether to print a warning, never for accounting.
_SCHEMA_CHARS_PER_TOKEN = 3.5

#: Estimated prompt tokens of tool schemas above which the catalog is
#: worth complaining about. Every tool definition is re-sent with every
#: request, so this is charged per message and counts against the
#: provider's input-token rate limit. The Gemini free tier allows 16 000
#: input tokens per minute; past ~8 000 a single tool-calling turn (two
#: requests) already exceeds it and the provider starts answering 429.
LARGE_TOOL_PROMPT_TOKENS = 8000


def estimate_tool_prompt_tokens(tools: list[dict]) -> int:
    """Approximate what the tool catalog adds to every single request."""
    if not tools:
        return 0
    return int(len(json.dumps(tools, ensure_ascii=False)) / _SCHEMA_CHARS_PER_TOKEN)


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
        super().__init__(intents=intents)

        # Set in on_ready from self.user; used for mention matching.
        self._me_id: int | None = None
        self.tree = app_commands.CommandTree(self)

        self._settings = settings
        self._engine = engine
        self._persona = PersonaProvider(settings.persona_path)
        self._reminders = ReminderStore(settings.reminder_db_path)
        self._store = ConversationStore(settings.history_max_messages)
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
        self._register_slash_commands()
        registered = await self._mcp.start(self._settings, self._registry.register)
        if registered:
            logger.info("Registered %d MCP tool(s).", registered)
        warn_if_tool_catalog_is_large(self._registry.to_openai_tools())
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

        @self.tree.command(
            name="reasoning",
            description="View or set how hard I think before answering",
        )
        @app_commands.describe(level="Leave empty to see the current setting")
        @app_commands.choices(
            level=[
                app_commands.Choice(name=f"{name} — {REASONING_LEVEL_HELP[name]}", value=name)
                for name in REASONING_LEVELS
            ]
        )
        async def reasoning_cmd(
            interaction: discord.Interaction,
            level: str | None = None,
        ) -> None:
            message = (
                self._describe_reasoning() if level is None else self._set_reasoning_level(level)
            )
            await interaction.response.send_message(message, ephemeral=True)

    # -- Reasoning ----------------------------------------------------------------

    def _describe_reasoning(self) -> str:
        """Render the active level plus what the alternatives mean."""
        current = self._reasoning_level
        lines = [f"🧠 Reasoning level: **{current}** ({REASONING_LEVEL_HELP[current]})"]

        llm = getattr(self._engine, "llm", None)
        if llm is not None and current != "auto" and not llm.reasoning_enabled:
            lines.append(
                "⚠️ This model rejected the reasoning parameter, so the setting has no effect."
            )
        lines.append("Change it with `/reasoning <level>` — " + ", ".join(REASONING_LEVELS))
        return "\n".join(lines)

    def _set_reasoning_level(self, level: str) -> str:
        """Apply a new reasoning level to the live LLM service.

        Runtime only: ``REASONING_LEVEL`` in .env still decides what the
        bot starts with after a restart.
        """
        if level not in REASONING_EFFORT_BY_LEVEL:
            return f"Unknown level {level!r}. Pick one of: " + ", ".join(REASONING_LEVELS)

        llm = getattr(self._engine, "llm", None)
        if llm is None:
            return "This engine has no adjustable LLM service."

        llm.set_reasoning_effort(REASONING_EFFORT_BY_LEVEL[level])
        previous, self._reasoning_level = self._reasoning_level, level
        logger.info("Reasoning level changed: %s -> %s", previous, level)
        return (
            f"🧠 Reasoning level: **{previous}** → **{level}** "
            f"({REASONING_LEVEL_HELP[level]})\n"
            "_Applies to this session; set `REASONING_LEVEL` in .env to make it permanent._"
        )

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

        logger.debug(
            "Answering message from %s in channel %s (reason=%s): %r",
            message.author,
            message.channel.id,
            reason,
            message.content[:100],
        )

        user_text = clean_message_text(message.content)
        reply_context = build_reply_context(message)

        if not user_text.strip() and not reply_context.strip():
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

        # Build the full prompt: reply context first, then the message
        # itself tagged with who sent it.
        prompt_text = f"{reply_context}{label_speaker(speaker_name(message.author), user_text)}"

        channel_id = message.channel.id
        self._engine.system_prompt = self._persona.get()
        token = set_current_channel(channel_id)

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
                reset_current_channel(token)

        self._store.append(channel_id, *new_messages)
        answer = format_reply(answer)
        for chunk in split_message(answer):
            if not await self._send(message.channel, chunk):
                break
        await self._compact_history(channel_id)

    # -- Internals ---------------------------------------------------------------

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

    async def _send(self, channel: Any, content: str) -> bool:
        """Send one message, turning Discord refusals into clear logs.

        A missing "Send Messages" permission is otherwise invisible: the
        bot does all the work and the channel stays empty.

        Returns:
            True when the message actually went out.
        """
        try:
            await channel.send(content)
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
