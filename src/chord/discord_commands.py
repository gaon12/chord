"""Slash commands and runtime settings, separate from Discord events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from chord.config import REASONING_EFFORT_BY_LEVEL, REASONING_LEVELS
from chord.message_splitter import split_message
from chord.skills._quota import get_quota_store, render_usage
from chord.skills.capabilities import render_capabilities

if TYPE_CHECKING:
    from chord.bot import ChordBot

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "**chord** — chat with me by mentioning me (`@chord 서울 날씨 어때?`) "
    "or reply to any message.\n"
    "`/help` — show this message\n"
    "`/tools` — list every tool I have (or just ask me what I can do)\n"
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
    "none": "lowest supported thinking effort — fastest",
    "light": "a little thinking on hard questions",
    "medium": "balanced",
    "heavy": "think it through — slowest",
}


def register_commands(bot: ChordBot) -> None:
    """Register all slash commands on the command tree."""

    @bot.tree.command(name="help", description="Show what chord can do")
    async def help_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(HELP_TEXT, ephemeral=True)

    @bot.tree.command(name="tools", description="List every tool I can use right now")
    async def tools_cmd(interaction: discord.Interaction) -> None:
        # Names only: with MCP servers loaded the descriptions run
        # well past Discord's 2000-character ceiling, and this
        # command is read to find out what exists.
        listing = render_capabilities(
            bot._registry,
            with_summaries=False,
            with_cost=True,
            on_demand=bot._settings.on_demand_tools,
        )
        await send_private(interaction, listing)

    @bot.tree.command(name="usage", description="Show remaining API quotas per provider")
    async def usage_cmd(interaction: discord.Interaction) -> None:
        store = get_quota_store(bot._settings.quota_store_path)
        text = render_usage(store)
        llm = getattr(bot._engine, "llm", None)
        if llm is not None and hasattr(llm, "describe_usage"):
            details = llm.describe_usage()
            if details:
                text += "\n\n" + details
        await send_private(interaction, text)

    @bot.tree.command(
        name="reminders",
        description="List pending reminders in this channel",
    )
    async def reminders_cmd(interaction: discord.Interaction) -> None:
        rows = bot._reminders.pending_for_channel(interaction.channel_id)
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

    @bot.tree.command(
        name="reset",
        description="Clear this channel's conversation memory",
    )
    async def reset_cmd(interaction: discord.Interaction) -> None:
        if interaction.guild is not None and not (
            is_owner(bot, interaction) or interaction.permissions.manage_messages
        ):
            await interaction.response.send_message(
                "Manage Messages permission is required to clear a shared conversation.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=True)
        async with bot._turns.hold(interaction.channel_id):
            bot._store.reset(interaction.channel_id)
        await interaction.followup.send("🧹 Conversation cleared.", ephemeral=True)

    @bot.tree.command(
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
        if not await require_owner(bot, interaction):
            return
        if action == "view":
            prompt = bot._system_prompt()
            preview = prompt[:300] + ("..." if len(prompt) > 300 else "")
            await interaction.response.send_message(
                f"Current persona:\n```\n{preview}\n```", ephemeral=True
            )
        else:
            changed = bot._persona.refresh()
            bot._engine.system_prompt = bot._persona.get()
            msg = "🔄 Persona reloaded from file." if changed else "✅ Persona unchanged."
            await interaction.response.send_message(msg, ephemeral=True)

    @bot.tree.command(
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
        if level is not None and not await require_owner(bot, interaction):
            return
        message = bot._describe_reasoning() if level is None else bot._set_reasoning_level(level)
        await interaction.response.send_message(message, ephemeral=True)


def is_owner(bot: ChordBot, interaction: discord.Interaction) -> bool:
    """Global settings belong to the application owner, across all guilds.

    A guild administrator must not be able to change the bot's behavior in
    another guild. Discord application ownership is loaded once at startup.
    """
    return interaction.user.id in bot._owner_ids


async def require_owner(bot: ChordBot, interaction: discord.Interaction) -> bool:
    if is_owner(bot, interaction):
        return True
    await interaction.response.send_message(
        "Only the application owner can view the persona or change global settings.",
        ephemeral=True,
    )
    return False


async def send_private(interaction: discord.Interaction, text: str) -> None:
    """Send every chunk instead of silently discarding a long command result."""
    for index, chunk in enumerate(split_message(text)):
        send = interaction.response.send_message if index == 0 else interaction.followup.send
        await send(chunk, ephemeral=True)


def describe_reasoning(bot: ChordBot) -> str:
    """Render the active level plus what the alternatives mean."""
    current = bot._reasoning_level
    lines = [f"🧠 Reasoning level: **{current}** ({REASONING_LEVEL_HELP[current]})"]

    llm = getattr(bot._engine, "llm", None)
    if llm is not None and current != "auto" and not llm.reasoning_enabled:
        lines.append(
            "⚠️ This model is not using the reasoning parameter, so the setting has no effect."
        )
    if llm is not None and hasattr(llm, "describe_reasoning"):
        details = llm.describe_reasoning()
        if details:
            lines.append(details)
    lines.append("Change it with `/reasoning <level>` — " + ", ".join(REASONING_LEVELS))
    return "\n".join(lines)


def set_reasoning_level(bot: ChordBot, level: str) -> str:
    """Apply a new reasoning level to the live LLM service.

    Runtime only: ``REASONING_LEVEL`` in .env still decides what the
    bot starts with after a restart.
    """
    if level not in REASONING_EFFORT_BY_LEVEL:
        return f"Unknown level {level!r}. Pick one of: " + ", ".join(REASONING_LEVELS)

    llm = getattr(bot._engine, "llm", None)
    if llm is None:
        return "This engine has no adjustable LLM service."

    llm.set_reasoning_effort(REASONING_EFFORT_BY_LEVEL[level])
    previous, bot._reasoning_level = bot._reasoning_level, level
    logger.info("Reasoning level changed: %s -> %s", previous, level)
    return (
        f"🧠 Reasoning level: **{previous}** → **{level}** "
        f"({REASONING_LEVEL_HELP[level]})\n"
        "_Applies to this session; set `REASONING_LEVEL` in .env to make it permanent._"
    )
