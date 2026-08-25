"""Answering "너 뭐 할 수 있어?" from the registry rather than from memory.

The model is handed a tool index in its system prompt, but that index is
one truncated line per tool and it is the same text every turn, so a
model asked what it can do tends to answer from its idea of itself: it
forgets the MCP tools that were added this morning, and cheerfully
offers abilities it does not have.

This skill reads the live registry instead. Whatever is registered right
now - built-in skills, plus whatever each MCP server turned out to
expose - is what comes back, which also makes it the fastest way to see
what an MCP server actually loaded.
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry, estimate_tool_prompt_tokens

#: Group heading for everything that ships with chord.
BUILT_IN = "Built-in"

#: Descriptions are trimmed hard: this list is read to find out what
#: exists, and the full text of every tool is already in the request.
MAX_SUMMARY = 70


def _summarize(description: str) -> str:
    """First sentence of a tool description, short enough to scan."""
    text = " ".join((description or "").split())
    text = text.split(". ")[0].rstrip(".")
    if len(text) > MAX_SUMMARY:
        text = text[: MAX_SUMMARY - 3] + "..."
    return text


def group_tools(registry: SkillRegistry) -> dict[str, list[tuple[str, str]]]:
    """Registered tools as ``{group: [(name, summary)]}``.

    MCP tools land under their own server's name, because "which server
    gave me this?" is the question anyone debugging an MCP config is
    actually asking.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for skill in registry.skills():
        server = getattr(skill, "server", None)
        group = f"MCP · {server}" if server else BUILT_IN
        groups.setdefault(group, []).append((skill.name, _summarize(skill.description)))

    for entries in groups.values():
        entries.sort()
    # Built-in first, then MCP servers alphabetically: the shipped tools
    # are the stable half and belong at the top.
    return dict(sorted(groups.items(), key=lambda item: (item[0] != BUILT_IN, item[0])))


def group_costs(registry: SkillRegistry) -> dict[str, int]:
    """Estimated prompt tokens each group adds to every message."""
    costs: dict[str, list[dict]] = {}
    for skill in registry.skills():
        server = getattr(skill, "server", None)
        group = f"MCP · {server}" if server else BUILT_IN
        costs.setdefault(group, []).append(dict(skill.to_openai_tool()))
    return {group: estimate_tool_prompt_tokens(tools) for group, tools in costs.items()}


def render_capabilities(
    registry: SkillRegistry,
    *,
    with_summaries: bool = True,
    with_cost: bool = False,
) -> str:
    """Human-readable listing of everything registered.

    ``with_cost`` prints what each group adds to the prompt of every
    single message. That is the number worth seeing before deciding
    whether an MCP server is earning its place, and it is invisible
    everywhere else.
    """
    groups = group_tools(registry)
    if not groups:
        return "No tools are registered at all - something went wrong at startup."

    costs = group_costs(registry) if with_cost else {}
    header = f"{len(registry)} tool(s) available."
    if with_cost:
        header = f"{len(registry)} tool(s), ~{sum(costs.values()):,} prompt tokens per message."

    lines = [header]
    for group, entries in groups.items():
        cost = f", ~{costs[group]:,} tokens" if with_cost else ""
        lines.append(f"\n{group} ({len(entries)}{cost}):")
        for name, summary in entries:
            lines.append(f"- {name}: {summary}" if (summary and with_summaries) else f"- {name}")
    return "\n".join(lines)


class CapabilitiesSkill(Skill):
    name = "list_capabilities"
    description = (
        "List the tools you actually have right now, with what each one "
        "does. Call this whenever someone asks what you can do (뭐 할 수 "
        "있어, 기능 알려줘, what can you do) instead of answering from "
        "memory - the list changes with the MCP servers that are loaded."
    )
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, registry: SkillRegistry) -> None:
        # The registry this skill lives in. Held rather than copied so
        # the answer includes MCP tools registered after startup.
        self._registry = registry

    async def run(self) -> str:
        return (
            render_capabilities(self._registry)
            + "\n\nDescribe these in the user's own words and group them "
            "naturally - do not read the list out verbatim."
        )
