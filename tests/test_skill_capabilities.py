"""Tests for list_capabilities - answering "what can you do?" from the registry."""

from __future__ import annotations

from typing import ClassVar

from chord.config import Settings
from chord.skills import create_default_registry
from chord.skills.base import Skill
from chord.skills.capabilities import (
    BUILT_IN,
    CapabilitiesSkill,
    group_costs,
    group_tools,
    render_capabilities,
)
from chord.skills.registry import SkillRegistry


class _Tool(Skill):
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, name: str, description: str = "", server: str | None = None):
        self.name = name
        self.description = description
        if server is not None:
            self.server = server

    async def run(self, **kwargs):  # pragma: no cover - never called
        return ""


def _registry(*skills: Skill) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


# -- Grouping ---------------------------------------------------------------------------


def test_built_in_tools_are_grouped_together():
    registry = _registry(_Tool("get_weather", "Weather."), _Tool("make_qr", "QR."))

    assert list(group_tools(registry)) == [BUILT_IN]
    assert [name for name, _ in group_tools(registry)[BUILT_IN]] == ["get_weather", "make_qr"]


def test_mcp_tools_are_grouped_by_the_server_that_gave_them():
    """ "Which server is this from?" is the MCP debugging question."""
    registry = _registry(
        _Tool("get_weather", "Weather."),
        _Tool("sqlite_query", "Query.", server="sqlite"),
        _Tool("sqlite_list", "List.", server="sqlite"),
        _Tool("fetch_fetch", "Fetch.", server="fetch"),
    )

    groups = group_tools(registry)

    assert list(groups) == [BUILT_IN, "MCP · fetch", "MCP · sqlite"]
    assert len(groups["MCP · sqlite"]) == 2


def test_the_shipped_tools_come_first():
    registry = _registry(_Tool("a_remote", "x", server="aaa"), _Tool("z_local", "y"))

    assert list(group_tools(registry))[0] == BUILT_IN


# -- Rendering ---------------------------------------------------------------------------


def test_the_listing_counts_and_names_everything():
    registry = _registry(_Tool("get_weather", "Get the weather for a city."))

    text = render_capabilities(registry)

    assert "1 tool(s) available." in text
    assert "- get_weather: Get the weather for a city" in text


def test_only_the_first_sentence_of_a_description_is_kept():
    registry = _registry(_Tool("t", "Does one thing. Then three paragraphs of caveats."))

    assert "caveats" not in render_capabilities(registry)


def test_a_rambling_description_is_truncated():
    registry = _registry(_Tool("t", "x" * 500))

    assert max(len(line) for line in render_capabilities(registry).splitlines()) < 100


def test_summaries_can_be_dropped_for_a_compact_listing():
    """Discord caps a message at 2000 characters; MCP blows past that."""
    registry = _registry(_Tool("get_weather", "Get the weather for a city."))

    text = render_capabilities(registry, with_summaries=False)

    assert "- get_weather" in text
    assert "Get the weather" not in text


def test_an_empty_registry_says_something_went_wrong():
    assert "something went wrong" in render_capabilities(SkillRegistry())


# -- The skill ----------------------------------------------------------------------------


async def test_the_skill_reports_the_live_registry():
    registry = _registry(_Tool("get_weather", "Weather."))
    registry.register(CapabilitiesSkill(registry))

    result = await CapabilitiesSkill(registry).run()

    assert "get_weather" in result
    assert "list_capabilities" in result


async def test_tools_registered_after_startup_still_show_up():
    """MCP servers connect after the registry is built, not before."""
    registry = _registry(_Tool("get_weather", "Weather."))
    skill = CapabilitiesSkill(registry)

    registry.register(_Tool("sqlite_query", "Query.", server="sqlite"))
    result = await skill.run()

    assert "MCP · sqlite" in result
    assert "sqlite_query" in result


async def test_the_model_is_told_not_to_read_the_list_aloud():
    result = await CapabilitiesSkill(_registry(_Tool("t", "x"))).run()

    assert "do not read the list out verbatim" in result


def test_the_skill_is_discovered_and_gets_its_registry():
    """It needs the registry injected, which nothing else asks for."""
    registry = create_default_registry(
        Settings(_env_file=None, discord_token="t", openai_api_key="k")
    )

    assert "list_capabilities" in registry


# -- What the catalogue costs ---------------------------------------------------------


def test_cost_is_reported_per_group():
    """Every schema is re-sent with every message; that is the number."""
    registry = _registry(
        _Tool("get_weather", "Weather."),
        _Tool("sqlite_query", "Query.", server="sqlite"),
    )

    costs = group_costs(registry)

    assert set(costs) == {BUILT_IN, "MCP · sqlite"}
    assert all(value > 0 for value in costs.values())


def test_the_listing_can_show_what_each_group_adds_to_a_prompt():
    registry = _registry(
        _Tool("get_weather", "Weather."),
        _Tool("law_search", "Search.", server="korean-law"),
    )

    text = render_capabilities(registry, with_summaries=False, with_cost=True)

    assert "prompt tokens per message" in text
    assert "tokens):" in text
    # An MCP server's price sits next to its name, which is where the
    # decision to keep or drop it actually gets made.
    assert "MCP · korean-law (1, ~" in text


def test_cost_is_off_by_default_because_the_model_does_not_need_it():
    text = render_capabilities(_registry(_Tool("get_weather", "Weather.")))

    assert "tokens" not in text
