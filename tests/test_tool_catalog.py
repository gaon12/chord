"""On-demand schemas must save prompt space without allowing guessed actions."""

import json

import pytest
from openai.types.chat import ChatCompletion

from chord.engine import ChatEngine
from chord.skills.registry import SkillRegistry
from chord.tool_catalog import SELECT_TOOLS, ToolCatalog
from test_engine import AddSkill, FakeLLM, _text_completion, _tool_call_completion


def _calls(*calls):
    return (
        ChatCompletion.model_validate(_tool_call_completion(*calls)).choices[0].message.tool_calls
    )


def _registry():
    registry = SkillRegistry()
    registry.register(AddSkill())
    return registry


async def test_greeting_offers_only_the_selection_schema():
    llm = FakeLLM(_text_completion("hello"))
    engine = ChatEngine(llm, _registry(), "system", on_demand_tools=True)
    await engine.reply("hello", [])
    assert [tool["function"]["name"] for tool in llm.requests[0]["tools"]] == [SELECT_TOOLS]


async def test_selected_tool_schema_is_offered_and_executed_on_the_next_round():
    llm = FakeLLM(
        _tool_call_completion(("select", SELECT_TOOLS, '{"names":["add"]}')),
        _tool_call_completion(("sum", "add", '{"a":2,"b":3}')),
        _text_completion("5"),
    )
    engine = ChatEngine(llm, _registry(), "system", on_demand_tools=True)
    answer, history = await engine.reply("add", [])
    assert answer == "5"
    assert [tool["function"]["name"] for tool in llm.requests[1]["tools"]] == [SELECT_TOOLS, "add"]
    assert history[-2]["content"] == "5"


async def test_selection_is_per_turn_and_does_not_leak_across_channels():
    first = ToolCatalog(_registry(), True)
    await first.execute_calls(_calls(("select", SELECT_TOOLS, '{"names":["add"]}')))
    second = ToolCatalog(_registry(), True)
    assert len(first.definitions()) == 2
    assert len(second.definitions()) == 1


async def test_a_guessed_tool_call_cannot_run_even_beside_a_selection():
    catalog = ToolCatalog(_registry(), True)
    results = await catalog.execute_calls(
        _calls(
            ("select", SELECT_TOOLS, '{"names":["add"]}'),
            ("guess", "add", '{"a":2,"b":3}'),
        )
    )
    assert results[1].startswith("Error: select this tool first")
    assert await catalog.execute_calls(_calls(("next", "add", '{"a":2,"b":3}'))) == ["5"]


@pytest.mark.parametrize(
    "arguments",
    [
        "null",
        "[]",
        "{}",
        '{"names":"add"}',
        '{"names":[]}',
        '{"names":[1]}',
        '{"names":["add","add"]}',
        '{"names":["unknown"]}',
        '{"names":["add"], "extra":true}',
        "{" * 1000,
    ],
)
async def test_bad_selections_do_not_expose_tools(arguments):
    catalog = ToolCatalog(_registry(), True)
    result = await catalog.execute_calls(_calls(("select", SELECT_TOOLS, arguments)))
    assert result[0].startswith("Error:")
    assert len(catalog.definitions()) == 1


def test_builtin_catalog_schema_cost_drops_by_more_than_ninety_percent():
    from chord.skills import create_default_registry
    from test_groq_provider import settings

    registry = create_default_registry(settings(), llm=FakeLLM())
    full = json.dumps(ToolCatalog(registry, False).definitions())
    initial = json.dumps(ToolCatalog(registry, True).definitions())
    assert len(initial) < len(full) / 10


def test_auto_catalog_mode_only_changes_groq_free_mode():
    from test_groq_provider import settings

    assert settings().on_demand_tools
    assert not settings(groq_free_tier=False).on_demand_tools
    assert not settings(tool_catalog_mode="full").on_demand_tools
