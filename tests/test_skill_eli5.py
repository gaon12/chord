"""Tests for the ELI5 skill."""

from __future__ import annotations

from chord.skills.eli5 import Eli5Skill, build_instruction
from fakes import FakeLLM


async def test_eli5_passes_topic_and_audience():
    llm = FakeLLM(reply="a database is like a toy box")
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(topic_or_text="database index", audience="age 5")

    assert result == "a database is like a toy box"
    messages = llm.calls[0]["messages"]
    assert "calibrate" in messages[0]["content"]
    assert messages[-1]["content"] == "database index"


async def test_eli5_default_audience_is_age_five():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="git merge conflicts")

    assert "The audience is: age 5." in llm.calls[0]["messages"][0]["content"]


async def test_eli5_custom_audience_reaches_system_prompt():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="recursion", audience="my manager")

    assert "The audience is: my manager." in llm.calls[0]["messages"][0]["content"]


async def test_eli5_empty_input_short_circuits():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(topic_or_text="   ")

    assert "explain" in result.lower()
    assert not llm.calls


def test_instruction_covers_calibration_axes():
    prompt = build_instruction()
    for axis in ("VOCABULARY", "ANALOGIES", "TONE", "DEPTH", "FRAMING"):
        assert axis in prompt


def test_tool_definition_shape():
    tool = Eli5Skill(FakeLLM()).to_openai_tool()  # type: ignore[arg-type]
    function = tool["function"]
    assert function["name"] == "explain_eli5"
    assert function["parameters"]["required"] == ["topic_or_text"]
    properties = function["parameters"]["properties"]
    assert set(properties) == {"topic_or_text", "audience"}
