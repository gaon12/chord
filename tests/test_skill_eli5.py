"""Tests for the ELI5 skill."""

from __future__ import annotations

from chord.skills.eli5 import DEFAULT_AUDIENCE, STYLES, Eli5Skill, build_instruction
from fakes import FakeLLM


async def test_eli5_passes_topic_and_audience():
    llm = FakeLLM(reply="a database is like a toy box")
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(topic_or_text="database index", audience="age 5")

    assert result == "a database is like a toy box"
    messages = llm.calls[0]["messages"]
    system = messages[0]["content"]
    assert "CALIBRATE" in system
    assert "The audience is: age 5." in system
    # The user turn carries the raw topic untouched.
    assert messages[-1]["content"] == "database index"


async def test_eli5_default_audience_is_age_five():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="git merge conflicts")

    system = llm.calls[0]["messages"][0]["content"]
    assert f"The audience is: {DEFAULT_AUDIENCE}." in system


async def test_eli5_custom_audience_reaches_system_prompt():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="recursion", audience="my manager")

    assert "The audience is: my manager." in llm.calls[0]["messages"][0]["content"]


async def test_eli5_style_directives():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="x", style="short")
    short_system = llm.calls[-1]["messages"][0]["content"]
    assert STYLES["short"] in short_system

    await skill.run(topic_or_text="x", style="structured")
    structured_system = llm.calls[-1]["messages"][0]["content"]
    assert STYLES["structured"] in structured_system


async def test_eli5_unknown_style_falls_back_to_auto():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    await skill.run(topic_or_text="x", style="haiku")

    system = llm.calls[0]["messages"][0]["content"]
    # 'auto' adds no style directive beyond the audience line.
    assert system.count("The audience is:") == 1
    assert "80 words" not in system
    assert "bullet" not in system.lower()


async def test_eli5_empty_input_short_circuits():
    llm = FakeLLM()
    skill = Eli5Skill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(topic_or_text="   ")

    assert "explain" in result.lower()
    assert not llm.calls


def test_instruction_covers_calibration_axes_and_shape():
    prompt = build_instruction()
    for axis in ("VOCABULARY", "ANALOGIES", "TONE", "DEPTH", "FRAMING"):
        assert axis in prompt
    # The fixed output shape is part of the instruction too.
    for marker in ("ONE sentence", "takeaway", "HARD RULES"):
        assert marker in prompt


def test_tool_definition_shape():
    tool = Eli5Skill(FakeLLM()).to_openai_tool()  # type: ignore[arg-type]
    function = tool["function"]
    assert function["name"] == "explain_eli5"
    assert function["parameters"]["required"] == ["topic_or_text"]
    properties = function["parameters"]["properties"]
    assert set(properties) == {"topic_or_text", "audience", "style"}
    assert properties["style"]["enum"] == ["auto", "short", "structured", "story"]
