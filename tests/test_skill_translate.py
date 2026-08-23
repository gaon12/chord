"""Tests for the translate skill."""

from __future__ import annotations

from chord.skills.translate import TranslateSkill
from fakes import FakeLLM


async def test_translate_prefixes_target_language_to_user_turn():
    llm = FakeLLM(reply="hello")
    skill = TranslateSkill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(text="안녕하세요", target_language="English")

    assert result == "hello"
    messages = llm.calls[0]["messages"]
    assert "professional translator" in messages[0]["content"]
    assert "[english]" in messages[-1]["content"].lower()
    assert "안녕하세요" in messages[-1]["content"]


async def test_translate_empty_text_short_circuits():
    llm = FakeLLM()
    skill = TranslateSkill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(text="", target_language="English")

    assert "nothing to translate" in result.lower()
    assert not llm.calls


async def test_translate_missing_language_reports_error_without_llm_call():
    llm = FakeLLM()
    skill = TranslateSkill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(text="hi", target_language="  ")

    assert "which language" in result.lower()
    assert not llm.calls


def test_tool_definition_shape():
    tool = TranslateSkill(FakeLLM()).to_openai_tool()  # type: ignore[arg-type]
    function = tool["function"]
    assert function["name"] == "translate_text"
    assert function["parameters"]["required"] == ["text", "target_language"]
