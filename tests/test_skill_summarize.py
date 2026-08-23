"""Tests for the summarize skill."""

from __future__ import annotations

from chord.skills.summarize import SummarizeSkill
from fakes import FakeLLM


async def test_summarize_sends_instruction_and_text():
    llm = FakeLLM(reply="short version")
    skill = SummarizeSkill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(text="long text ...", max_sentences=2)

    assert result == "short version"
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "summarizer" in messages[0]["content"]
    # Length limit travels inside the system prompt.
    assert "at most 2" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "long text ..."}
    # Text-only sub-task must not advertise tools.
    assert llm.calls[0]["tools"] is None


async def test_summarize_default_sentence_limit():
    llm = FakeLLM()
    skill = SummarizeSkill(llm=llm)  # type: ignore[arg-type]

    await skill.run(text="some text")

    assert "at most 3" in llm.calls[0]["messages"][0]["content"]


async def test_summarize_empty_text_short_circuits():
    llm = FakeLLM()
    skill = SummarizeSkill(llm=llm)  # type: ignore[arg-type]

    result = await skill.run(text="   ")

    assert "nothing to summarize" in result.lower()
    assert not llm.calls  # no LLM call wasted


def test_tool_definition_shape():
    tool = SummarizeSkill(FakeLLM()).to_openai_tool()  # type: ignore[arg-type]
    function = tool["function"]
    assert function["name"] == "summarize_text"
    assert function["parameters"]["required"] == ["text"]
