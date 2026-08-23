"""Tests for chord.engine - the tool-calling chat loop."""

from __future__ import annotations

from openai.types.chat import ChatCompletion

from chord.engine import ChatEngine
from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry

# -- Test doubles ---------------------------------------------------------------


class FakeLLM:
    """Returns scripted completions in order and records requests."""

    def __init__(self, *completions: dict):
        self._completions = list(completions)
        self.requests: list[dict] = []

    async def complete(self, messages, tools=None) -> ChatCompletion:
        # Snapshot the request: the engine keeps appending to the same
        # list across rounds, so a plain reference would show future
        # state instead of what was actually sent.
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return ChatCompletion.model_validate(self._completions.pop(0))


class AddSkill(Skill):
    """Deterministic skill used to observe tool execution."""

    name = "add"
    description = "Add two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    async def run(self, a: float, b: float) -> str:
        return f"{a + b}"


class FlakySkill(Skill):
    name = "flaky"
    description = "Always fails."

    async def run(self) -> str:
        raise RuntimeError("nope")


def _text_completion(text: str) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def _tool_call_completion(*calls: tuple[str, str, str]) -> dict:
    """Build a completion that requests the given (id, name, args) calls."""
    return {
        "id": "c2",
        "object": "chat.completion",
        "created": 2,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for call_id, name, arguments in calls
                    ],
                },
            }
        ],
    }


def _engine(llm: FakeLLM) -> ChatEngine:
    registry = SkillRegistry()
    registry.register(AddSkill())
    return ChatEngine(
        llm=llm,  # type: ignore[arg-type]
        registry=registry,
        system_prompt="sys",
    )


# -- Plain answers ----------------------------------------------------------------


async def test_plain_answer_without_tools():
    llm = FakeLLM(_text_completion("hi there"))
    engine = _engine(llm)

    answer, new_messages = await engine.reply("hello", [])

    assert answer == "hi there"
    # Only the user message and the answer should be remembered.
    assert new_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    # System prompt is sent but not stored as part of the turn.
    sent = llm.requests[0]["messages"]
    assert sent[0] == {"role": "system", "content": "sys"}


async def test_history_is_included_after_system_prompt():
    llm = FakeLLM(_text_completion("ok"))
    engine = _engine(llm)
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
    ]

    await engine.reply("next question", history)

    sent = llm.requests[0]["messages"]
    assert sent[1:] == [*history, {"role": "user", "content": "next question"}]


# -- Tool-calling round trip --------------------------------------------------------


async def test_tool_call_round_trip():
    llm = FakeLLM(
        _tool_call_completion(("call-1", "add", '{"a": 2, "b": 3}')),
        _text_completion("the sum is 5"),
    )
    engine = _engine(llm)

    answer, new_messages = await engine.reply("what is 2+3?", [])

    assert answer == "the sum is 5"

    # Second request must contain the assistant tool-call plus the result.
    second = llm.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "add"
    assert second[-1] == {"role": "tool", "tool_call_id": "call-1", "content": "5"}

    # Stored turn keeps the full trace for the model's next request.
    roles = [m["role"] for m in new_messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_multiple_tool_calls_in_one_round():
    llm = FakeLLM(
        _tool_call_completion(
            ("c1", "add", '{"a": 1, "b": 1}'),
            ("c2", "add", '{"a": 2, "b": 2}'),
        ),
        _text_completion("2 and 4"),
    )
    engine = _engine(llm)

    await engine.reply("1+1 and 2+2?", [])

    second = llm.requests[1]["messages"]
    tool_results = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["c1", "c2"]
    assert [m["content"] for m in tool_results] == ["2", "4"]


async def test_unknown_tool_error_is_fed_back_not_raised():
    llm = FakeLLM(
        _tool_call_completion(("c1", "does_not_exist", "{}")),
        _text_completion("sorry, no such tool"),
    )
    engine = _engine(llm)

    answer, _ = await engine.reply("do the thing", [])

    second = llm.requests[1]["messages"]
    assert "unknown tool" in second[-1]["content"]
    assert answer == "sorry, no such tool"


async def test_tool_error_text_reaches_the_model():
    registry = SkillRegistry()
    registry.register(FlakySkill())
    llm = FakeLLM(
        _tool_call_completion(("c1", "flaky", "{}")),
        _text_completion("it broke"),
    )
    engine = ChatEngine(llm=llm, registry=registry, system_prompt="sys")  # type: ignore[arg-type]

    await engine.reply("run it", [])

    second = llm.requests[1]["messages"]
    assert "Error while running 'flaky'" in second[-1]["content"]
    assert "nope" in second[-1]["content"]


async def test_tool_loop_limit_returns_fallback():
    # Every completion asks for another tool call -> loop must stop.
    endless = _tool_call_completion(("c1", "add", '{"a": 0, "b": 0}'))
    llm = FakeLLM(*[dict(endless) for _ in range(10)])
    engine = _engine(llm)

    answer, new_messages = await engine.reply("loop", [])

    assert "stuck" in answer
    assert len(llm.requests) == 6  # DEFAULT_MAX_TOOL_ROUNDS
    assert new_messages[-1]["role"] == "assistant"
