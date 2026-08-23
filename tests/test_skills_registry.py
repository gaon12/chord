"""Tests for the skill base class and registry."""

from __future__ import annotations

import json

import pytest

from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry


class EchoSkill(Skill):
    """Tiny test double that returns its arguments."""

    name = "echo"
    description = "Echo back the given text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, text: str) -> str:
        return f"echo: {text}"


class BoomSkill(Skill):
    """Test double that always raises."""

    name = "boom"
    description = "Always fails."

    async def run(self) -> str:
        raise RuntimeError("kaboom")


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(EchoSkill())
    return registry


# -- Registration -------------------------------------------------------------


def test_register_and_lookup():
    registry = _registry()
    assert len(registry) == 1
    assert "echo" in registry
    assert "nope" not in registry
    assert registry.names() == ["echo"]


def test_duplicate_name_rejected():
    registry = _registry()
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(EchoSkill())


def test_empty_name_rejected():
    class Anonymous(Skill):
        async def run(self):
            return ""

    registry = SkillRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(Anonymous())


# -- OpenAI tool conversion -----------------------------------------------------


def test_to_openai_tools_shape():
    tools = _registry().to_openai_tools()
    assert len(tools) == 1
    function = tools[0]["function"]
    assert tools[0]["type"] == "function"
    assert function["name"] == "echo"
    assert function["description"] == "Echo back the given text."
    assert function["parameters"]["required"] == ["text"]


# -- Execution --------------------------------------------------------------------


async def test_execute_runs_skill_with_json_arguments():
    result = await _registry().execute("echo", json.dumps({"text": "hi"}))
    assert result == "echo: hi"


async def test_execute_accepts_dict_arguments_too():
    result = await _registry().execute("echo", {"text": "dict"})
    assert result == "echo: dict"


async def test_execute_empty_json_means_no_arguments():
    class NoArgsSkill(Skill):
        name = "ping"
        description = "Returns pong."

        async def run(self) -> str:
            return "pong"

    registry = SkillRegistry()
    registry.register(NoArgsSkill())
    # LLMs sometimes send "" or "{}" for argument-less tools.
    assert await registry.execute("ping", "") == "pong"
    assert await registry.execute("ping", "{}") == "pong"


async def test_execute_unknown_skill_returns_error_text():
    result = await _registry().execute("missing", "{}")
    assert "unknown tool" in result


async def test_execute_invalid_json_returns_error_text():
    result = await _registry().execute("echo", "{not json")
    assert "not valid JSON" in result


async def test_execute_bad_argument_names_return_error_text():
    result = await _registry().execute("echo", json.dumps({"wrong": "arg"}))
    assert "bad arguments" in result


async def test_execute_skill_exception_becomes_error_text():
    registry = SkillRegistry()
    registry.register(BoomSkill())

    result = await registry.execute("boom", "{}")

    assert result.startswith("Error while running 'boom'")
    assert "kaboom" in result
