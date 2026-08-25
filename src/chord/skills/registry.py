"""Skill registry - collects every available tool for the LLM.

The registry is the bridge between Python code and OpenAI-style tool
calling:

* ``to_openai_tools()``  - build the tool list sent with each request
  (built-in skills first, MCP tools merged on top by the bot).
* ``execute()``          - safely run one tool call from the LLM,
  converting both bad arguments and internal errors into readable text
  so the conversation can continue instead of crashing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from chord.skills.base import Skill

logger = logging.getLogger(__name__)

#: Rough JSON-schema characters per token. Calibrated against a
#: provider's own ``usage.prompt_tokens`` on this project: growing the
#: catalog by 11 086 schema characters cost 2 866 tokens, and by 25 250
#: characters cost 8 947 - i.e. 2.8-3.9 chars/token. Only ever used to
#: show or warn about a cost, never for accounting.
SCHEMA_CHARS_PER_TOKEN = 3.5


def estimate_tool_prompt_tokens(tools: list[dict]) -> int:
    """Approximate what a set of tool schemas adds to every request.

    Every tool definition is re-sent with every message, and a
    tool-calling turn sends them several times over, so this is the
    number that decides whether a channel fits inside a per-minute
    input-token limit.
    """
    if not tools:
        return 0
    return int(len(json.dumps(tools, ensure_ascii=False)) / SCHEMA_CHARS_PER_TOKEN)


class SkillRegistry:
    """Simple name -> skill mapping with safe execution."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # -- Registration --------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Add a skill; duplicate names are a programming mistake."""
        if not skill.name:
            raise ValueError(f"{type(skill).__name__} must set a non-empty 'name'.")
        if skill.name in self._skills:
            raise ValueError(f"Duplicate skill name: {skill.name!r}")
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        """Remove a skill by name; returns True if it existed."""
        return self._skills.pop(name, None) is not None

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        """Registered skill names, sorted for stable ordering."""
        return sorted(self._skills)

    def skills(self) -> list[Skill]:
        """Every registered skill, for callers that need more than names."""
        return list(self._skills.values())

    # -- OpenAI tool plumbing -------------------------------------------------

    def to_openai_tools(self) -> list[ChatCompletionToolParam]:
        """All registered skills as OpenAI tool definitions."""
        return [skill.to_openai_tool() for skill in self._skills.values()]

    # -- Execution -------------------------------------------------------------

    async def execute(self, name: str, raw_arguments: str | dict) -> str:
        """Run one skill call coming from the LLM.

        Args:
            name: Tool name chosen by the model.
            raw_arguments: JSON string (or already-parsed dict) with the
                arguments, exactly as produced by the model.

        Returns:
            Always a string: either the skill result or a readable error.
            Returning errors as text keeps the tool-call loop alive - the
            model can apologise, retry or ask the user for better input.
        """
        skill = self._skills.get(name)
        if skill is None:
            logger.warning("LLM called unknown skill %r", name)
            return f"Error: unknown tool '{name}'."

        try:
            arguments = _parse_arguments(raw_arguments)
        except json.JSONDecodeError:
            return f"Error: arguments for '{name}' are not valid JSON."

        try:
            logger.info("Running skill %s(%s)", name, arguments)
            return await skill.run(**arguments)
        except TypeError as exc:
            # Wrong argument names/types against the declared JSON Schema.
            logger.warning("Bad arguments for skill %s: %s", name, exc)
            return f"Error: bad arguments for '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - reported to the LLM below
            logger.exception("Skill %s failed", name)
            message = str(exc) or type(exc).__name__
            return f"Error while running '{name}': {message}"


def _parse_arguments(raw_arguments: str | dict) -> dict[str, Any]:
    """Normalise model-produced arguments into a plain dict."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        parsed = json.loads(raw_arguments)
        if isinstance(parsed, dict):
            return parsed
    return {}
