"""Offer complete tool schemas only after the model selects the tools it needs."""

from __future__ import annotations

import json

from openai.types.chat import ChatCompletionMessageToolCall

from chord.skills.registry import SkillRegistry

SELECT_TOOLS = "chord_select_tools"
SELECTION_RULE = (
    "Tool schemas are loaded on demand. Before calling a tool from AVAILABLE TOOLS, "
    "use chord_select_tools with its exact name. Selected schemas become available "
    "on the next request. For ordinary conversation, answer directly without selecting tools."
)
_SELECT_DEFINITION = {
    "type": "function",
    "function": {
        "name": SELECT_TOOLS,
        "description": "Load the schemas of tools needed to answer the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                    "uniqueItems": True,
                }
            },
            "required": ["names"],
            "additionalProperties": False,
        },
    },
}


class ToolCatalog:
    """A catalog snapshot owned by one turn, never shared between channels."""

    def __init__(self, registry: SkillRegistry, on_demand: bool) -> None:
        self._registry = registry
        self._on_demand = on_demand
        definitions = registry.to_openai_tools()
        self._definitions = {tool["function"]["name"]: tool for tool in definitions}
        self._selected: set[str] = set()
        if on_demand and SELECT_TOOLS in self._definitions:
            raise ValueError(f"{SELECT_TOOLS} is reserved for tool selection.")

    def definitions(self) -> list[dict]:
        if not self._on_demand:
            return list(self._definitions.values())
        return [
            _SELECT_DEFINITION,
            *(
                definition
                for name, definition in self._definitions.items()
                if name in self._selected
            ),
        ]

    async def execute_calls(self, calls: list[ChatCompletionMessageToolCall]) -> list[str]:
        if not self._on_demand:
            return await self._registry.execute_calls(calls)
        results: list[str] = []
        pending: list[ChatCompletionMessageToolCall] = []

        async def flush() -> None:
            if pending:
                results.extend(await self._registry.execute_calls(pending))
                pending.clear()

        # Permission is based on what this request offered, not on selections
        # earlier in the same completion. The model must see a tool's schema
        # before using it; a guessed tool call must not cause a side effect.
        offered = set(self._selected)
        for call in calls:
            name = call.function.name
            if name == SELECT_TOOLS:
                await flush()
                results.append(self._select(call.function.arguments))
            elif name not in offered:
                await flush()
                results.append("Error: select this tool first with chord_select_tools.")
            else:
                pending.append(call)
        await flush()
        return results

    def _select(self, raw_arguments: str) -> str:
        try:
            if len(raw_arguments) > 4096:
                raise ValueError
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict) or set(arguments) != {"names"}:
                raise ValueError
            names = arguments["names"]
            if not isinstance(names, list) or not 1 <= len(names) <= 8:
                raise ValueError
            if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
                raise ValueError
            if any(name not in self._definitions for name in names):
                return "Error: unknown tool name. Use exact names from AVAILABLE TOOLS."
        except (ValueError, TypeError, RecursionError):
            return "Error: provide names as an array of one to eight unique tool names."
        self._selected.update(names)
        return "Tool schemas loaded for the next request: " + ", ".join(names)
