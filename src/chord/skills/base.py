"""Base class every built-in skill inherits from.

A *skill* is one tool the LLM may call. To add a new one:

1. Create ``chord/skills/<your_skill>.py`` with a small subclass of Skill.
2. Register it in ``chord/skills/__init__.py`` (one line).
3. Done - the registry turns it into an OpenAI tool definition
   automatically, so the model can call it by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from openai.types.chat import ChatCompletionToolParam


class Skill(ABC):
    """One callable tool for the LLM.

    Class attributes double as the OpenAI tool definition:

    * ``name``        - the function name the model will call.
    * ``description`` - shown to the model; make it say *when* to use
      the skill. Clear descriptions are the main quality lever for
      tool calling.
    * ``parameters``  - JSON Schema for the arguments.
    """

    #: Unique function name, snake_case, as the model will call it.
    name: str = ""

    #: One or two sentences telling the model what the tool does.
    description: str = ""

    #: JSON Schema describing accepted arguments.
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute the skill and return the result as text.

        The return value is fed back to the LLM, so it should be
        concise, self-explanatory text (not raw JSON dumps).
        """

    def to_openai_tool(self) -> ChatCompletionToolParam:
        """Render this skill as an OpenAI function-tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
