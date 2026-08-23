"""Shared plumbing for skills that are themselves LLM-powered.

Summarize / translate / eli5 all follow the same shape: send the user's
text to the model with a dedicated instruction and return the answer.
This base class centralizes that call so each concrete skill only
declares its tool metadata and instruction prompt.
"""

from __future__ import annotations

from typing import ClassVar

from openai.types.chat import ChatCompletion

from chord.llm import LLMService
from chord.skills.base import Skill


class LLMSkill(Skill):
    """A skill whose ``run`` is a focused one-shot LLM request."""

    #: System instruction for the sub-task. Concrete skills override
    #: this; it is the single most important knob for output quality.
    instruction: ClassVar[str] = ""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def transform(self, text: str, *, extra_instruction: str = "") -> str:
        """Run the sub-task on ``text`` and return the model's answer.

        Args:
            text: The user-provided content to transform.
            extra_instruction: Optional additional sentence appended to
                the system prompt (e.g. a length limit or audience).
        """
        system = self.instruction
        if extra_instruction:
            system = f"{system} {extra_instruction}".strip()

        completion: ChatCompletion = await self._llm.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
        )
        return completion.choices[0].message.content or ""
