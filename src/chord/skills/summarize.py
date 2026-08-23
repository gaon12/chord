"""Summarize skill - condenses long text into a few sentences."""

from __future__ import annotations

from typing import ClassVar

from chord.skills._llm_transform import LLMSkill


class SummarizeSkill(LLMSkill):
    name = "summarize_text"
    description = (
        "Summarize a long piece of text (an article, document, or long "
        "chat log) into a few key sentences. Keeps the original language."
    )
    instruction: ClassVar[str] = (
        "You are a precise summarizer. Write the summary in the same "
        "language as the text, keep only the essential points, and do "
        "not add information that is not in the text."
    )

    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize.",
            },
            "max_sentences": {
                "type": "integer",
                "description": "Maximum number of sentences (default 3).",
            },
        },
        "required": ["text"],
    }

    async def run(self, text: str, max_sentences: int = 3) -> str:
        if not text.strip():
            return "There is nothing to summarize."
        summary = await self.transform(
            text, extra_instruction=f"Use at most {max(int(max_sentences), 1)} sentences."
        )
        return summary or "The text was too short to summarize."
