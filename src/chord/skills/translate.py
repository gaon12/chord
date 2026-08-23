"""Translate skill - converts text between languages."""

from __future__ import annotations

from typing import ClassVar

from chord.skills._llm_transform import LLMSkill


class TranslateSkill(LLMSkill):
    name = "translate_text"
    description = (
        "Translate text into another language, e.g. Korean to English "
        "or English to Japanese. Outputs only the translation."
    )
    instruction: ClassVar[str] = (
        "You are a professional translator. Translate the user's text "
        "into the requested language. Output ONLY the translation with "
        "no notes, no quotes around it and no explanation. Preserve "
        "tone, formatting and line breaks."
    )

    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to translate.",
            },
            "target_language": {
                "type": "string",
                "description": ("Language to translate into, e.g. 'English', '한국어', '日本語'."),
            },
        },
        "required": ["text", "target_language"],
    }

    async def run(self, text: str, target_language: str) -> str:
        if not text.strip():
            return "There is nothing to translate."
        if not target_language.strip():
            return "Please tell me which language to translate into."

        translation = await self.transform(
            f"[{target_language.strip()}]\n{text}",
        )
        # The target language is prefixed to the user turn (not appended
        # to the system prompt) so the instruction stays cache-friendly
        # and identical across calls.
        return translation or "Translation failed; please try again."
