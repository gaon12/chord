"""ELI5 skill - explains anything, tuned to the audience.

Modeled after the ELI5 concept (https://github.com/dreambigou/eli5):
the same topic is explained differently for a 5-year-old, a manager,
an engineer or your parents. The instruction makes the model calibrate
five axes:

* VOCABULARY - no jargon for kids, proper terms for engineers
* ANALOGIES  - toys and playgrounds vs business outcomes
* TONE       - playful for children, professional for directors
* DEPTH      - short and sweet vs nuanced detail
* FRAMING    - impact/risk for managers, UX for designers
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills._llm_transform import LLMSkill

DEFAULT_AUDIENCE = "age 5"

_INSTRUCTION_TEMPLATE = """You are an expert explainer. Explain the user's topic \
or text so that exactly the right audience understands it.

First decide what the audience needs, then calibrate:
- VOCABULARY: match their words; zero unexplained jargon.
- ANALOGIES: use things they already know and love.
- TONE: fit how they like to be spoken to.
- DEPTH: as short as possible, but not shorter than understanding needs.
- FRAMING: lead with what this audience cares about.

Write in the same language the input is written in."""


def build_instruction(audience: str = DEFAULT_AUDIENCE) -> str:
    """Render the system instruction, including the audience line."""
    return f"{_INSTRUCTION_TEMPLATE}\nThe audience is: {audience.strip() or DEFAULT_AUDIENCE}."


class Eli5Skill(LLMSkill):
    name = "explain_eli5"
    description = (
        "Explain a topic or simplify a text for a specific audience "
        "(e.g. 'age 5', 'my manager', 'a designer', 'grandma'). Adapts "
        "vocabulary, analogies, tone and depth to that audience."
    )
    instruction: ClassVar[str] = _INSTRUCTION_TEMPLATE

    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "topic_or_text": {
                "type": "string",
                "description": "The concept to explain, or the text to simplify.",
            },
            "audience": {
                "type": "string",
                "description": (
                    "Who is listening: an age ('age 5', 'age 15'), a role "
                    "('my manager', 'a designer') or a relationship "
                    "('my mom'). Default 'age 5'."
                ),
            },
        },
        "required": ["topic_or_text"],
    }

    async def run(self, topic_or_text: str, audience: str = DEFAULT_AUDIENCE) -> str:
        if not topic_or_text.strip():
            return "Give me something to explain."

        # The audience travels through extra_instruction so the shared
        # class-level instruction stays immutable and thread-safe.
        return await self.transform(
            topic_or_text,
            extra_instruction=f"The audience is: {audience.strip() or DEFAULT_AUDIENCE}.",
        )
