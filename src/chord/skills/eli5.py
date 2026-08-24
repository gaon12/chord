"""ELI5 skill - explains anything, tuned precisely to the audience.

Modeled after the ELI5 concept (https://github.com/dreambigou/eli5):
the same topic lands differently for a 5-year-old, an engineer, your
manager or your parents. The prompt makes the model work in three
steps - identify the audience, calibrate five axes, then write in a
fixed shape - instead of vaguely "being simple":

* VOCABULARY - zero unexplained jargon; terms defined inline when needed
* ANALOGIES  - drawn from what that audience already knows and likes
* TONE       - playful for kids, professional for executives, warm for family
* DEPTH      - as short as understanding allows, not shorter
* FRAMING    - impact/risk for managers, UX for designers, mechanics for engineers

Optional output styles: ``short`` (<80 words), ``structured``
(bullet points) and ``story`` (a tiny narrative).
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills._llm_transform import LLMSkill

DEFAULT_AUDIENCE = "age 5"

#: Output styles the model may be asked to follow.
STYLES: dict[str, str] = {
    "auto": "",
    "short": "Keep the whole answer under 80 words.",
    "structured": (
        "Format as short bullet points: one core-idea bullet, two to "
        "four explanation bullets, one takeaway bullet."
    ),
    "story": "Explain through a tiny story (under 120 words) with the takeaway at the end.",
}

_INSTRUCTION_TEMPLATE = """You are an elite explainer. Make the user's topic or text \
genuinely understood by ONE specific audience - not simplified into \
uselessness, not dumbed down, translated.

STEP 1 - AUDIENCE. Picture one concrete member of the audience and \
what they already know, use daily and care about.

STEP 2 - CALIBRATE five axes before writing:
- VOCABULARY: match their words. Zero unexplained jargon; when a technical \
term is unavoidable, define it inside parentheses in plain words.
- ANALOGIES: build on things they already know and like. One strong analogy \
carries the whole explanation - develop it, do not stack three weak ones.
- TONE: how they like being spoken to. Playful for children, crisp and \
professional for executives, warm for family. Never cringe, never condescending.
- DEPTH: as short as real understanding allows - but not shorter.
- FRAMING: open with what THIS audience cares about (impact and risk for a \
manager, how it feels/works for a designer, mechanism for an engineer).

STEP 3 - WRITE with this shape:
1. The core idea in ONE sentence.
2. The analogy, developed just enough to click.
3. Two to four short passages that rebuild the real concept on top of \
the analogy, each adding exactly one new piece.
4. A single-line takeaway they could repeat to someone else.

HARD RULES:
- Write in the same language as the user's input.
- No filler ("simply put", "as you know"), no hedging, no emoji spam.
- Concrete beats abstract; use a number only when it truly helps.
- If the input is a text rather than a topic, simplify THAT text while \
preserving every claim's meaning."""


def build_instruction(audience: str = DEFAULT_AUDIENCE) -> str:
    """Render the full system instruction for one audience."""
    return f"{_INSTRUCTION_TEMPLATE}\nThe audience is: {audience.strip() or DEFAULT_AUDIENCE}."


class Eli5Skill(LLMSkill):
    name = "explain_eli5"
    description = (
        "Explain a topic or simplify a text for a specific audience "
        "(e.g. 'age 5', 'my manager', 'a designer', 'grandma'). Calibrates "
        "vocabulary, analogies, tone, depth and framing; optional styles: "
        "'short', 'structured', 'story'."
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
            "style": {
                "type": "string",
                "enum": list(STYLES),
                "description": (
                    "Output shape: 'auto' (default), 'short' (<80 words), "
                    "'structured' (bullets), 'story' (tiny narrative)."
                ),
            },
        },
        "required": ["topic_or_text"],
    }

    async def run(
        self,
        topic_or_text: str,
        audience: str = DEFAULT_AUDIENCE,
        style: str = "auto",
    ) -> str:
        if not topic_or_text.strip():
            return "Give me something to explain."

        # Audience and style travel through extra_instruction so the
        # shared class-level instruction stays immutable and thread-safe.
        directives = [f"The audience is: {audience.strip() or DEFAULT_AUDIENCE}."]
        style_directive = STYLES.get(style.strip().lower(), STYLES["auto"])
        if style_directive:
            directives.append(style_directive)

        return await self.transform(topic_or_text, extra_instruction=" ".join(directives))
