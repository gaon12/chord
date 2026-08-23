"""Built-in skills shipped with chord.

Adding a new skill is two steps:

1. Create a module here defining a :class:`chord.skills.base.Skill` subclass.
2. Add it to :func:`create_default_registry` below.
"""

from __future__ import annotations

from chord.skills.exchange_rate import ExchangeRateSkill
from chord.skills.registry import SkillRegistry
from chord.skills.weather import WeatherSkill

__all__ = ["SkillRegistry", "create_default_registry"]


def create_default_registry() -> SkillRegistry:
    """Build a registry containing every built-in skill.

    Each skill contributes exactly one line here, which keeps the list
    readable and makes additions/removals obvious in diffs.
    """
    registry = SkillRegistry()

    # -- Data skills (real-world lookups) ------------------------------------
    registry.register(WeatherSkill())
    registry.register(ExchangeRateSkill())

    # -- LLM-powered skills (summarize / translate / eli5) --------------------
    # (registered as they are implemented)

    return registry
