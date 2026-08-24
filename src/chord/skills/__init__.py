"""Built-in skills shipped with chord.

Adding a new skill is two steps:

1. Create a module here defining a :class:`chord.skills.base.Skill` subclass.
2. Add it to :func:`create_default_registry` below.
"""

from __future__ import annotations

from chord.config import Settings
from chord.llm import LLMService
from chord.skills.air_quality import AirQualitySkill
from chord.skills.datetime_info import ConvertTimezoneSkill, CurrentDatetimeSkill
from chord.skills.delivery import DeliverySkill
from chord.skills.exchange_rate import ExchangeRateSkill
from chord.skills.flight import FlightSkill
from chord.skills.map import FindPlacesSkill, GetDirectionsSkill
from chord.skills.registry import SkillRegistry
from chord.skills.stock import StockPriceSkill
from chord.skills.summarize import SummarizeSkill
from chord.skills.translate import TranslateSkill
from chord.skills.unit_convert import ConvertUnitsSkill
from chord.skills.url_safety import CheckUrlSafetySkill
from chord.skills.url_shortener import ExpandUrlSkill, ShortenUrlSkill
from chord.skills.weather import WeatherSkill
from chord.skills.web_search import WebSearchSkill

__all__ = ["SkillRegistry", "create_default_registry"]


def create_default_registry(settings: Settings) -> SkillRegistry:
    """Build a registry containing every built-in skill.

    Each skill contributes exactly one line here, which keeps the list
    readable and makes additions/removals obvious in diffs.
    """
    registry = SkillRegistry()

    # -- Data skills (real-world lookups) ------------------------------------
    registry.register(WeatherSkill(settings))
    registry.register(ExchangeRateSkill())
    registry.register(StockPriceSkill())
    registry.register(AirQualitySkill(settings))
    registry.register(DeliverySkill(settings))
    registry.register(FlightSkill(settings))
    registry.register(ShortenUrlSkill(settings))
    registry.register(ExpandUrlSkill(settings))
    registry.register(CheckUrlSafetySkill(settings))
    registry.register(FindPlacesSkill(settings))
    registry.register(GetDirectionsSkill(settings))
    registry.register(CurrentDatetimeSkill())
    registry.register(ConvertTimezoneSkill())
    registry.register(ConvertUnitsSkill())
    registry.register(WebSearchSkill())

    # -- LLM-powered skills (summarize / translate / eli5) --------------------
    # They reuse the same chat model as the main conversation, so a
    # single LLMService instance is shared across all of them.
    llm = LLMService(settings)
    registry.register(SummarizeSkill(llm))
    registry.register(TranslateSkill(llm))

    return registry

    # -- LLM-powered skills (summarize / translate / eli5) --------------------
    # (registered as they are implemented)

    return registry
