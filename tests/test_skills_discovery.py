"""Tests for skill auto-discovery in chord.skills.

The point of these is the failure modes: a plugin that cannot be
imported or constructed should cost one tool, not the whole bot.
"""

from __future__ import annotations

import chord.skills as skills_package
from chord.config import Settings
from chord.skills import create_default_registry


def _settings() -> Settings:
    return Settings(_env_file=None, discord_token="t", openai_api_key="k")


def test_every_built_in_skill_is_discovered():
    registry = create_default_registry(_settings())

    assert "get_weather" in registry
    assert "get_price_history" in registry


def test_a_module_that_will_not_import_is_skipped_not_fatal(monkeypatch, caplog):
    """A skill needing an absent library must not stop the bot starting."""
    real_import = skills_package.importlib.import_module

    def explode(name, package=None):
        if name == ".price_history":
            raise ImportError("No module named 'PIL'")
        return real_import(name, package)

    monkeypatch.setattr(skills_package.importlib, "import_module", explode)

    with caplog.at_level("ERROR"):
        registry = create_default_registry(_settings())

    assert "get_price_history" not in registry
    assert "get_weather" in registry  # everything else still loaded
    assert "price_history" in caplog.text


def test_a_skill_that_will_not_construct_is_skipped_not_fatal(monkeypatch, caplog):
    def explode(skill_class, services):
        if skill_class.__name__ == "WeatherSkill":
            raise RuntimeError("bad config")
        return skill_class(
            **{
                name: services[name]
                for name, param in skills_package.inspect.signature(skill_class).parameters.items()
                if name != "self" and param.default is skills_package.inspect.Parameter.empty
            }
        )

    monkeypatch.setattr(skills_package, "_construct", explode)

    with caplog.at_level("ERROR"):
        registry = create_default_registry(_settings())

    assert "get_weather" not in registry
    assert len(registry) > 10
