"""Built-in skills shipped with chord.

Skills are **plugins**: any public module in this package that defines
one or more ``Skill`` subclasses whose names end in ``Skill`` is
discovered and registered automatically by
:func:`create_default_registry` - no manual registration line needed.

Constructor dependencies are injected by parameter name:

* a ``settings`` parameter receives the shared :class:`Settings`
* an ``llm`` parameter receives one shared :class:`LLMService`

so a data skill is just ``def __init__(self, settings)`` (or no
``__init__`` at all) and an LLM-backed skill is
``def __init__(self, llm)``.

Private modules (leading underscore) are helpers, not plugins.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

from chord.config import Settings
from chord.llm import LLMService
from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

__all__ = ["SkillRegistry", "create_default_registry"]


def _iter_skill_classes():
    """Yield every concrete plugin class found in this package."""
    package = importlib.import_module(__package__)
    for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        name = module_info.name
        if name.startswith("_") or name in {"registry", "base"}:
            continue  # private helpers / infrastructure
        try:
            module = importlib.import_module(f".{name}", __package__)
        except Exception:  # noqa: BLE001 - one broken plugin != broken bot
            # A skill that needs a third-party library (Pillow, for the
            # charts) must not take the whole bot down with it when that
            # library is missing from the environment.
            logger.exception("Skipping skill module %s (import failed).", name)
            continue
        for attr_name, attr_value in vars(module).items():
            if (
                inspect.isclass(attr_value)
                and issubclass(attr_value, Skill)
                and attr_value is not Skill
                and attr_name.endswith("Skill")
                and attr_value.__module__ == module.__name__
                and not inspect.isabstract(attr_value)
            ):
                yield f"{name}.{attr_name}", attr_value


def _construct(skill_class, services: dict):
    """Instantiate a skill, injecting services by constructor param name.

    Uses ``inspect.signature(skill_class)`` so inherited no-op ``__init__``
    methods resolve to zero arguments instead of object's ``*args``.
    """
    signature = inspect.signature(skill_class)
    kwargs = {}
    for param_name, param in signature.parameters.items():
        if param_name == "self" or param.default is not inspect.Parameter.empty:
            continue
        if param_name not in services:
            raise ValueError(
                f"{skill_class.__name__} needs a '{param_name}' argument; "
                f"available services: {sorted(services)}."
            )
        kwargs[param_name] = services[param_name]
    return skill_class(**kwargs)


def create_default_registry(settings: Settings) -> SkillRegistry:
    """Discover every built-in skill and return a ready registry.

    Discovery rules keep additions trivial - drop a module with a
    ``SomethingSkill`` class here and it just works:

    * public modules only (no leading underscore)
    * classes must subclass :class:`chord.skills.base.Skill` and end in
      ``Skill``
    * ``settings`` / ``llm`` constructor parameters are injected
      automatically (one shared LLMService instance)

    A module that will not import, or a skill that will not construct,
    is logged and skipped. Losing one tool is a degraded bot; refusing
    to start is no bot at all.
    """
    registry = SkillRegistry()
    llm = LLMService(settings)
    services = {"settings": settings, "llm": llm}

    for origin, skill_class in _iter_skill_classes():
        try:
            registry.register(_construct(skill_class, services))
        except Exception:  # noqa: BLE001 - one broken plugin != broken bot
            logger.exception("Skipping skill %s (construction failed).", origin)

    logger.info("Discovered %d built-in skills.", len(registry))
    return registry
