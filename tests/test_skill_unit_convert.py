"""Tests for the unit conversion skill."""

from __future__ import annotations

import pytest

from chord.skills.registry import SkillRegistry
from chord.skills.unit_convert import (
    ConvertUnitsSkill,
    convert,
    format_value,
)


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(ConvertUnitsSkill())
    return registry


# -- Conversion math ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1, "km", "m", 1000.0),
        (5280, "ft", "mi", pytest.approx(1.0, abs=0.001)),
        (100, "kg", "lb", pytest.approx(220.462, rel=0.0001)),
        (2, "l", "ml", 2000.0),
        (100, "m2", "pyeong", pytest.approx(30.2489, rel=0.0001)),
        (36.0, "평", "m2", pytest.approx(119.0084, rel=0.0001)),
        (100, "kmh", "mph", pytest.approx(62.137, rel=0.0001)),
        (1, "gb", "mb", 1024.0),
        (500, "g", "oz", pytest.approx(17.637, rel=0.0001)),
    ],
)
def test_convert_categories(value, src, dst, expected):
    assert convert(value, src, dst) == expected


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (0, "c", "f", 32.0),
        (100, "c", "f", 212.0),
        (77, "f", "c", 25.0),
        (0, "k", "c", -273.15),
        (-40, "c", "f", -40.0),
    ],
)
def test_convert_temperature(value, src, dst, expected):
    assert convert(value, src, dst) == pytest.approx(expected, abs=0.0001)


def test_cross_category_raises():
    with pytest.raises(ValueError, match="same category"):
        convert(1, "km", "kg")


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown unit"):
        convert(1, "parsec", "km")


# -- Skill-level behavior ------------------------------------------------------------


async def test_skill_renders_clean_line():
    result = await _registry().execute(
        "convert_units", {"value": 5, "from_unit": "km", "to_unit": "mi"}
    )
    assert result == "5 km = 3.11 mi"


async def test_skill_temperature_rendering():
    result = await _registry().execute(
        "convert_units", {"value": 25, "from_unit": "celsius", "to_unit": "f"}
    )
    assert result == "25 c = 77 f"


async def test_bad_units_become_error_text():
    result = await _registry().execute(
        "convert_units", {"value": 1, "from_unit": "km", "to_unit": "warp"}
    )
    assert "Unknown unit 'warp'" in result


def test_format_value_trims_zeros():
    assert format_value(3.1068) == "3.11"
    assert format_value(1000.0) == "1,000"
    assert format_value(0.5) == "0.5"
    assert format_value(0) == "0"
