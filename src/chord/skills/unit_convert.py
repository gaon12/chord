"""Unit conversion skill (pure Python, no network needed).

Supports the categories that come up most in chat:

* length     : mm cm m km in ft yd mi
* mass       : mg g kg t oz lb
* temperature: C F K  (special formulas, not simple factors)
* volume     : ml l m3 gal(US) cup
* area       : m2 ha pyeong acre
* speed      : kmh ms mph knot
* data       : B KB MB GB TB  (binary, 1024-based)

Unit names are matched case-insensitively with a few friendly aliases
('celsius'/'섭씨' -> C, '평' -> pyeong, ...).
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills.base import Skill

#: Conversion factors relative to each category's base unit.
LENGTH = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
}
MASS = {
    "mg": 0.000001,
    "g": 0.001,
    "kg": 1.0,
    "t": 1000.0,
    "oz": 0.028349523125,
    "lb": 0.45359237,
}
VOLUME = {
    "ml": 0.001,
    "l": 1.0,
    "m3": 1000.0,
    "gal": 3.785411784,  # US gallon
    "cup": 0.24,  # US legal-ish cup used in recipes (240 ml)
}
AREA = {
    "m2": 1.0,
    "ha": 10000.0,
    "pyeong": 3.30579,  # 평
    "acre": 4046.8564224,
}
SPEED = {
    "kmh": 1.0,  # km/h
    "ms": 3.6,  # m/s expressed in km/h
    "mph": 1.609344,
    "knot": 1.852,
}
DATA = {
    "b": 1.0,
    "kb": 1024.0,
    "mb": 1024.0**2,
    "gb": 1024.0**3,
    "tb": 1024.0**4,
}

CATEGORIES: dict[str, dict[str, float]] = {
    "length": LENGTH,
    "mass": MASS,
    "volume": VOLUME,
    "area": AREA,
    "speed": SPEED,
    "data": DATA,
}

#: Friendly aliases -> canonical unit names.
ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "millimeter": "mm",
    "kilometer": "km",
    "kilometers": "km",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "yard": "yd",
    "yards": "yd",
    "mile": "mi",
    "miles": "mi",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "ton": "t",
    "tonne": "t",
    "ounce": "oz",
    "ounces": "oz",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "milliliter": "ml",
    "gallon": "gal",
    "gallons": "gal",
    "celsius": "c",
    "섭씨": "c",
    "fahrenheit": "f",
    "화씨": "f",
    "kelvin": "k",
    "켈빈": "k",
    "평": "pyeong",
    "km/h": "kmh",
    "kph": "kmh",
    "m/s": "ms",
    "knots": "knot",
    "bytes": "b",
    "byte": "b",
    "kilobyte": "kb",
    "megabyte": "mb",
    "gigabyte": "gb",
    "terabyte": "tb",
}


def canonical_unit(name: str) -> str:
    """Normalize a unit name; raises ValueError for unknown units."""
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    for table in (*CATEGORIES.values(), {"c": 0, "f": 0, "k": 0}):
        if key in table:
            return key
    raise ValueError(f"Unknown unit '{name}'.")


def find_category(*units: str) -> str:
    """Find the shared category of two units; None-compatible error otherwise."""
    for category, table in CATEGORIES.items():
        if all(unit in table for unit in units):
            return category
    raise ValueError("Those units are not in the same category (or are unknown).")


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperatures via Celsius as the hub."""
    celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[from_unit]
    return {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[to_unit]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units of the same category."""
    from_c, to_c = canonical_unit(from_unit), canonical_unit(to_unit)
    if from_c in ("c", "f", "k") and to_c in ("c", "f", "k"):
        return convert_temperature(value, from_c, to_c)

    category = find_category(from_c, to_c)
    factors = CATEGORIES[category]
    return value * factors[from_c] / factors[to_c]


def format_value(value: float) -> str:
    """Trim trailing zeros but keep small numbers readable."""
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.6f}".rstrip("0").rstrip(".")


class ConvertUnitsSkill(Skill):
    name = "convert_units"
    description = (
        "Convert a value between units: length, mass, temperature, "
        "volume, area (incl. pyeong), speed and data size."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert.",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit, e.g. 'km', 'lb', 'c', 'pyeong'.",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit, e.g. 'mi', 'kg', 'f', 'm2'.",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    async def run(self, value: float, from_unit: str, to_unit: str) -> str:
        try:
            result = convert(float(value), from_unit, to_unit)
        except ValueError as exc:
            raise SkillInputError(str(exc)) from exc
        # Echo canonical unit names so 'celsius' reads back as 'c'.
        from_c = canonical_unit(from_unit)
        to_c = canonical_unit(to_unit)
        return f"{format_value(float(value))} {from_c} = {format_value(result)} {to_c}"


class SkillInputError(ValueError):
    """Bad user input; the registry renders it as readable text."""
