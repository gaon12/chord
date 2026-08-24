"""Random utility skill - dice, coin flips, picks and shuffles.

Pure Python, no network. One tool with a ``mode`` parameter keeps the
LLM-facing surface small:

* ``dice``    - roll ``count`` dice with ``sides`` sides (default d6)
* ``coin``    - flip a coin ``count`` times
* ``number``  - random integer between ``min_value`` and ``max_value``
* ``pick``    - choose one entry from a comma-separated list
* ``shuffle`` - return the list in random order
"""

from __future__ import annotations

import random
from typing import ClassVar

from chord.skills.base import Skill


class RandomPickSkill(Skill):
    name = "random_pick"
    description = (
        "Random utilities: dice rolls, coin flips, random numbers, "
        "picking one item from a list, or shuffling a list."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["dice", "coin", "number", "pick", "shuffle"],
                "description": (
                    "'dice' roll dice, 'coin' flip coins, 'number' pick an "
                    "integer range, 'pick' choose one item, 'shuffle' "
                    "reorder items."
                ),
            },
            "items": {
                "type": "string",
                "description": (
                    "Comma-separated list for 'pick'/'shuffle', e.g. 'pizza, kimchi, burger'."
                ),
            },
            "sides": {
                "type": "integer",
                "description": "Dice sides for 'dice' mode (default 6).",
            },
            "count": {
                "type": "integer",
                "description": "How many dice/coins (default 1, max 20).",
            },
            "min_value": {"type": "integer", "description": "Min for 'number' mode."},
            "max_value": {"type": "integer", "description": "Max for 'number' mode."},
        },
        "required": ["mode"],
    }

    async def run(
        self,
        mode: str,
        items: str = "",
        sides: int = 6,
        count: int = 1,
        min_value: int = 1,
        max_value: int = 100,
    ) -> str:
        mode = mode.strip().lower()
        count = max(1, min(int(count), 20))

        if mode == "dice":
            return _dice(sides, count)
        if mode == "coin":
            return _coin(count)
        if mode == "number":
            return _number(min_value, max_value)
        if mode in ("pick", "shuffle"):
            return _list_op(mode, items)
        return f"Unknown mode '{mode}'. Use dice, coin, number, pick or shuffle."


def _clamp_sides(sides: int) -> int:
    return max(2, min(int(sides), 1000))


def _dice(sides: int, count: int) -> str:
    sides = _clamp_sides(sides)
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) if count > 1 else None
    detail = f" (total {total})" if total is not None else ""
    return f"Rolled {count}d{sides}: {', '.join(map(str, rolls))}{detail}"


def _coin(count: int) -> str:
    flips = [random.choice(["앞면", "뒷면"]) for _ in range(count)]
    heads = flips.count("앞면")
    detail = f" - 앞면 {heads}/{len(flips)}" if len(flips) > 1 else ""
    return f"Coin: {', '.join(flips)}{detail}"


def _number(min_value: int, max_value: int) -> str:
    low, high = sorted((int(min_value), int(max_value)))
    return f"Number between {low} and {high}: {random.randint(low, high)}"


def _split_items(items: str) -> list[str]:
    return [part.strip() for part in items.split(",") if part.strip()]


def _list_op(mode: str, items: str) -> str:
    values = _split_items(items)
    if not values:
        return "Please provide comma-separated items."
    if mode == "pick":
        chosen = random.choice(values)
        return f"Picked from {len(values)} items: {chosen}"
    shuffled = values[:]
    random.shuffle(shuffled)
    return "Shuffled: " + ", ".join(shuffled)
