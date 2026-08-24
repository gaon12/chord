"""Tests for the random utility skill."""

from __future__ import annotations

import re

import pytest

from chord.skills.random_utils import RandomPickSkill, _split_items


@pytest.mark.parametrize(
    ("mode", "pattern"),
    [
        ({"mode": "dice"}, r"^Rolled 1d6: [1-6]$"),
        ({"mode": "dice", "sides": 20}, r"^Rolled 1d20: (?:[1-9]|1\d|20)$"),
        ({"mode": "dice", "count": 3}, r"^Rolled 3d6: \d, \d, \d \(total \d+\)$"),
        ({"mode": "coin"}, r"^Coin: (앞면|뒷면)$"),
        ({"mode": "coin", "count": 4}, r"^Coin: .+ - 앞면 \d/4$"),
        ({"mode": "number"}, r"^Number between 1 and 100: \d+$"),
        ({"mode": "number", "min_value": 5, "max_value": 5}, r"Number between 5 and 5: 5$"),
    ],
)
async def test_random_modes_shape(mode, pattern):
    result = await RandomPickSkill().run(**mode)
    assert re.fullmatch(pattern, result), result


async def test_pick_returns_one_of_items():
    result = await RandomPickSkill().run(mode="pick", items="피자, 김치, 버거")

    assert "Picked from 3 items:" in result
    chosen = result.rsplit(": ", 1)[-1]
    assert chosen in {"피자", "김치", "버거"}


async def test_shuffle_keeps_all_items():
    items = "a, b, c, d"
    result = await RandomPickSkill().run(mode="shuffle", items=items)

    body = result.removeprefix("Shuffled: ")
    assert sorted(body.split(", ")) == ["a", "b", "c", "d"]


async def test_pick_without_items_prompts_for_list():
    result = await RandomPickSkill().run(mode="pick")
    assert "comma-separated" in result


async def test_unknown_mode_reports_options():
    result = await RandomPickSkill().run(mode="lottery")
    assert "Unknown mode 'lottery'" in result
    assert "dice" in result


def test_split_items_strips_blanks():
    assert _split_items(" a , ,b ") == ["a", "b"]
